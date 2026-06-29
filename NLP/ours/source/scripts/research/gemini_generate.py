#!/usr/bin/env python3
"""Strong-generator answer generation with Gemini over T2-RAGBench retrieval top-k.

For each query we collect, in one pass, everything the reliability study needs:
  * RAW answer (temperature 0)            -> the system's answer under a strong LLM
  * verbalized self-confidence (0..1)     -> an LLM-confidence baseline reviewers ask for
  * k sampled answers (temperature > 0)   -> the self-consistency baseline

The output JSONL is rebuilt against the same retrieval records by the reliability
eval, which reconstructs the fact ledger and scores CPR / value-only / self-consistency /
verbalized confidence head to head. Resumable at the record level + Gemini disk cache,
so the run survives quota throttling and restarts without re-spending calls.

Usage:
  PYTHONPATH=src python scripts/research/gemini_generate.py --dataset finqa \
      --sample 300 --k-sc 5 --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.utils.gemini_client import GeminiClient

RETR = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}

RAW_SYSTEM = (
    "You are a precise financial QA model. Use only the retrieved financial tables and "
    "text. Return exactly one line in the form 'Answer: <number>'. No explanation. "
    "For percentage questions, prefer the decimal fraction when the evidence supports it."
)
CONF_SYSTEM = (
    "You are a careful financial answer verifier. Given a question, the evidence, and a "
    "candidate answer, judge the probability that the candidate is numerically correct. "
    "Return exactly one line 'Confidence: <p>' where p is a probability in [0,1]."
)


def _truncate(text: str, n: int) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "\n[TRUNCATED]"


def raw_evidence_block(retrieved: list[dict], k: int = 3, max_doc_chars: int = 1400) -> str:
    blocks = []
    for i, doc in enumerate(retrieved[:k], 1):
        meta = doc.get("meta") or doc.get("metadata") or {}
        title = (f"[Doc {i}] id={doc.get('context_id') or doc.get('id','')} "
                 f"company={meta.get('company_name','')} year={meta.get('report_year','')}")
        body = doc.get("table") or doc.get("page_content") or doc.get("context") or ""
        blocks.append(f"{title}\n{_truncate(body, max_doc_chars)}")
    return "\n\n".join(blocks)


def build_prompt(question: str, evidence: str, meta: dict) -> str:
    head = []
    if meta.get("company_name"):
        line = f"Company: {meta['company_name']}"
        if meta.get("report_year"):
            line += f" | Reporting year: {meta['report_year']}"
        head.append(line)
    head.append(evidence.strip())
    head.append(f"Question: {question}")
    head.append("Return one line only: Answer: <number>")
    return "\n\n".join(head)


def build_conf_prompt(question: str, evidence: str, candidate: str) -> str:
    return "\n\n".join([
        evidence.strip(),
        f"Question: {question}",
        f"Candidate answer: {candidate}",
        "Return one line only: Confidence: <p in [0,1]>",
    ])


_CONF_RE = re.compile(r"(?:confidence|prob[a-z]*)\s*[:=]?\s*([01](?:\.\d+)?|\.\d+)", re.I)


def parse_conf(text: str) -> float | None:
    m = _CONF_RE.search(text or "")
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            return None
    # fall back: first float in [0,1]
    for tok in re.findall(r"[01]?\.\d+|[01]\b", text or ""):
        try:
            v = float(tok)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return None


def modal_answer(samples: list[str], rel_tol: float = 1e-2) -> tuple[float | None, float]:
    """Return (modal numeric answer, agreement fraction) over sampled answers."""
    vals = [extract_final_number(s) for s in samples]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    best_v, best_count = None, 0
    for i, v in enumerate(vals):
        c = sum(1 for u in vals if abs(u - v) <= rel_tol * max(1.0, abs(v)))
        if c > best_count:
            best_v, best_count = v, c
    return best_v, best_count / len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(RETR))
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--k-sc", type=int, default=5, help="self-consistency samples")
    ap.add_argument("--sc-temp", type=float, default=0.7)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--out", default="outputs/research/gemini_gen/{ds}_predictions.jsonl")
    ap.add_argument("--max-doc-chars", type=int, default=1400)
    args = ap.parse_args()

    retr_path = Path(RETR[args.dataset])
    records = [json.loads(l) for l in retr_path.read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.sample > 0:
        records = records[: args.sample]

    out_path = Path(args.out.format(ds=args.dataset))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                try:
                    done_ids.add(json.loads(l)["query_id"])
                except Exception:
                    pass
    print(f"[{args.dataset}] {len(records)} target queries, {len(done_ids)} already done", flush=True)

    cli = GeminiClient(model=args.model)
    t0 = time.time()
    n_correct = 0
    n_total = 0
    with out_path.open("a") as fh:
        for ri, rec in enumerate(records):
            qid = rec.get("query_id", ri)
            if qid in done_ids:
                continue
            question = rec.get("raw_question") or rec.get("query") or ""
            meta = rec.get("query_meta") or {}
            gold = rec.get("gold")
            evidence = raw_evidence_block(rec.get("retrieved", []), max_doc_chars=args.max_doc_chars)
            prompt = build_prompt(question, evidence, meta)

            raw = cli.generate(prompt, system=RAW_SYSTEM, temperature=0.0, max_tokens=48, idx=0)
            # verbalized confidence
            conf_txt = cli.generate(build_conf_prompt(question, evidence, raw),
                                    system=CONF_SYSTEM, temperature=0.0, max_tokens=16, idx=0)
            vconf = parse_conf(conf_txt)
            # self-consistency samples
            sc = []
            for j in range(args.k_sc):
                sc.append(cli.generate(prompt, system=RAW_SYSTEM, temperature=args.sc_temp,
                                       max_tokens=48, idx=j + 1))
            sc_val, sc_agree = modal_answer(sc)

            correct = bool(number_match(raw, gold)) if gold is not None else None
            if correct is not None:
                n_correct += int(correct); n_total += 1

            fh.write(json.dumps({
                "query_id": qid,
                "question": question,
                "query": rec.get("query", question),
                "query_meta": meta,
                "gold": gold,
                "raw_pred": raw,
                "verbalized_conf": vconf,
                "sc_samples": sc,
                "sc_modal": sc_val,
                "sc_agreement": sc_agree,
                "raw_correct": correct,
            }) + "\n")
            fh.flush()

            if (ri + 1) % 25 == 0:
                acc = n_correct / max(n_total, 1)
                print(f"  [{args.dataset}] {ri+1}/{len(records)}  acc={acc:.3f}  "
                      f"{cli.stats()}  {time.time()-t0:.0f}s", flush=True)

    acc = n_correct / max(n_total, 1)
    print(f"[{args.dataset}] DONE acc(raw)={acc:.4f} n={n_total} {cli.stats()} "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
