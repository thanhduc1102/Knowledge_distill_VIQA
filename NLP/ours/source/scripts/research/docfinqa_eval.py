#!/usr/bin/env python3
"""DocFinQA long-document end-to-end slice (generalization to 123K-word filings).

Pipeline per question: chunk the full SEC filing -> BM25 retrieve top-k chunks -> build a
Fact Ledger from them -> Gemini answers from the retrieved chunks -> CPR / value-only /
verbalized reliability + Number-Match. Tests whether structure-grounded reliability holds
when the evidence is buried in a very long document (the gap DocFinQA targets).

Data acquired by fetch_docfinqa.py (outputs/data/docfinqa/test.jsonl).
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.ledger.extract import extract_ledger
from gsr_cacl.ledger.numeric import number_match, extract_numbers, parse_financial_number
from gsr_cacl.generation.verifier import verify, extract_final_number
from gsr_cacl.research.cpr_verifier import verify_cpr
from gsr_cacl.utils.gemini_client import GeminiClient

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "is", "are", "what", "how",
         "much", "many", "did", "was", "year", "during", "between", "change", "total"}
RAW_SYSTEM = ("You are a precise financial QA model. Use only the retrieved excerpts from the "
              "financial report. Return exactly one line: Answer: <number>.")
CONF_SYSTEM = ("Given a question, evidence and a candidate answer, return one line "
               "'Confidence: <p in [0,1]>' = probability the candidate is correct.")


def _toks(t):
    return [x for x in _TOK.findall((t or "").lower()) if x not in _STOP and len(x) > 1]


def chunk_doc(text, size=700, overlap=120):
    lines = str(text).splitlines()
    chunks, buf, cur = [], [], 0
    for ln in lines:
        buf.append(ln); cur += len(ln) + 1
        if cur >= size:
            chunks.append("\n".join(buf))
            keep = "\n".join(buf)[-overlap:]
            buf, cur = ([keep] if keep else []), len(keep)
    if buf:
        chunks.append("\n".join(buf))
    return [c for c in chunks if c.strip()]


def parse_conf(t):
    m = re.search(r"(?:confidence|prob[a-z]*)\s*[:=]?\s*([01](?:\.\d+)?|\.\d+)", t or "", re.I)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            return None
    return None


def _auroc(scores, labels):
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i]); ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = a
        i = j + 1
    return (sum(ranks[i] for i in range(len(scores)) if labels[i]) - pos * (pos + 1) / 2) / (pos * neg)


def run(path, n, k, client):
    from rank_bm25 import BM25Okapi
    rows = []
    ev_recall = 0; ev_tot = 0
    with open(path) as fh:
        for li, line in enumerate(fh):
            if li >= n:
                break
            r = json.loads(line)
            q = r["Question"]; gold = r.get("Answer")
            chunks = chunk_doc(r["Context"])
            if len(chunks) < 3:
                continue
            bm = BM25Okapi([_toks(c) for c in chunks])
            sc = bm.get_scores(_toks(q))
            top = np.argsort(-sc)[:k]
            evidence = "\n---\n".join(chunks[i] for i in top)
            # evidence recall: do the Program's operand numbers appear in the top-k chunks?
            prog_nums = extract_numbers(str(r.get("Program", "")))
            if prog_nums:
                ev_tot += 1
                ev_text_nums = set(round(x, 4) for x in extract_numbers(evidence))
                hit = sum(1 for pn in prog_nums if any(abs(pn - en) <= 1e-2 * max(1, abs(pn)) for en in ev_text_nums))
                ev_recall += int(hit >= max(1, len(prog_nums) - 1))
            prompt = f"{evidence}\n\nQuestion: {q}\nReturn one line only: Answer: <number>"
            raw = client.generate(prompt, system=RAW_SYSTEM, temperature=0.0, max_tokens=48, idx=0)
            vconf = parse_conf(client.generate(
                f"Evidence:\n{evidence[:4000]}\n\nQuestion: {q}\nCandidate: {raw}\nReturn: Confidence: <p>",
                system=CONF_SYSTEM, temperature=0.0, max_tokens=16, idx=0))
            led = extract_ledger(context=evidence, doc_id=f"doc{li}", meta={})
            vr = verify(raw, led, q, gold=gold)
            cpr = verify_cpr(raw, led, q, gold=gold)
            correct = bool(number_match(raw, gold)) if gold is not None else False
            rows.append({"correct": correct,
                         "value_only": 1.0 if vr.grounded else (0.85 if vr.derivable else (0.1 if vr.pred_value is not None else 0.0)),
                         "cpr": cpr.confidence,
                         "verbalized": 0.5 if vconf is None else vconf})
    n_ = len(rows)
    out = {"n": n_, "base_accuracy": round(sum(r["correct"] for r in rows) / max(n_, 1), 4),
           "evidence_recall_topk": round(ev_recall / max(ev_tot, 1), 4),
           "auroc": {}}
    for s in ("value_only", "cpr", "verbalized"):
        out["auroc"][s] = round(_auroc([r[s] for r in rows], [r["correct"] for r in rows]), 4)
    # simple cpr+verbalized mean
    for r in rows:
        r["cpr_verb"] = 0.5 * (min(1, r["cpr"]) + r["verbalized"])
    out["auroc"]["cpr+verbalized"] = round(_auroc([r["cpr_verb"] for r in rows], [r["correct"] for r in rows]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="outputs/data/docfinqa/test.jsonl")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default="outputs/research/docfinqa/report.json")
    args = ap.parse_args()
    client = GeminiClient()
    t0 = time.time()
    res = run(args.path, args.n, args.k, client)
    res["seconds"] = round(time.time() - t0, 1)
    res["client"] = client.stats()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
