#!/usr/bin/env python3
"""Generalization test: does Concept-Period-Role structure grounding transfer to an
out-of-distribution benchmark (FinanceBench, real SEC 10-K filings)?

Setup isolates *verification* from retrieval: each question is answered from its GOLD
evidence text (so retrieval noise is removed), a 7B LLM produces the answer, and we test
whether CPR confidence separates correct from incorrect numeric answers better than the
legacy value-only verifier — exactly as on T2-RAGBench. If the structure-grounded
auditability is real (not a T2-RAGBench artifact), the separation/AUROC gain should hold.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from datasets import load_dataset

from gsr_cacl.generation.verifier import verify, extract_final_number
from gsr_cacl.ledger.extract import extract_ledger
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.research.cpr_verifier import verify_cpr


SYSTEM = (
    "You are a precise financial QA model. Use only the provided 10-K evidence. "
    "Return exactly one line: Answer: <number>. No explanation."
)


def _gold_number(answer: str):
    return extract_final_number(answer)


def load_numeric_financebench(sample: int):
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = []
    for r in ds:
        ev = r.get("evidence") or []
        text = "\n\n".join((e.get("evidence_text") or "") for e in ev).strip()
        gold = _gold_number(str(r.get("answer") or ""))
        if not text or gold is None:
            continue
        rows.append({
            "id": r.get("financebench_id"),
            "question": str(r.get("question") or ""),
            "company": str(r.get("company") or ""),
            "period": str(r.get("doc_period") or ""),
            "evidence": text,
            "gold_answer": str(r.get("answer") or ""),
            "gold": [str(gold)],
        })
    if sample:
        rows = rows[:sample]
    return rows


def _auroc(scores, labels):
    pos = sum(1 for l in labels if l); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _risk_coverage(confs, correct):
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    n = len(order); out = []
    for cov in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(round(n * cov)))
        acc = sum(correct[i] for i in order[:k]) / k
        out.append({"coverage": cov, "accuracy": round(acc, 4)})
    return {"curve": out, "selective_acc_auc": round(sum(c["accuracy"] for c in out) / len(out), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--max-evidence-chars", type=int, default=3500)
    ap.add_argument("--out", default="outputs/research/financebench_cpr")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_numeric_financebench(args.sample)
    print(f"FinanceBench numeric-answer rows: {len(rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="auto" if args.device == "auto" else {"": args.device},
    )
    model.eval()
    in_dev = next(model.parameters()).device

    def generate(prompts):
        outs = []
        for s in range(0, len(prompts), args.batch_size):
            batch = prompts[s:s + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_input_tokens).to(in_dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            plen = enc["input_ids"].shape[1]
            for r in gen:
                txt = tok.decode(r[plen:], skip_special_tokens=True)
                m = re.search(r"answer:\s*(-?[\d,]*\.?\d+%?)", txt, re.I)
                outs.append("Answer: " + (m.group(1).replace(",", "") if m else txt.strip().splitlines()[0] if txt.strip() else ""))
        return outs

    prompts = []
    for r in rows:
        ev = r["evidence"][:args.max_evidence_chars]
        user = f"Company: {r['company']} | Year: {r['period']}\n\n{ev}\n\nQuestion: {r['question']}\nReturn one line only: Answer: <number>"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))

    t0 = time.time()
    preds = generate(prompts)
    elapsed = time.time() - t0

    recs = []
    for r, pred in zip(rows, preds):
        ledger = extract_ledger(context=r["evidence"], doc_id=str(r["id"]),
                                meta={"company_name": r["company"], "report_year": r["period"]})
        gold = r["gold"]
        pv = extract_final_number(pred)
        correct = bool(pv is not None and number_match(pv, gold))
        vr = verify(pred, ledger, r["question"], gold=gold)
        cpr = verify_cpr(pred, ledger, r["question"], gold=gold)
        recs.append({
            "id": r["id"], "question": r["question"], "gold_answer": r["gold_answer"],
            "prediction": pred, "correct": correct,
            "legacy_supported": bool(vr.grounded or vr.derivable),
            "legacy_conf": 1.0 if vr.grounded else (0.85 if vr.derivable else (0.1 if pv is not None else 0.0)),
            "cpr_supported": cpr.supported, "cpr_conf": cpr.confidence, "cpr_level": cpr.level,
            "n_facts": len(ledger.numeric_facts()),
        })

    n = len(recs)
    correct = [r["correct"] for r in recs]
    raw_acc = sum(correct) / max(n, 1)

    def sep(key):
        sup = [r for r in recs if r[key]]; uns = [r for r in recs if not r[key]]
        a_s = sum(r["correct"] for r in sup) / max(len(sup), 1)
        a_u = sum(r["correct"] for r in uns) / max(len(uns), 1)
        return {"supported_n": len(sup), "acc_supported": round(a_s, 4),
                "acc_unsupported": round(a_u, 4), "separation": round(a_s - a_u, 4),
                "supported_but_wrong": sum(1 for r in sup if not r["correct"])}

    legacy = sep("legacy_supported"); cpr = sep("cpr_supported")
    legacy["auroc"] = round(_auroc([r["legacy_conf"] for r in recs], correct), 4)
    cpr["auroc"] = round(_auroc([r["cpr_conf"] for r in recs], correct), 4)
    legacy["risk_coverage"] = _risk_coverage([r["legacy_conf"] for r in recs], correct)
    cpr["risk_coverage"] = _risk_coverage([r["cpr_conf"] for r in recs], correct)
    lsw = [r for r in recs if r["legacy_supported"] and not r["correct"]]
    downgraded = [r for r in lsw if not r["cpr_supported"]]

    summary = {
        "benchmark": "financebench", "n": n, "raw_accuracy": round(raw_acc, 4),
        "model": args.model, "seconds": round(elapsed, 1),
        "legacy_value_only": legacy, "cpr": cpr,
        "over_firing_reduction": {
            "legacy_supported_wrong": len(lsw),
            "cpr_downgraded": len(downgraded),
            "downgrade_precision": round(len(downgraded) / max(len(lsw), 1), 4),
        },
    }
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "predictions.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
