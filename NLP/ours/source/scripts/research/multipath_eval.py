#!/usr/bin/env python3
"""Multi-path agreement calibration: NM and coverage vs number of agreeing paths.

Runs the 3-path deterministic ensemble (heuristic × coordinate × ontology) on the GOLD doc
(grounding isolated) and reports Number-Match precision when gating on vote count. The headline
claim: k-of-N path agreement gives calibrated, high-precision, abstaining symbolic answering.

Usage:  PYTHONPATH=src python scripts/research/multipath_eval.py --dataset finqa
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.multipath import multipath_answer
from gsr_cacl.ledger.select import infer_task_type
from gsr_cacl.ledger.numeric import number_match

DEFAULTS = {
    "finqa": "outputs/strong_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/strong_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/strong_retrieval/tat-dqa/retrieval_top3.jsonl",
}
TASKS = {"lookup", "difference", "percent_change", "ratio", "sum", "average"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default="outputs/research/multipath")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.input or DEFAULTS[args.dataset]) if l.strip()]

    rows = []   # (votes, n_paths, nm)
    n_q = 0
    for rec in recs:
        gt = str(rec.get("ground_truth_id") or "")
        gdoc = next((d for d in (rec.get("retrieved") or [])
                     if str(d.get("context_id") or d.get("id")) == gt), None)
        if gdoc is None:
            continue
        ledger = extract_ledger_from_table(gdoc.get("table") or gdoc.get("page_content", ""),
                                           doc_id=gt, meta=gdoc.get("meta") or {})
        if not ledger.numeric_facts():
            continue
        task = infer_task_type(rec["query"])
        if task not in TASKS:
            continue
        n_q += 1
        res = multipath_answer(rec["query"], ledger, task)
        if res.answer is None:
            continue
        nm = int(number_match(res.answer, rec.get("gold")))
        rows.append((res.votes, res.n_paths, nm))

    rows = np.array(rows, dtype=float) if rows else np.zeros((0, 3))
    print(f"\n=== Multi-path agreement calibration — {args.dataset} "
          f"(answerable arithmetic Qs with gold doc = {n_q}) ===")
    print(f"{'gate':<16}{'coverage(of n_q)':>18}{'NM|gated':>12}{'n':>7}")
    out = {"dataset": args.dataset, "n_q": n_q, "gates": {}}
    for k in (1, 2, 3):
        sel = rows[rows[:, 0] >= k] if len(rows) else rows
        cov = len(sel) / max(n_q, 1)
        nm = sel[:, 2].mean() if len(sel) else 0.0
        print(f"votes>={k:<10}{cov:>18.3f}{nm:>12.3f}{len(sel):>7}")
        out["gates"][f">={k}"] = {"coverage": round(cov, 4), "nm": round(float(nm), 4), "n": int(len(sel))}
    print("   exact vote-count buckets:")
    for k in (1, 2, 3):
        sel = rows[rows[:, 0] == k] if len(rows) else rows
        if len(sel):
            print(f"      votes={k}: n={len(sel):5}  NM={sel[:,2].mean():.3f}")
    # overall symbolic NM if we always answer (votes>=1) for context
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset+'.json')}")


if __name__ == "__main__":
    main()
