#!/usr/bin/env python3
"""Weakly learned coordinate grounding evaluation on gold documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.ledger.coordinate import coordinate_answer
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.learned_coordinate import train_weak
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.ledger.select import calculation_plan, infer_task_type, select_facts

DEFAULTS = {
    "finqa": "outputs/strong_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/strong_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/strong_retrieval/tat-dqa/retrieval_top3.jsonl",
}


def _gold_examples(path: str):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for rec in recs:
        gt = str(rec.get("ground_truth_id") or "")
        gdoc = next((d for d in (rec.get("retrieved") or [])
                     if str(d.get("context_id") or d.get("id")) == gt), None)
        if gdoc is None:
            continue
        led = extract_ledger_from_table(gdoc.get("table") or gdoc.get("page_content", ""),
                                        doc_id=gt, meta=gdoc.get("meta") or {})
        if led.numeric_facts():
            out.append((rec["query"], led, rec.get("gold")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/learned_coordinate")
    args = ap.parse_args()

    examples = _gold_examples(args.input or DEFAULTS[args.dataset])
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(examples))
    cut = len(idx) // 2
    train = [examples[i] for i in idx[:cut]]
    test = [examples[i] for i in idx[cut:]]
    model = train_weak(train, seed=args.seed)

    counts = {"n": 0, "heuristic": 0, "coord": 0, "learned": 0, "either": 0}
    by_task: dict[str, dict[str, int]] = {}
    for q, led, gold in test:
        task = infer_task_type(q)
        if task not in {"lookup", "difference", "percent_change"}:
            continue
        by_task.setdefault(task, {"n": 0, "heuristic": 0, "coord": 0, "learned": 0, "either": 0})
        cur = calculation_plan(q, select_facts(q, [led], top_n=6))
        co = coordinate_answer(q, led, task)
        lc = model.answer(q, led, task)
        ok_cur = cur.get("answer") is not None and number_match(cur["answer"], gold)
        ok_co = co is not None and number_match(co["answer"], gold)
        ok_lc = lc is not None and number_match(lc["answer"], gold)
        for bucket in (counts, by_task[task]):
            bucket["n"] += 1
            bucket["heuristic"] += int(ok_cur)
            bucket["coord"] += int(ok_co)
            bucket["learned"] += int(ok_lc)
            bucket["either"] += int(ok_cur or ok_co or ok_lc)

    def rates(d):
        n = max(d["n"], 1)
        return {k: (round(d[k] / n, 4) if k != "n" else d[k]) for k in d}

    out = {
        "dataset": args.dataset,
        "n_train": len(train),
        "n_test_total": len(test),
        "weights": [round(float(x), 4) for x in model.weights.tolist()],
        "all": rates(counts),
        "by_task": {k: rates(v) for k, v in by_task.items()},
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / f"{args.dataset}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== Learned coordinate grounding — {args.dataset} ===")
    print(f"train={len(train)} test={counts['n']} weights={out['weights']}")
    print(f"heuristic={out['all']['heuristic']:.4f} coord={out['all']['coord']:.4f} "
          f"learned={out['all']['learned']:.4f} either={out['all']['either']:.4f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
