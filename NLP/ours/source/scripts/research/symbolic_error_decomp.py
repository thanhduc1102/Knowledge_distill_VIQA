#!/usr/bin/env python3
"""Decompose KG symbolic-answer errors: format/scale vs wrong-operand vs genuinely-hard.

The calibration probe showed symbolic NM stays low (0.2-0.38) even at maximum agreement, so
the bottleneck is the answer itself, not the gating. This script asks WHY a symbolic answer
fails Number-Match, by trying value-preserving transforms before declaring it "wrong":

  exact            number_match(pred, gold) already true
  percent_format   pred*100 or pred/100 matches gold   (decimal-vs-percent mismatch)
  scale_format     pred*10^k matches gold for k in -6..6 (thousands/millions/unit mismatch)
  sign_format      -pred matches gold                    (accounting-sign convention)
  wrong_value_present  none of the above, BUT the gold value IS among the selected facts
                       (operand was available → cell-selection / arithmetic error)
  wrong_value_absent   gold value not in selected facts  (extraction/retrieval miss)

If format classes dominate, the KG's *reasoning* is right and the fix is answer
representation (cheap, large NM gain) — a very different conclusion than "KG can't reason".
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.ledger.numeric import number_match, parse_financial_number

DEFAULTS = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}


def _gold_float(g):
    if isinstance(g, (list, tuple)) and g:
        g = g[0]
    return parse_financial_number(str(g)) if not isinstance(g, (int, float)) else float(g)


def classify(pred, gold_val, gold_raw, facts):
    if number_match(pred, gold_raw):
        return "exact"
    if gold_val is None:
        return "wrong_value_absent"
    for t in (pred * 100, pred / 100):
        if number_match(t, gold_raw):
            return "percent_format"
    for k in range(-6, 7):
        if k != 0 and number_match(pred * (10.0 ** k), gold_raw):
            return "scale_format"
    if number_match(-pred, gold_raw):
        return "sign_format"
    # is the gold value present among selected facts (operand available)?
    for f in facts:
        if f.value is not None and abs(f.value - gold_val) <= 1e-2 * max(abs(gold_val), 1.0):
            return "wrong_value_present"
        if f.value not in (None, 0) and abs(abs(f.value) - abs(gold_val)) <= 1e-2 * max(abs(gold_val), 1.0):
            return "wrong_value_present"
    return "wrong_value_absent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/symbolic_errdecomp")
    args = ap.parse_args()

    path = args.input or DEFAULTS[args.dataset]
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    cats: dict[str, int] = {}
    by_task: dict[str, dict[str, int]] = {}
    total = 0
    for rec in recs:
        pack = build_evidence_pack(rec["query"], (rec.get("retrieved") or [])[:3])
        calc = pack.calculation or {}
        ans = calc.get("answer")
        if ans is None:
            continue
        total += 1
        gv = _gold_float(rec.get("gold"))
        c = classify(float(ans), gv, rec.get("gold"), pack.selected_facts)
        cats[c] = cats.get(c, 0) + 1
        t = pack.task
        by_task.setdefault(t, {}).setdefault(c, 0)
        by_task[t][c] += 1

    order = ["exact", "percent_format", "scale_format", "sign_format",
             "wrong_value_present", "wrong_value_absent"]
    print(f"\n=== Symbolic error decomposition — {args.dataset} (symbolic answers={total}) ===")
    for c in order:
        v = cats.get(c, 0)
        print(f"   {c:<22}{v:>6}  {v/max(total,1):>7.1%}")
    recoverable = sum(cats.get(c, 0) for c in ("exact", "percent_format", "scale_format", "sign_format"))
    print(f"   {'-'*36}")
    print(f"   exact + format-recoverable : {recoverable/max(total,1):.1%}  "
          f"(ceiling if answer representation were normalised)")
    print(f"   wrong-operand (value present): {cats.get('wrong_value_present',0)/max(total,1):.1%}  "
          f"→ cell-addressing target")
    print(f"   value absent (extract/retr.) : {cats.get('wrong_value_absent',0)/max(total,1):.1%}")

    out = {"dataset": args.dataset, "n_symbolic": total, "categories": cats,
           "recoverable_ceiling": round(recoverable / max(total, 1), 4), "by_task": by_task}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset+'.json')}")


if __name__ == "__main__":
    main()
