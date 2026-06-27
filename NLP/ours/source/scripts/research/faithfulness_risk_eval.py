#!/usr/bin/env python3
"""Paper-facing faithfulness diagnostics from generation predictions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.research.faithfulness import (
    FaithfulnessRecord,
    auc_risk_coverage,
    binary_auc,
    bootstrap_group_gap,
    grouped_correctness,
    hallucination_proxy,
    provenance_summary,
    risk_coverage_curve,
)

_CELL = re.compile(r"grounded@([^;]+)")


def _record(row: dict) -> FaithfulnessRecord:
    gold = row.get("gold")
    pred = row.get("pred_value")
    correct = bool(row.get("answer_match", False))
    if not correct and gold is not None and pred is not None:
        correct = bool(number_match(pred, gold))
    explanation = str(row.get("explanation") or "")
    m = _CELL.search(explanation)
    has_prov = bool(m)
    # Proxy: if the answer is correct and grounded, the attached provenance is useful.
    # A stricter cell-level human audit can be plugged in by adding provenance_correct
    # to the prediction rows.
    prov_correct = row.get("provenance_correct")
    if prov_correct is None and has_prov:
        prov_correct = bool(correct)
    return FaithfulnessRecord(
        correct=correct,
        grounded=bool(row.get("grounded", False)),
        reward=float(row.get("reward", 0.0) or 0.0),
        grounding_fraction=float(row.get("grounding_fraction", 0.0) or 0.0),
        arithmetic_fraction=float(row.get("arithmetic_fraction", 0.0) or 0.0),
        has_provenance=has_prov,
        provenance_correct=prov_correct,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="predictions.jsonl")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", default="outputs/research/faithfulness")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    recs = [_record(r) for r in rows]
    out = {
        "dataset": args.dataset or Path(args.input).parent.name,
        "input": args.input,
        "n": len(recs),
        "grouped_correctness": grouped_correctness(recs),
        "grounded_gap_bootstrap": bootstrap_group_gap(recs),
        "selective_risk": auc_risk_coverage(recs),
        "confidence_auroc": binary_auc(recs),
        "hallucination_proxy": hallucination_proxy(recs),
        "provenance": provenance_summary(recs),
        "risk_coverage": risk_coverage_curve(recs, steps=20),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / f"{out['dataset']}.json"
    out_path.write_text(json.dumps(out, indent=2))

    g = out["grouped_correctness"]
    h = out["hallucination_proxy"]
    print(f"\n=== Faithfulness risk eval — {out['dataset']} (n={len(recs)}) ===")
    print(f"grounded: n={g['grounded_n']} acc={g['grounded_accuracy']:.4f} | "
          f"ungrounded: n={g['ungrounded_n']} acc={g['ungrounded_accuracy']:.4f} | "
          f"separation={g['separation']:.4f}")
    print(f"gap CI={out['grounded_gap_bootstrap']['ci95']} "
          f"p<=0={out['grounded_gap_bootstrap']['p_gap_le_0']:.4f} | "
          f"AUROC={out['confidence_auroc']:.4f} AURC={out['selective_risk']['aurc']:.4f}")
    print(f"hallucination catch proxy={h['hallucination_catch_rate']:.4f} "
          f"({h['ungrounded_wrong_n']}/{h['wrong_n']} wrong answers)")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
