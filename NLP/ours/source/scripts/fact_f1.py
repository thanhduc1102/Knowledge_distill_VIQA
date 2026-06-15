#!/usr/bin/env python3
"""Fact-F1: measure Fact-Ledger extraction quality on the GOLD document.

This diagnoses *where* the generator bottleneck is (extraction vs. selection/reasoning)
by measuring, per query, on the gold table only:

  * cell_recall   — fraction of the table's numeric cells captured as ledger facts
                    (ledger fidelity: are we losing cells?)
  * answer_is_lookup — gold answer equals some raw table cell (no arithmetic needed)
  * answer_in_ledger — gold answer value appears among extracted fact values
  * answer_derivable — gold answer is reachable from ledger facts via one op
                    (a, a-b, b-a, a+b, a/b, pct-change) over fact-value pairs
                    → "operand coverage": can the answer be COMPUTED from the ledger?
  * canonical_rate  — fraction of facts mapped to a canonical IFRS/GAAP concept
  * period_rate     — fraction of facts with an attached period

Reading the result:
  high answer_derivable + low generation NM  -> bottleneck is selection/reasoning
  low  answer_derivable                       -> bottleneck is extraction (fix the ledger)

Usage:
  python scripts/fact_f1.py --dataset FinQA --split test --sample 0   # 0 = all
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import permutations
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
sys.path.insert(0, "src")

from datasets import load_dataset

from gsr_cacl.kg.parser import parse_markdown_rows
from gsr_cacl.ledger.numeric import parse_financial_number, numbers_close
from gsr_cacl.ledger.extract import extract_ledger_from_table, _is_index_column


def table_numeric_cells(table_md: str) -> list[float]:
    """All numeric cell values in the table, skipping the pandas index column."""
    rows = parse_markdown_rows(table_md)
    if len(rows) < 2:
        return []
    data = rows[1:]
    ncol = max((len(r) for r in data), default=0)
    index_cols = {c for c in range(ncol)
                  if _is_index_column([r[c] if c < len(r) else "" for r in data])}
    vals: list[float] = []
    for r in data:
        for c, cell in enumerate(r):
            if c in index_cols:
                continue
            v = parse_financial_number(cell)
            if v is not None:
                vals.append(v)
    return vals


def _value_multiset_recall(extracted: list[float], reference: list[float]) -> float:
    """Fraction of reference values matched by some extracted value (greedy, tol-aware)."""
    if not reference:
        return 1.0
    remaining = list(extracted)
    hit = 0
    for r in reference:
        for i, e in enumerate(remaining):
            if numbers_close(e, r, rel_tol=1e-4) or e == r:
                hit += 1
                remaining.pop(i)
                break
    return hit / len(reference)


def answer_derivable(answer: float, fact_vals: list[float], max_facts: int = 40) -> bool:
    """Is `answer` reachable from the ledger via a single common financial op?"""
    vals = fact_vals[:max_facts]
    # direct lookup
    for v in vals:
        if numbers_close(v, answer, rel_tol=1e-2):
            return True
        for f in (100.0, 1000.0, 1e6):  # scale-annotation drift
            if numbers_close(v * f, answer, rel_tol=1e-2) or numbers_close(v, answer * f, rel_tol=1e-2):
                return True
    # one-op over ordered pairs
    for a, b in permutations(vals, 2):
        cands = [a - b, a + b]
        if abs(b) > 1e-9:
            cands.append(a / b)             # ratio
            cands.append((a - b) / abs(b))  # pct change
        for c in cands:
            if numbers_close(c, answer, rel_tol=1e-2):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=["FinQA", "ConvFinQA", "TAT-DQA"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=0, help="0 = all queries")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = load_dataset("G4KMU/t2-ragbench", args.dataset, split=args.split)
    n = len(ds) if args.sample <= 0 else min(args.sample, len(ds))

    agg = {
        "cell_recall": 0.0, "canonical_rate": 0.0, "period_rate": 0.0,
        "answer_is_lookup": 0, "answer_in_ledger": 0, "answer_derivable": 0,
        "answer_parsed": 0, "n_facts": 0, "empty_ledger": 0,
    }
    used = 0
    for i in range(n):
        row = ds[i]
        table_md = str(row.get("table", "") or "")
        if not table_md.strip():
            continue
        used += 1

        ledger = extract_ledger_from_table(table_md, meta={
            "company_name": str(row.get("company_name", "")),
        })
        facts = ledger.numeric_facts()
        fact_vals = [f.value for f in facts if f.value is not None]
        if not facts:
            agg["empty_ledger"] += 1

        cells = table_numeric_cells(table_md)
        agg["cell_recall"] += _value_multiset_recall(fact_vals, cells)
        agg["canonical_rate"] += (sum(1 for f in facts if f.concept_canonical) / len(facts)) if facts else 0.0
        agg["period_rate"] += (sum(1 for f in facts if f.period) / len(facts)) if facts else 0.0
        agg["n_facts"] += len(facts)

        golds = [row.get("program_answer"), row.get("original_answer")]
        gv = next((parse_financial_number(g) for g in golds if parse_financial_number(g) is not None), None)
        if gv is not None:
            agg["answer_parsed"] += 1
            if any(numbers_close(c, gv, rel_tol=1e-2) for c in cells):
                agg["answer_is_lookup"] += 1
            if any(numbers_close(v, gv, rel_tol=1e-2) for v in fact_vals):
                agg["answer_in_ledger"] += 1
            if answer_derivable(gv, fact_vals):
                agg["answer_derivable"] += 1

    p = agg["answer_parsed"] or 1
    report = {
        "dataset": args.dataset, "split": args.split,
        "n_queries_used": used, "answer_parsed": agg["answer_parsed"],
        "ledger": {
            "mean_facts_per_doc": round(agg["n_facts"] / used, 2),
            "empty_ledger_rate": round(agg["empty_ledger"] / used, 4),
            "cell_recall": round(agg["cell_recall"] / used, 4),
            "canonical_rate": round(agg["canonical_rate"] / used, 4),
            "period_rate": round(agg["period_rate"] / used, 4),
        },
        "answer_reachability": {
            "answer_is_lookup": round(agg["answer_is_lookup"] / p, 4),
            "answer_in_ledger": round(agg["answer_in_ledger"] / p, 4),
            "answer_derivable_one_op": round(agg["answer_derivable"] / p, 4),
        },
    }
    out_dir = Path(args.out or f"outputs/fact_f1/{args.dataset.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fact_f1.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out_dir}/fact_f1.json")


if __name__ == "__main__":
    main()
