#!/usr/bin/env python3
"""Derivation-path parsimony: is answer reliability governed by the *minimal typed derivation
path* from query intent to answer through the structure graph?

Unifying hypothesis (the framework's core): grounding, CPR typing, and multi-operand derivation
are all instances of one principle — an answer is reliable to the degree that it admits a SHORT,
TYPE-CONSISTENT derivation path over the ledger. We test it by binning each generated answer by:
  * depth      : 0 grounded (answer is a cell), 1 two-operand, 2 three-operand, 3 none,
  * typedness  : whether the supporting fact(s) are concept+period consistent with the query,
and measuring accuracy per bin. A monotone accuracy↓ with depth, and typed>untyped at equal
depth, would establish "typed derivation parsimony" as a calibrated, general reliability law.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number, is_grounded
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, extract_years
from gsr_cacl.ledger.select import _tokens, _target_year_pair
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.research.cpr_verifier import _concept_consistency, _period_consistency, _value_match_facts
from gsr_cacl.research.derivation import derivation_depth

DEPTH_NAME = {0: "0_grounded", 1: "1_two_op", 2: "2_three_op", 3: "3_none"}


def _union(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    m = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _typed_at_grounding(value, ledger, query) -> bool:
    """For a grounded answer, is the matched cell concept+period consistent with the query?"""
    q_tokens = _tokens(query)
    q_concepts = concepts_in_text(query)
    pair = _target_year_pair(query)
    q_years = list(pair) if pair else sorted(set(extract_years(query)))
    for f in _value_match_facts(value, ledger, 1e-2):
        cc = _concept_consistency(f, q_concepts, q_tokens)
        pc = _period_consistency(f, q_years, 0.5)
        if cc >= 0.5 and pc >= 0.5:
            return True
    return False


def run(ds, pred_path, retr_path, limit):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}
    if limit:
        preds = preds[:limit]

    depth_acc = defaultdict(lambda: {"n": 0, "correct": 0})
    typed = {"typed": {"n": 0, "correct": 0}, "untyped": {"n": 0, "correct": 0}}
    for p in preds:
        rec = by_id.get(str(p.get("query_id"))) or by_q.get(p.get("query"))
        if not rec:
            continue
        retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
        if not retrieved:
            continue
        query, gold = p["query"], p.get("gold")
        pred_text = p.get("raw_prediction") or p.get("prediction") or ""
        pv = extract_final_number(pred_text)
        if pv is None:
            continue
        pack = build_evidence_pack(query, retrieved, top_n_facts=8)
        ledger = _union(pack)
        if ledger is None or not ledger.numeric_facts():
            continue
        correct = bool(p.get("raw_correct") if "raw_correct" in p else number_match(pv, gold))
        vals = [f.value for f in ledger.numeric_facts() if f.value is not None]
        dd = derivation_depth(pv, vals, max_ops=3)
        depth = {"grounded": 0, "2op": 1, "3op": 2, None: 3}[dd]
        b = depth_acc[depth]; b["n"] += 1; b["correct"] += int(correct)
        if depth == 0:
            key = "typed" if _typed_at_grounding(pv, ledger, query) else "untyped"
            typed[key]["n"] += 1; typed[key]["correct"] += int(correct)

    def acc(b):
        return {"n": b["n"], "accuracy": round(b["correct"] / max(b["n"], 1), 4)}

    curve = {DEPTH_NAME[d]: acc(depth_acc[d]) for d in sorted(depth_acc)}
    # monotonicity: fraction of adjacent depth pairs where accuracy decreases
    accs = [depth_acc[d]["correct"] / max(depth_acc[d]["n"], 1) for d in sorted(depth_acc) if depth_acc[d]["n"] >= 5]
    mono = all(accs[i] >= accs[i + 1] - 1e-9 for i in range(len(accs) - 1)) if len(accs) > 1 else None
    return {
        "dataset": ds, "n": sum(b["n"] for b in depth_acc.values()),
        "accuracy_by_depth": curve,
        "monotone_decreasing": mono,
        "grounded_typed_vs_untyped": {"typed": acc(typed["typed"]), "untyped": acc(typed["untyped"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/generation_system_q35_s400/{ds}_predictions.jsonl")
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/derivation_reliability")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for ds in args.datasets:
        r = run(ds, args.pred.format(ds=ds), args.retr.format(ds=ds), args.limit)
        allres[ds] = r
        print(f"\n=== {ds} (n={r['n']}) — accuracy by minimal derivation depth ===")
        for k, v in r["accuracy_by_depth"].items():
            print(f"    {k:12s} n={v['n']:4d}  acc={v['accuracy']}")
        print(f"    monotone_decreasing={r['monotone_decreasing']}")
        t = r["grounded_typed_vs_untyped"]
        print(f"    grounded TYPED acc={t['typed']['accuracy']} (n={t['typed']['n']}) "
              f"vs UNTYPED acc={t['untyped']['accuracy']} (n={t['untyped']['n']})")
    (out / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
