#!/usr/bin/env python3
"""Provenance precision audit (semi-automatic, inspectable).

For every answer the CPR verifier calls *grounded*, we record WHICH ledger cell it was grounded
to, and whether that cell is concept- and period-consistent with the query. This yields:
  * provenance precision = fraction of grounded answers whose cited cell is concept+period
    consistent (split by whether the answer is actually correct), and
  * a dump of N samples (query, answer, cited cell + provenance string) for human inspection.

This is the reviewer-required check that the *cited evidence* is the right cell, not just that
the number appears somewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, extract_years
from gsr_cacl.ledger.select import _tokens, _target_year_pair
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.research.cpr_verifier import _concept_consistency, _period_consistency, _value_match_facts


def _union(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    m = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def run(ds, pred_path, retr_path, n_dump, limit):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}
    if limit:
        preds = preds[:limit]

    grounded = {"n": 0, "concept_ok": 0, "period_ok": 0, "both_ok": 0,
                "correct": 0, "both_ok_and_correct": 0, "both_ok_and_wrong": 0}
    samples = []
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
        if ledger is None:
            continue
        matches = _value_match_facts(pv, ledger, 1e-2)
        if not matches:
            continue  # not grounded
        q_tokens = _tokens(query); q_concepts = concepts_in_text(query)
        pr = _target_year_pair(query)
        q_years = list(pr) if pr else sorted(set(extract_years(query)))
        # choose the best-typed matching cell as the cited provenance
        best = max(matches, key=lambda f: _concept_consistency(f, q_concepts, q_tokens)
                   * _period_consistency(f, q_years, 0.5))
        cc = _concept_consistency(best, q_concepts, q_tokens)
        pc = _period_consistency(best, q_years, 0.5)
        correct = bool(p.get("raw_correct") if "raw_correct" in p else number_match(pv, gold))
        c_ok, p_ok = cc >= 0.5, pc >= 0.5
        grounded["n"] += 1
        grounded["concept_ok"] += int(c_ok)
        grounded["period_ok"] += int(p_ok)
        grounded["both_ok"] += int(c_ok and p_ok)
        grounded["correct"] += int(correct)
        grounded["both_ok_and_correct"] += int(c_ok and p_ok and correct)
        grounded["both_ok_and_wrong"] += int(c_ok and p_ok and not correct)
        if len(samples) < n_dump:
            samples.append({
                "query": query[:140], "answer": pv, "gold": gold, "correct": correct,
                "cited_cell": f"{best.concept} [{best.period}] = {best.raw_text or best.value}",
                "provenance": best.provenance, "concept_consistent": round(cc, 2),
                "period_consistent": round(pc, 2),
            })

    g = grounded; n = max(g["n"], 1)
    summary = {
        "dataset": ds, "grounded_n": g["n"],
        "provenance_precision_concept": round(g["concept_ok"] / n, 4),
        "provenance_precision_period": round(g["period_ok"] / n, 4),
        "provenance_precision_both": round(g["both_ok"] / n, 4),
        "of_grounded_correct": round(g["correct"] / n, 4),
        # key audit numbers: among answers whose cited cell is concept+period consistent,
        # how many are actually correct (precision of a *well-cited* grounding)?
        "precision_when_well_cited": round(g["both_ok_and_correct"] / max(g["both_ok"], 1), 4),
    }
    return summary, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/generation_system_q35_s400/{ds}_predictions.jsonl")
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--n-dump", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/provenance_audit")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for ds in args.datasets:
        summary, samples = run(ds, args.pred.format(ds=ds), args.retr.format(ds=ds), args.n_dump, args.limit)
        allres[ds] = summary
        (out / f"{ds}_samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
        s = summary
        print(f"\n=== {ds} (grounded n={s['grounded_n']}) ===")
        print(f"  provenance precision: concept={s['provenance_precision_concept']} "
              f"period={s['provenance_precision_period']} both={s['provenance_precision_both']}")
        print(f"  precision_when_well_cited (cited cell concept+period ok -> correct) = "
              f"{s['precision_when_well_cited']}  vs of_grounded_correct={s['of_grounded_correct']}")
    (out / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out}/ ({args.n_dump} samples/dataset for manual review)")


if __name__ == "__main__":
    main()
