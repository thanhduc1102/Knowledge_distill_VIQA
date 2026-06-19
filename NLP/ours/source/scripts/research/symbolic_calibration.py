#!/usr/bin/env python3
"""Deterministic AGREEMENT calibration for KG symbolic answers (breakthrough probe).

Blueprint weakness #1: the symbolic answer is gated by a HAND-TUNED confidence (>=0.8) yet
``symbolic_nm_when_available`` is only 0.15-0.38 — the confidence is poorly calibrated.

Hypothesis: a *deterministic multi-signal agreement* is a far better-calibrated trust signal
than a tuned scalar. For each symbolic answer we count how many INDEPENDENT grounding signals
corroborate the chosen operands:

  a_concept  — operand concept (canonical or tokens) matches the question's target concept
  a_period   — operand period matches a question year (or the question names no year)
  a_winner   — operands come from the KG arbitration-winner document
  a_noconf   — no value conflict for that (concept, period) across the top-k
  a_tier     — document tier is TRUST_TOP1 (high doc-selection margin)
  a_idok     — the source doc's accounting identities are not violated

agreement = sum(signals) ∈ [0, 6].  We then report symbolic NM and coverage when gating on
``agreement >= k``, and compare to the current ``confidence >= 0.8`` policy.  If NM rises
sharply and monotonically with agreement, agreement-gating gives a calibrated abstention
mechanism — the finance-critical "knows when it is sure" property — without any training.

Usage:  PYTHONPATH=src python scripts/research/symbolic_calibration.py --dataset finqa
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.ledger.select import _tokens
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.ledger.numeric import extract_years

DEFAULTS = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}


def _agreement(pack, question) -> int:
    calc = pack.calculation or {}
    ops = calc.get("operands") or []
    if not ops:
        return 0
    q_concepts = concepts_in_text(question)
    q_years = set(extract_years(question))
    q_tokens = _tokens(question)
    winner = pack.ranked[0].doc_id if pack.ranked else ""
    conflict_keys = {(str(c.get("concept") or ""), str(c.get("period") or "_")) for c in pack.conflicts}

    table_ops = [o for o in ops if o.get("provenance") != "question"]
    if not table_ops:
        return 1   # purely question-literal operands → weak corroboration

    a_concept = a_period = a_winner = a_noconf = 0
    for o in table_ops:
        cc = o.get("concept_canonical")
        ctoks = _tokens(str(o.get("concept") or ""))
        if (cc and cc in q_concepts) or (ctoks & q_tokens):
            a_concept = 1
        per = str(o.get("period") or "")
        if not q_years or (per.isdigit() and int(per) in q_years):
            a_period = 1
        prov = str(o.get("provenance") or "")
        if winner and winner in prov:
            a_winner = 1
        key = (str(cc or str(o.get("concept") or "").lower()), str(o.get("period") or "_"))
        if key not in conflict_keys:
            a_noconf = 1
    a_tier = 1 if pack.tier == "TRUST_TOP1" else 0
    v = pack.verification or {}
    a_idok = 1 if v.get("n_violated", 0) == 0 else 0
    return a_concept + a_period + a_winner + a_noconf + a_tier + a_idok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/symbolic_calibration")
    args = ap.parse_args()

    path = args.input or DEFAULTS[args.dataset]
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if args.limit:
        recs = recs[: args.limit]
    n = len(recs)

    rows = []           # (agreement, nm, conf, has_answer)
    conf_has = conf_nm = 0
    for rec in recs:
        pack = build_evidence_pack(rec["query"], (rec.get("retrieved") or [])[:3])
        calc = pack.calculation or {}
        ans = calc.get("answer")
        if ans is None:
            continue
        nm = int(number_match(ans, rec.get("gold")))
        conf = float(calc.get("confidence", 0.0) or 0.0)
        agree = _agreement(pack, rec["query"])
        rows.append((agree, nm, conf))
        if conf >= 0.8:
            conf_has += 1; conf_nm += nm

    rows = np.array(rows, dtype=float) if rows else np.zeros((0, 3))
    out = {"dataset": args.dataset, "n": n, "raw_symbolic_coverage": round(len(rows) / max(n, 1), 4)}

    print(f"\n=== Symbolic-answer calibration — {args.dataset} (N={n}, raw symbolic={len(rows)}) ===")
    print("current policy (confidence>=0.8):")
    print(f"   coverage={conf_has/max(n,1):.3f}  symbolic_NM_when_available={conf_nm/max(conf_has,1):.3f}")
    out["confidence>=0.8"] = {"coverage": round(conf_has/max(n,1), 4),
                              "nm_when_available": round(conf_nm/max(conf_has, 1), 4)}

    print("\nagreement-gating (deterministic multi-signal):")
    print(f"   {'agree>=k':<10}{'coverage':>10}{'NM|avail':>10}{'n':>7}")
    gate = {}
    for k in range(0, 7):
        sel = rows[rows[:, 0] >= k] if len(rows) else rows
        cov = len(sel) / max(n, 1)
        nm = sel[:, 1].mean() if len(sel) else 0.0
        print(f"   {k:<10}{cov:>10.3f}{nm:>10.3f}{len(sel):>7}")
        gate[f">={k}"] = {"coverage": round(cov, 4), "nm_when_available": round(float(nm), 4), "n": int(len(sel))}
    out["agreement_gating"] = gate

    # NM by exact agreement bucket (calibration curve)
    print("\n   per-bucket calibration (exact agreement):")
    buck = {}
    for k in range(0, 7):
        sel = rows[rows[:, 0] == k] if len(rows) else rows
        nm = sel[:, 1].mean() if len(sel) else 0.0
        buck[str(k)] = {"n": int(len(sel)), "nm": round(float(nm), 4)}
        if len(sel):
            print(f"      agree={k}: n={len(sel):4}  NM={nm:.3f}")
    out["per_bucket"] = buck

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset+'.json')}")


if __name__ == "__main__":
    main()
