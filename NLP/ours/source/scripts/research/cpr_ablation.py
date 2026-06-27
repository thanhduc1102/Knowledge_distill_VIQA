#!/usr/bin/env python3
"""CPR component ablation: which of Concept / Period / Role drives the reliability gain?

Reuses cached generation predictions; for each query rebuilds the ledger once and scores the
RAW answer with several CPR configurations, reporting AUROC of confidence vs correctness:

  value_only          (legacy baseline)
  C / P / R           (single component only)
  CP / CR / PR        (pairs)
  CPR                 (full)

This isolates the contribution of each criterion (answers: which is thin, which carries the gain).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import verify, extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.research.cpr_verifier import verify_cpr


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
    sp = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sp - pos * (pos + 1) / 2.0) / (pos * neg)


def _union(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    m = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


CONFIGS = {
    "value_only": None,
    "C": frozenset({"concept"}),
    "P": frozenset({"period"}),
    "R": frozenset({"role"}),
    "CP": frozenset({"concept", "period"}),
    "CR": frozenset({"concept", "role"}),
    "PR": frozenset({"period", "role"}),
    "CPR": frozenset({"concept", "period", "role"}),
}


def run(ds, pred_path, retr_path, limit):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}
    if limit:
        preds = preds[:limit]

    correct, conf = [], {k: [] for k in CONFIGS}
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
        correct.append(bool(p.get("raw_correct") if "raw_correct" in p else number_match(pv, gold)))
        for name, comps in CONFIGS.items():
            if comps is None:
                vr = verify(pred_text, ledger, query, gold=gold)
                conf[name].append(1.0 if vr.grounded else (0.85 if vr.derivable else 0.1))
            else:
                cpr = verify_cpr(pred_text, ledger, query, gold=gold,
                                 selected_facts=pack.selected_facts, components=comps)
                conf[name].append(cpr.confidence)

    return {
        "dataset": ds, "n": len(correct),
        "raw_accuracy": round(sum(correct) / max(len(correct), 1), 4),
        "auroc": {k: round(_auroc(conf[k], correct), 4) for k in CONFIGS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/generation_system_q35_s400/{ds}_predictions.jsonl")
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/cpr_ablation")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    order = ["value_only", "C", "P", "R", "CP", "CR", "PR", "CPR"]
    print(f"{'dataset':10s} " + " ".join(f"{k:>10s}" for k in order))
    for ds in args.datasets:
        r = run(ds, args.pred.format(ds=ds), args.retr.format(ds=ds), args.limit)
        allres[ds] = r
        (out / f"{ds}.json").write_text(json.dumps(r, indent=2))
        print(f"{ds:10s} " + " ".join(f"{r['auroc'][k]:>10.4f}" for k in order))
    (out / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
