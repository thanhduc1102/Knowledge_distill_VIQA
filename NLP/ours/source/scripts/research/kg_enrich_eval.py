#!/usr/bin/env python3
"""Measure the KG-construction enrichment: semantic canonicalisation → coverage + identities.

Compares the typed fact graph BEFORE vs AFTER semantic concept canonicalisation, on a corpus
sample. Reports canonical coverage, identity-edge firing, and — the key safety check — the
satisfied fraction of fired identities (a drop would mean the enrichment invents false concepts).

Usage:  PYTHONPATH=src python scripts/research/kg_enrich_eval.py --dataset FinQA --sample 250 --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.kg.fact_graph import FinancialFactGraph
from gsr_cacl.ledger.semantic_concepts import SemanticCanonicalizer

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}


def stats(ledgers):
    cov, id_docs, id_edges, id_sat, n = [], 0, [], [], 0
    for led in ledgers:
        nf = led.numeric_facts()
        if not nf:
            continue
        n += 1
        cov.append(sum(1 for f in nf if f.concept_canonical) / len(nf))
        g = FinancialFactGraph(led)
        if g.identity_edges:
            id_docs += 1
            id_edges.append(len(g.identity_edges))
            id_sat.append(sum(1 for e in g.identity_edges if e.satisfied) / len(g.identity_edges))
    return {
        "canonical_coverage": round(float(np.mean(cov)), 3) if cov else 0.0,
        "identity_firing": round(id_docs / max(n, 1), 3),
        "avg_identity_edges": round(float(np.mean(id_edges)), 2) if id_edges else 0.0,
        "identity_satisfied_frac": round(float(np.mean(id_sat)), 3) if id_sat else 0.0,
        "n_docs": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=list(SPLITS))
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--out", default="outputs/research/kg_enrich")
    args = ap.parse_args()

    data = load_t2ragbench_split(args.dataset, split=SPLITS[args.dataset], sample_size=None)
    base_ledgers = []
    for d in data.corpus[: args.sample]:
        tmd = extract_table(d.page_content)
        if not tmd:
            continue
        try:
            base_ledgers.append(extract_ledger_from_table(
                tmd, doc_id=str(d.id),
                meta=dict(d.meta_data) if isinstance(d.meta_data, dict) else {}))
        except Exception:
            pass

    before = stats(base_ledgers)
    canon = SemanticCanonicalizer(device=args.device, threshold=args.threshold)
    for led in base_ledgers:
        canon.enrich(led)
    after = stats(base_ledgers)

    print(f"\n=== KG enrichment (semantic canonicalisation, thr={args.threshold}) — {args.dataset} ===")
    print(f"{'metric':<28}{'before':>10}{'after':>10}")
    for k in ("canonical_coverage", "identity_firing", "avg_identity_edges", "identity_satisfied_frac"):
        print(f"{k:<28}{before[k]:>10}{after[k]:>10}")
    out = {"dataset": args.dataset, "before": before, "after": after, "threshold": args.threshold}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset.lower()}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset.lower()+'.json')}")


if __name__ == "__main__":
    main()
