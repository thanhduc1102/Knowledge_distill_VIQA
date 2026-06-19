#!/usr/bin/env python3
"""Diagnose the typed Financial Fact Graph CONSTRUCTION quality (where the KG is thin).

Reports, over a corpus sample, the coverage of every KG ingredient so we know exactly what to
enrich: canonical-concept coverage (drives identities + ontology answering), identity-edge
firing rate (drives verification), temporal-edge density (drives Δ/%-change), and 2-D grid
recoverability (drives coordinate grounding).

Usage:  PYTHONPATH=src python scripts/research/kg_construction_diag.py --dataset FinQA --sample 400
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

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=list(SPLITS))
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--out", default="outputs/research/kg_diag")
    args = ap.parse_args()

    data = load_t2ragbench_split(args.dataset, split=SPLITS[args.dataset], sample_size=None)
    corpus = data.corpus[: args.sample]

    n_doc = 0
    facts_per = []; canon_frac = []; grid_frac = []
    id_docs = 0; id_edges = []; id_sat = []
    temp_docs = 0; temp_edges = []
    distinct_concepts = set()
    for d in corpus:
        tmd = extract_table(d.page_content)
        if not tmd:
            continue
        try:
            led = extract_ledger_from_table(tmd, doc_id=str(d.id),
                                            meta=dict(d.meta_data) if isinstance(d.meta_data, dict) else {})
        except Exception:
            continue
        nf = led.numeric_facts()
        if not nf:
            continue
        n_doc += 1
        facts_per.append(len(nf))
        canon_frac.append(sum(1 for f in nf if f.concept_canonical) / len(nf))
        grid_frac.append(sum(1 for f in nf if f.row_idx is not None and f.row_idx >= 0
                             and f.col_idx is not None and f.col_idx >= 0) / len(nf))
        for f in nf:
            if f.concept_canonical:
                distinct_concepts.add(f.concept_canonical)
        g = FinancialFactGraph(led)
        if g.identity_edges:
            id_docs += 1
            id_edges.append(len(g.identity_edges))
            id_sat.append(sum(1 for e in g.identity_edges if e.satisfied) / len(g.identity_edges))
        if g.temporal_edges:
            temp_docs += 1
            temp_edges.append(len(g.temporal_edges))

    def m(x): return round(float(np.mean(x)), 3) if x else 0.0
    out = {
        "dataset": args.dataset, "n_docs": n_doc,
        "avg_facts_per_doc": m(facts_per),
        "canonical_concept_coverage(frac of facts)": m(canon_frac),
        "grid_recoverable(frac of facts w/ row,col)": m(grid_frac),
        "distinct_canonical_concepts_seen": len(distinct_concepts),
        "identity_edge_firing(frac of docs)": round(id_docs / max(n_doc, 1), 3),
        "avg_identity_edges_per_firing_doc": m(id_edges),
        "avg_identity_satisfied_frac": m(id_sat),
        "temporal_edge_firing(frac of docs)": round(temp_docs / max(n_doc, 1), 3),
        "avg_temporal_edges_per_firing_doc": m(temp_edges),
    }
    print(f"\n=== KG construction diagnosis — {args.dataset} (docs={n_doc}) ===")
    for k, v in out.items():
        if k not in ("dataset", "n_docs"):
            print(f"   {k:<48} {v}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset.lower()}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset.lower()+'.json')}")


if __name__ == "__main__":
    main()
