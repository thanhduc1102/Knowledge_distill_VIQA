#!/usr/bin/env python3
"""Collect the final paper-facing ablation table from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _read(path: str):
    return json.loads(Path(path).read_text())


def main():
    rows = {}
    for ds, pretty in (("finqa", "FinQA"), ("convfinqa", "ConvFinQA"), ("tat-dqa", "TAT-DQA")):
        diff = _read(f"outputs/research/difficulty/{ds}.json")
        key = "tatqa" if ds == "tat-dqa" else ds
        final = _read(f"outputs/final_retrieval/{key}/ablation.json")
        cp = _read(f"outputs/cp_fusion/{ds}/modular.json")
        rows[pretty] = {
            "no_metadata_corpus_bm25": diff["regimes"]["A_corpus_bm25"]["MRR@3"],
            "company_pool_only_meta_rank": cp["experts"]["meta"]["MRR@3"],
            "company_pool_global_bm25": diff["regimes"]["B_companypool_bm25"]["MRR@3"],
            "company_pool_loclex": diff["regimes"]["C_companypool_loclex"]["MRR@3"],
            "concept": cp["experts"]["concept"]["MRR@3"],
            "cell": cp["experts"]["cell"]["MRR@3"],
            "fusion_meta_loclex_concept_cell": max(v["MRR@3"] for v in cp["fusion"].values()),
            "dense": final["arms"]["dense"]["MRR@3"],
            "full_entity_meta": final["arms"]["FULL (entity+meta, β=0.6)"]["MRR@3"],
            "full_plus_concept_coverage": final["arms"]["FULL + C3 δ=0.1 (fixed w)"]["MRR@3"],
        }

    faith = {}
    for ds in ("finqa", "convfinqa", "tatqa"):
        faith[ds] = _read(f"outputs/research/faithfulness/{ds}.json")
    reask = {}
    for ds in ("finqa", "convfinqa", "tatqa"):
        reask[ds] = _read(f"outputs/research/verify_reask/{ds}.json")
    reask_policy = {}
    for ds in ("finqa", "convfinqa", "tatqa"):
        reask_policy[ds] = _read(f"outputs/research/reask_policy/{ds}.json")
    learned = {}
    for ds in ("finqa", "convfinqa", "tatqa"):
        learned[ds] = _read(f"outputs/research/learned_coordinate/{ds}.json")
    financebench = _read("outputs/research/external_financebench/financebench_retrieval.json")
    failure = _read("outputs/research/global_failure_audit.json")

    out = {
        "retrieval_ablation_mrr3": rows,
        "external_financebench": financebench["metrics"],
        "faithfulness": {
            ds: {
                "grounded_accuracy": v["grouped_correctness"]["grounded_accuracy"],
                "ungrounded_accuracy": v["grouped_correctness"]["ungrounded_accuracy"],
                "separation": v["grouped_correctness"]["separation"],
                "gap_ci95": v["grounded_gap_bootstrap"]["ci95"],
                "confidence_auroc": v["confidence_auroc"],
                "aurc": v["selective_risk"]["aurc"],
                "hallucination_catch_rate": v["hallucination_proxy"]["hallucination_catch_rate"],
            }
            for ds, v in faith.items()
        },
        "verify_then_reask": {
            ds: {"raw_nm": v["raw_nm"], "verify_reask_nm": v["verify_reask_nm"],
                 "reasked": v["reasked"], "n": v["n"]}
            for ds, v in reask.items()
        },
        "reask_policy_sweep": {
            ds: v["policies"] for ds, v in reask_policy.items()
        },
        "learned_coordinate": {
            ds: v["all"] for ds, v in learned.items()
        },
        "global_failure_audit": failure,
    }
    path = Path("outputs/research/paper_ablation_report.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
