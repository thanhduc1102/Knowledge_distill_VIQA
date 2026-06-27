#!/usr/bin/env python3
"""Evaluate structure-level KG support on retrieved top-k documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.ledger.extract import extract_ledger, extract_ledger_from_table
from gsr_cacl.kg.structure_graph import build_structure_graph, score_structure_support

DEFAULTS = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}


def _doc_id(d: dict) -> str:
    return str(d.get("context_id") or d.get("id") or "")


def _ledger_for(d: dict):
    meta = d.get("meta") or d.get("metadata") or {}
    doc_id = _doc_id(d)
    table = d.get("table") or ""
    context = d.get("page_content") or d.get("context") or ""
    if table:
        return extract_ledger_from_table(table, doc_id=doc_id, meta=meta, caption=context[:300])
    return extract_ledger(context=context, doc_id=doc_id, meta=meta)


def evaluate(path: str, limit: int = 0) -> dict:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if limit:
        recs = recs[:limit]
    weights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    result = {f"structure+rankprior_{w:.2f}": {"hit": 0, "changed": 0, "rescue": 0, "harm": 0}
              for w in weights}
    original_hit = topk_hit = structure_hit = 0
    stats_acc = []
    score_gold, score_non = [], []
    examples = {"rescue": [], "harm": [], "paths": []}

    for rec in recs:
        gt = str(rec.get("ground_truth_id") or "")
        q = rec.get("query") or rec.get("raw_question") or ""
        docs = (rec.get("retrieved") or [])[:3]
        ids = [_doc_id(d) for d in docs]
        if not docs:
            continue
        orig = ids[0]
        orig_ok = orig == gt
        original_hit += int(orig_ok)
        in_topk = gt in ids
        topk_hit += int(in_topk)

        scored = []
        for rank, d in enumerate(docs, 1):
            led = _ledger_for(d)
            sg = build_structure_graph(led)
            sup = score_structure_support(q, sg)
            scored.append((d, rank, sg, sup))
            stats_acc.append(sg.stats())
            if _doc_id(d) == gt:
                score_gold.append(sup.score)
            else:
                score_non.append(sup.score)

        best = max(scored, key=lambda x: x[3].score)
        structure_hit += int(_doc_id(best[0]) == gt)
        if len(examples["paths"]) < 5:
            examples["paths"].append({
                "query": q,
                "doc": _doc_id(best[0]),
                "score": round(best[3].score, 4),
                "paths": best[3].evidence_paths[:3],
                "reasons": best[3].reasons,
            })
        if (not orig_ok) and in_topk and _doc_id(best[0]) == gt and len(examples["rescue"]) < 5:
            examples["rescue"].append({"query": q, "orig": orig, "chosen": _doc_id(best[0]),
                                       "score": best[3].score, "reasons": best[3].reasons})
        if orig_ok and _doc_id(best[0]) != gt and len(examples["harm"]) < 5:
            examples["harm"].append({"query": q, "chosen": _doc_id(best[0]),
                                     "score": best[3].score, "reasons": best[3].reasons})

        for w in weights:
            key = f"structure+rankprior_{w:.2f}"
            chosen = max(scored, key=lambda x: x[3].score + w / max(x[1], 1))
            cid = _doc_id(chosen[0])
            result[key]["hit"] += int(cid == gt)
            result[key]["changed"] += int(cid != orig)
            result[key]["rescue"] += int((not orig_ok) and in_topk and cid == gt)
            result[key]["harm"] += int(orig_ok and cid != gt)

    n = len(recs)
    avg_stats = {}
    if stats_acc:
        keys = sorted({k for s in stats_acc for k in s})
        avg_stats = {k: round(float(np.mean([s.get(k, 0) for s in stats_acc])), 3) for k in keys}
    policies = {
        k: {
            "top1_acc": round(v["hit"] / max(n, 1), 4),
            "delta": round((v["hit"] - original_hit) / max(n, 1), 4),
            "changed_frac": round(v["changed"] / max(n, 1), 4),
            "rescue": v["rescue"],
            "harm": v["harm"],
        }
        for k, v in result.items()
    }
    best_policy = max(policies.items(), key=lambda kv: kv[1]["top1_acc"])
    return {
        "path": path,
        "n": n,
        "top3_recall": round(topk_hit / max(n, 1), 4),
        "original_top1_acc": round(original_hit / max(n, 1), 4),
        "structure_only_top1_acc": round(structure_hit / max(n, 1), 4),
        "score_gold_mean": round(float(np.mean(score_gold)), 4) if score_gold else 0.0,
        "score_non_gold_mean": round(float(np.mean(score_non)), 4) if score_non else 0.0,
        "avg_graph_stats": avg_stats,
        "policies": policies,
        "best_policy": {"name": best_policy[0], **best_policy[1]},
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/structure_graph")
    args = ap.parse_args()

    res = evaluate(args.input or DEFAULTS[args.dataset], args.limit)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{args.dataset}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\n=== Structure graph eval — {args.dataset} ===")
    print(f"orig={res['original_top1_acc']:.4f} struct_only={res['structure_only_top1_acc']:.4f} "
          f"top3={res['top3_recall']:.4f}")
    print(f"gold_score={res['score_gold_mean']:.4f} non_gold_score={res['score_non_gold_mean']:.4f}")
    print(f"best={res['best_policy']}")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
