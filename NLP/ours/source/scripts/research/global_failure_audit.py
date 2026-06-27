#!/usr/bin/env python3
"""Generate a candid global failure audit for the current research system."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from gsr_cacl.ledger.select import infer_task_type
from gsr_cacl.ledger.numeric import number_match


DATASETS = {
    "finqa": {
        "pred": "outputs/final_generation/qwen3_5/finqa/predictions.jsonl",
        "top3": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "reask": "outputs/research/verify_reask/finqa_predictions.jsonl",
    },
    "convfinqa": {
        "pred": "outputs/final_generation/qwen3_5/convfinqa/predictions.jsonl",
        "top3": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "reask": "outputs/research/verify_reask/convfinqa_predictions.jsonl",
    },
    "tatqa": {
        "pred": "outputs/final_generation/qwen3_5/tatqa/predictions.jsonl",
        "top3": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
        "reask": "outputs/research/verify_reask/tatqa_predictions.jsonl",
    },
}


def _jsonl(path: str):
    return [json.loads(l) for l in open(path) if l.strip()]


def _retrieval_audit(rows):
    rank_counts = Counter()
    task_miss = Counter()
    for r in rows:
        gt = str(r.get("ground_truth_id") or "")
        ids = [str(d.get("context_id") or d.get("id")) for d in r.get("retrieved") or []]
        rank = ids.index(gt) + 1 if gt in ids else 0
        rank_counts[rank] += 1
        if rank == 0:
            task_miss[infer_task_type(r.get("query", ""))] += 1
    n = len(rows)
    return {
        "n": n,
        "top1": round(rank_counts[1] / max(n, 1), 4),
        "top3": round(sum(rank_counts[i] for i in (1, 2, 3)) / max(n, 1), 4),
        "miss": rank_counts[0],
        "miss_by_task": dict(task_miss.most_common()),
    }


def _generation_audit(rows, reask_rows):
    buckets = Counter()
    task_buckets = Counter()
    examples = {"grounded_wrong": [], "ungrounded_correct": [], "reask_worsened": []}
    for i, r in enumerate(rows):
        correct = bool(r.get("answer_match", False))
        if not correct and r.get("gold") is not None and r.get("pred_value") is not None:
            correct = bool(number_match(r.get("pred_value"), r.get("gold")))
        grounded = bool(r.get("grounded", False))
        if grounded and correct:
            bucket = "grounded_correct"
        elif grounded and not correct:
            bucket = "grounded_wrong"
        elif (not grounded) and correct:
            bucket = "ungrounded_correct"
        else:
            bucket = "ungrounded_wrong"
        buckets[bucket] += 1
        task_buckets[(infer_task_type(r.get("query", "")), bucket)] += 1
        if bucket in examples and len(examples[bucket]) < 5:
            examples[bucket].append({
                "query": r.get("query", "")[:240],
                "gold": r.get("gold"),
                "prediction": str(r.get("prediction", ""))[:240],
                "explanation": r.get("explanation", ""),
            })
        if i < len(reask_rows):
            raw_ok = bool(reask_rows[i].get("raw_correct", False))
            final_ok = bool(reask_rows[i].get("final_correct", False))
            if raw_ok and not final_ok and len(examples["reask_worsened"]) < 5:
                examples["reask_worsened"].append({
                    "query": r.get("query", "")[:240],
                    "gold": r.get("gold"),
                    "raw_answer": reask_rows[i].get("raw_answer"),
                    "final_answer": reask_rows[i].get("final_answer"),
                })
    n = len(rows)
    return {
        "n": n,
        "bucket_counts": dict(buckets),
        "bucket_frac": {k: round(v / max(n, 1), 4) for k, v in buckets.items()},
        "task_bucket_counts": {f"{k[0]}::{k[1]}": v for k, v in task_buckets.most_common()},
        "examples": examples,
    }


def main():
    out = {"datasets": {}}
    for ds, paths in DATASETS.items():
        preds = _jsonl(paths["pred"])
        top3 = _jsonl(paths["top3"])
        reask = _jsonl(paths["reask"])
        out["datasets"][ds] = {
            "retrieval": _retrieval_audit(top3),
            "generation": _generation_audit(preds, reask),
        }
    out_path = Path("outputs/research/global_failure_audit.json")
    out_path.write_text(json.dumps(out, indent=2))

    md = ["# Global Failure Audit\n"]
    for ds, d in out["datasets"].items():
        md.append(f"## {ds}\n")
        r = d["retrieval"]
        md.append(f"- Retrieval top1={r['top1']}, top3={r['top3']}, miss={r['miss']}\n")
        g = d["generation"]
        md.append(f"- Generation buckets: `{g['bucket_counts']}`\n")
        md.append(f"- Main retrieval misses by task: `{r['miss_by_task']}`\n")
    md_path = Path("outputs/research/global_failure_audit.md")
    md_path.write_text("\n".join(md))
    print(md_path.read_text())
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
