#!/usr/bin/env python3
"""Replay verify-then-reask policy over saved generation predictions.

Use this after improving the verifier: it rebuilds the ledger/evidence packs from the
original retrieval JSONL, re-verifies saved RAW/KG/REASK predictions, and recomputes the
deployable policy without spending GPU time on generation again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import verify
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.select import infer_task_type

from scripts.research.generation_system_eval import DEFAULTS, summarize, support_confidence


def _get_retrieved(rec: dict) -> list[dict]:
    if "retrieved" in rec:
        return rec["retrieved"]
    return rec.get("retrieved_docs", [])


def _gold(rec: dict):
    return rec.get("gold") or rec.get("answers") or rec.get("answer")


def union_ledger(pack) -> FactLedger | None:
    if not pack.ranked:
        return None
    merged = FactLedger(doc_id="union", facts=list(pack.ranked[0].ledger.facts), meta=dict(pack.ranked[0].ledger.meta))
    for doc in pack.ranked[1:]:
        merged.facts.extend(doc.ledger.facts)
    return merged


def _load_retrieval(path: str) -> dict:
    records = [json.loads(line) for line in open(path) if line.strip()]
    by_qid = {str(r.get("query_id")): r for r in records if r.get("query_id") is not None}
    by_query = {r.get("query"): r for r in records}
    return {"by_qid": by_qid, "by_query": by_query}


def _find_retrieval(row: dict, idx: dict) -> dict:
    qid = row.get("query_id")
    if qid is not None and str(qid) in idx["by_qid"]:
        return idx["by_qid"][str(qid)]
    return idx["by_query"][row["query"]]


def replay(dataset: str, predictions: str, retrieval: str, top_n_facts: int) -> tuple[dict, list[dict]]:
    idx = _load_retrieval(retrieval)
    rows = [json.loads(line) for line in open(predictions) if line.strip()]
    out_rows: list[dict] = []

    for row in rows:
        rec = _find_retrieval(row, idx)
        pack = build_evidence_pack(rec["query"], _get_retrieved(rec), top_n_facts=top_n_facts)
        ledger = union_ledger(pack)
        gold = _gold(rec)

        raw_vr = verify(row["raw_prediction"], ledger, rec["query"], gold=gold) if ledger else None
        kg_vr = verify(row["kg_prediction"], ledger, rec["query"], gold=gold) if ledger else None
        reask_pred = row.get("reask_prediction") or ""
        reask_vr = verify(reask_pred, ledger, rec["query"], gold=gold) if (ledger and reask_pred) else None
        raw_supported = bool(raw_vr and (raw_vr.grounded or raw_vr.derivable))
        final_pred = row["raw_prediction"] if raw_supported else (reask_pred or row["kg_prediction"])
        final_vr = raw_vr if raw_supported else (reask_vr or kg_vr)

        new = dict(row)
        new.update({
            "task": infer_task_type(rec["query"]),
            "raw_pred_value": raw_vr.pred_value if raw_vr else None,
            "kg_pred_value": kg_vr.pred_value if kg_vr else None,
            "raw_correct": bool(raw_vr.answer_match if raw_vr else False),
            "kg_correct": bool(kg_vr.answer_match if kg_vr else False),
            "raw_grounded": bool(raw_vr.grounded if raw_vr else False),
            "raw_derivable": bool(raw_vr.derivable if raw_vr else False),
            "raw_supported": raw_supported,
            "raw_confidence": support_confidence(raw_vr),
            "raw_explanation": raw_vr.explanation if raw_vr else "",
            "kg_grounded": bool(kg_vr.grounded if kg_vr else False),
            "kg_derivable": bool(kg_vr.derivable if kg_vr else False),
            "kg_confidence": support_confidence(kg_vr),
            "kg_explanation": kg_vr.explanation if kg_vr else "",
            "reask_triggered": not raw_supported,
            "reask_correct": bool(reask_vr.answer_match if reask_vr else False),
            "reask_confidence": support_confidence(reask_vr),
            "reask_explanation": reask_vr.explanation if reask_vr else "",
            "final_prediction": final_pred,
            "final_correct": bool(final_vr.answer_match if final_vr else False),
            "final_confidence": support_confidence(final_vr),
        })
        out_rows.append(new)

    args = SimpleNamespace(
        dataset=dataset,
        input=retrieval,
        model=rows[0].get("model", "") if rows else "",
        sample=len(rows),
        seed=None,
        device="replay",
        dtype="replay",
        batch_size=0,
        max_new_tokens=0,
        max_input_tokens=0,
    )
    summary = summarize(out_rows, args, elapsed=0.0)
    summary["predictions"] = predictions
    summary["replay_note"] = "No generation was rerun; saved predictions were reverified with the current verifier."
    summary["task_breakdown"] = {}
    for task in sorted({r["task"] for r in out_rows}):
        rs = [r for r in out_rows if r["task"] == task]
        summary["task_breakdown"][task] = {
            "n": len(rs),
            "raw": round(sum(r["raw_correct"] for r in rs) / len(rs), 4),
            "kg": round(sum(r["kg_correct"] for r in rs) / len(rs), 4),
            "final": round(sum(r["final_correct"] for r in rs) / len(rs), 4),
            "supported_rate": round(sum(r["raw_supported"] for r in rs) / len(rs), 4),
        }
    n = max(len(out_rows), 1)
    summary["raw_kg_arbitration"] = {
        "oracle_raw_or_kg": round(sum(r["raw_correct"] or r["kg_correct"] for r in out_rows) / n, 4),
        "confidence_margin": {},
        "task_diagnostics": {},
    }
    for margin in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        preds = []
        use_kg = 0
        for r in out_rows:
            choose_kg = r["kg_confidence"] > r["raw_confidence"] + margin
            use_kg += int(choose_kg)
            preds.append(r["kg_correct"] if choose_kg else r["raw_correct"])
        summary["raw_kg_arbitration"]["confidence_margin"][f"{margin:+.1f}"] = {
            "accuracy": round(sum(preds) / n, 4),
            "use_kg_rate": round(use_kg / n, 4),
        }
    # Post-hoc diagnostics only. These are not final claims until tuned on a held-out
    # validation split, but they expose where KG context is genuinely useful.
    task_sets = {
        "difference_only": {"difference"},
        "difference_average": {"difference", "average"},
        "difference_percent_change": {"difference", "percent_change"},
        "broad_structure": {"difference", "percent_change", "lookup", "comparison", "sum"},
    }
    for label, tasks in task_sets.items():
        preds = []
        use_kg = 0
        for r in out_rows:
            choose_kg = r["task"] in tasks
            use_kg += int(choose_kg)
            preds.append(r["kg_correct"] if choose_kg else r["raw_correct"])
        summary["raw_kg_arbitration"]["task_diagnostics"][label] = {
            "accuracy": round(sum(preds) / n, 4),
            "use_kg_rate": round(use_kg / n, 4),
        }
    return summary, out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--retrieval", default=None)
    ap.add_argument("--top-n-facts", type=int, default=8)
    ap.add_argument("--out", default="outputs/research/generation_system_7b_scaleaware")
    args = ap.parse_args()

    retrieval = args.retrieval or DEFAULTS[args.dataset]
    summary, rows = replay(args.dataset, args.predictions, retrieval, args.top_n_facts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.dataset}.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / f"{args.dataset}_predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
