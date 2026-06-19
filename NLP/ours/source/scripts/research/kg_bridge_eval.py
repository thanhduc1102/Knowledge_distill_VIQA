#!/usr/bin/env python3
"""Evaluate KG evidence arbitration over existing retrieval top-k outputs.

This script tests the three inference-time KG goals before running an LLM:

1. Does the KG focus the correct document among noisy top-k?
2. Can it compute/carry a symbolic answer from selected facts?
3. Does it expose auditable provenance/conflicts for generation?

Input is a retrieval JSONL with records containing ``query``, ``ground_truth_id``, ``gold`` and
``retrieved`` docs. Output is a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.ledger.numeric import number_match


DEFAULTS = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}


def _doc_id(d: dict) -> str:
    return str(d.get("context_id") or d.get("id") or "")


def evaluate(path: str, top_n_facts: int = 6, limit: int = 0) -> dict:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    n = len(records)
    orig_hit = kg_hit = topk_hit = 0
    rescue = harm = changed = 0
    symbolic_has = symbolic_nm = 0
    raw_symbolic_has = raw_symbolic_nm = 0
    tiers: dict[str, int] = {}
    tier_hits: dict[str, int] = {}
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    policy_hits = {f"override_margin>={t:.2f}": 0 for t in thresholds}
    policy_changed = {k: 0 for k in policy_hits}
    policy_rescue = {k: 0 for k in policy_hits}
    policy_harm = {k: 0 for k in policy_hits}
    policy_hits["trust_only"] = 0
    policy_changed["trust_only"] = 0
    policy_rescue["trust_only"] = 0
    policy_harm["trust_only"] = 0
    prior_weights = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    for w in prior_weights:
        key = f"kg+rankprior_{w:.2f}"
        policy_hits[key] = 0
        policy_changed[key] = 0
        policy_rescue[key] = 0
        policy_harm[key] = 0
    examples = {"rescue": [], "harm": [], "symbolic_fail": []}

    for rec in records:
        gt = str(rec.get("ground_truth_id") or "")
        retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
        retrieved = retrieved[:3]
        ids = [_doc_id(d) for d in retrieved]
        orig = ids[0] if ids else ""
        orig_ok = orig == gt
        in_topk = gt in ids
        topk_hit += int(in_topk)
        orig_hit += int(orig_ok)

        pack = build_evidence_pack(rec["query"], retrieved, top_n_facts=top_n_facts)
        kg = pack.ranked[0].doc_id if pack.ranked else ""
        kg_ok = kg == gt
        kg_hit += int(kg_ok)
        changed += int(kg != orig)
        rescue += int((not orig_ok) and in_topk and kg_ok)
        harm += int(orig_ok and not kg_ok)

        tiers[pack.tier] = tiers.get(pack.tier, 0) + 1
        tier_hits[pack.tier] = tier_hits.get(pack.tier, 0) + int(kg_ok)

        for t in thresholds:
            key = f"override_margin>={t:.2f}"
            chosen = kg if pack.margin >= t else orig
            policy_hits[key] += int(chosen == gt)
            policy_changed[key] += int(chosen != orig)
            policy_rescue[key] += int((not orig_ok) and in_topk and chosen == gt)
            policy_harm[key] += int(orig_ok and chosen != gt)
        key = "trust_only"
        chosen = kg if pack.tier == "TRUST_TOP1" else orig
        policy_hits[key] += int(chosen == gt)
        policy_changed[key] += int(chosen != orig)
        policy_rescue[key] += int((not orig_ok) and in_topk and chosen == gt)
        policy_harm[key] += int(orig_ok and chosen != gt)

        for w in prior_weights:
            key = f"kg+rankprior_{w:.2f}"
            best_doc = ""
            best_score = float("-inf")
            for d in pack.ranked:
                # Prior favours original high-ranked retrieval docs but still lets KG evidence win.
                rank_bonus = w / max(d.original_rank, 1)
                s = d.score + rank_bonus
                if s > best_score:
                    best_score = s
                    best_doc = d.doc_id
            chosen = best_doc or orig
            policy_hits[key] += int(chosen == gt)
            policy_changed[key] += int(chosen != orig)
            policy_rescue[key] += int((not orig_ok) and in_topk and chosen == gt)
            policy_harm[key] += int(orig_ok and chosen != gt)

        calc = pack.calculation or {}
        ans = calc.get("answer")
        if ans is not None:
            ok = number_match(ans, rec.get("gold"))
            raw_symbolic_has += 1
            raw_symbolic_nm += int(ok)
            if float(calc.get("confidence", 0.0) or 0.0) >= 0.8:
                symbolic_has += 1
                symbolic_nm += int(ok)
            if not ok and float(calc.get("confidence", 0.0) or 0.0) >= 0.8 and len(examples["symbolic_fail"]) < 5:
                examples["symbolic_fail"].append({
                    "query": rec.get("query"),
                    "gt": gt,
                    "kg_doc": kg,
                    "gold": rec.get("gold"),
                    "symbolic_answer": ans,
                    "confidence": calc.get("confidence"),
                    "trace": calc.get("trace"),
                })

        if (not orig_ok) and in_topk and kg_ok and len(examples["rescue"]) < 5:
            examples["rescue"].append({
                "query": rec.get("query"),
                "gt": gt,
                "orig": orig,
                "kg": kg,
                "tier": pack.tier,
                "reasons": pack.ranked[0].reasons if pack.ranked else [],
            })
        if orig_ok and not kg_ok and len(examples["harm"]) < 5:
            examples["harm"].append({
                "query": rec.get("query"),
                "gt": gt,
                "kg": kg,
                "tier": pack.tier,
                "reasons": pack.ranked[0].reasons if pack.ranked else [],
            })

    tier_acc = {
        t: round(tier_hits.get(t, 0) / max(c, 1), 4)
        for t, c in sorted(tiers.items())
    }
    policies = {
        k: {
            "top1_acc": round(v / max(n, 1), 4),
            "delta": round((v - orig_hit) / max(n, 1), 4),
            "changed_frac": round(policy_changed[k] / max(n, 1), 4),
            "rescue": policy_rescue[k],
            "harm": policy_harm[k],
        }
        for k, v in policy_hits.items()
    }
    best_policy = max(policies.items(), key=lambda kv: kv[1]["top1_acc"]) if policies else None

    return {
        "path": path,
        "n": n,
        "top3_recall": round(topk_hit / max(n, 1), 4),
        "original_top1_acc": round(orig_hit / max(n, 1), 4),
        "kg_top1_acc": round(kg_hit / max(n, 1), 4),
        "delta_top1": round((kg_hit - orig_hit) / max(n, 1), 4),
        "changed_frac": round(changed / max(n, 1), 4),
        "rescue_count": rescue,
        "harm_count": harm,
        "symbolic_coverage": round(symbolic_has / max(n, 1), 4),
        "symbolic_nm_all": round(symbolic_nm / max(n, 1), 4),
        "symbolic_nm_when_available": round(symbolic_nm / max(symbolic_has, 1), 4),
        "raw_symbolic_coverage": round(raw_symbolic_has / max(n, 1), 4),
        "raw_symbolic_nm_all": round(raw_symbolic_nm / max(n, 1), 4),
        "tiers": tiers,
        "tier_kg_acc": tier_acc,
        "policies": policies,
        "best_policy": {"name": best_policy[0], **best_policy[1]} if best_policy else None,
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--top-n-facts", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/kg_bridge")
    args = ap.parse_args()

    inp = args.input or DEFAULTS[args.dataset]
    result = evaluate(inp, top_n_facts=args.top_n_facts, limit=args.limit)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{args.dataset}.json"
    outpath.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)[:5000])
    print(f"Saved -> {outpath}")


if __name__ == "__main__":
    main()
