#!/usr/bin/env python3
"""CPR as a structure-grounded answer ROUTER (end-to-end Number-Match optimization).

The pipeline already produces two complementary answers per query: a RAW answer (from the
raw retrieved tables) and a KG answer (from structure/CPR-selected facts). Neither dominates,
and an oracle that picks the better one per query gains +9-13 NM points. This script tests
whether the CPR confidence — our structure-grounded reliability signal — can ROUTE to the
better answer, turning auditability into accuracy.

Routers compared (all annotation-free at inference):
  * raw-only / kg-only                       (single-view baselines)
  * legacy-router : pick argmax legacy verifier confidence
  * cpr-router    : pick argmax CPR confidence (ours)
  * oracle        : pick any correct answer  (upper bound)

Also reports selective Number-Match: NM among the top-coverage fraction by router confidence.
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
from gsr_cacl.ledger.numeric import number_match, numbers_close
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


def _union_ledger(pack) -> FactLedger | None:
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    merged = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        merged.facts.extend(d.ledger.facts)
    return merged


def _legacy_conf(vr) -> float:
    if vr is None or vr.pred_value is None:
        return 0.0
    return 1.0 if vr.grounded else (0.85 if vr.derivable else 0.1)


def _nm(pred_value, gold) -> bool:
    return bool(pred_value is not None and gold is not None and number_match(pred_value, gold))


def _sel_nm(items, conf_key, correct_key, cov):
    """Number-Match among the top-`cov` fraction ranked by confidence."""
    s = sorted(items, key=lambda r: r[conf_key], reverse=True)
    k = max(1, int(round(len(s) * cov)))
    kept = s[:k]
    return round(sum(r[correct_key] for r in kept) / max(len(kept), 1), 4)


def run(ds, pred_path, retr_path, limit):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}
    if limit:
        preds = preds[:limit]

    rows = []
    for p in preds:
        rec = by_id.get(str(p.get("query_id"))) or by_q.get(p.get("query"))
        if rec is None:
            continue
        retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
        if not retrieved:
            continue
        query, gold = p["query"], p.get("gold")
        pack = build_evidence_pack(query, retrieved, top_n_facts=8)
        ledger = _union_ledger(pack)
        if ledger is None:
            continue

        cands = []  # (name, pred_text, value, correct, cpr_conf, legacy_conf)
        for name in ("raw", "kg", "reask"):
            txt = p.get(f"{name}_prediction") or ""
            if not txt.strip():
                continue
            val = extract_final_number(txt)
            if val is None:
                continue
            cpr = verify_cpr(txt, ledger, query, gold=gold, selected_facts=pack.selected_facts)
            leg = verify(txt, ledger, query, gold=gold)
            cands.append({
                "name": name, "value": val, "correct": _nm(val, gold),
                "cpr_conf": cpr.confidence, "legacy_conf": _legacy_conf(leg),
                "cpr_level": cpr.level,
            })
        if not cands:
            continue

        raw_c = next((c for c in cands if c["name"] == "raw"), cands[0])
        kg_c = next((c for c in cands if c["name"] == "kg"), None)
        # raw-KG agreement (self-consistency-style reliability signal for the RAW answer)
        raw_c["agreement"] = 1.0 if (kg_c is not None and raw_c["value"] is not None
                                     and kg_c["value"] is not None
                                     and numbers_close(raw_c["value"], kg_c["value"])) else 0.0
        # routers (tie-break: prefer raw to avoid needless switching)
        def pick(conf_key):
            best = raw_c
            for c in cands:
                if c[conf_key] > best[conf_key] + 1e-9:
                    best = c
            return best
        cpr_pick = pick("cpr_conf")
        leg_pick = pick("legacy_conf")

        # Conservative high-precision override: keep raw unless an alternative answer is
        # strongly CPR-grounded (concept+period+role consistent) while raw is NOT supported.
        STRONG = {"cpr_grounded", "cpr_derivable", "scaled_grounded", "scaled_derivable"}
        cons_pick = raw_c
        if raw_c["cpr_level"] not in STRONG:
            alts = [c for c in cands if c["name"] != "raw" and c["cpr_level"] in STRONG]
            if alts:
                cons_pick = max(alts, key=lambda c: c["cpr_conf"])
        rows.append({
            "raw_correct": raw_c["correct"],
            "kg_correct": next((c["correct"] for c in cands if c["name"] == "kg"), False),
            "oracle_correct": any(c["correct"] for c in cands),
            "cpr_router_correct": cpr_pick["correct"],
            "legacy_router_correct": leg_pick["correct"],
            "cons_router_correct": cons_pick["correct"],
            "cpr_router_conf": cpr_pick["cpr_conf"],
            "cons_router_conf": cons_pick["cpr_conf"],
            "switched_from_raw": cpr_pick["name"] != "raw",
            "cons_switched": cons_pick["name"] != "raw",
            # reliability signals for the RAW answer (head-to-head AUROC)
            "raw_correct_flag": raw_c["correct"],
            "sig_legacy": raw_c["legacy_conf"],
            "sig_cpr": raw_c["cpr_conf"],
            "sig_agreement": raw_c["agreement"],
            "sig_cpr_plus_agreement": raw_c["cpr_conf"] + 0.5 * raw_c["agreement"],
        })

    n = len(rows)
    def acc(k): return round(sum(r[k] for r in rows) / max(n, 1), 4)
    switched = [r for r in rows if r["switched_from_raw"]]
    sw_help = sum(1 for r in switched if r["cpr_router_correct"] and not r["raw_correct"])
    sw_hurt = sum(1 for r in switched if not r["cpr_router_correct"] and r["raw_correct"])
    cons_sw = [r for r in rows if r["cons_switched"]]
    cons_help = sum(1 for r in cons_sw if r["cons_router_correct"] and not r["raw_correct"])
    cons_hurt = sum(1 for r in cons_sw if not r["cons_router_correct"] and r["raw_correct"])
    return {
        "dataset": ds, "n": n,
        "number_match": {
            "raw_only": acc("raw_correct"),
            "kg_only": acc("kg_correct"),
            "legacy_router": acc("legacy_router_correct"),
            "cpr_router_argmax": acc("cpr_router_correct"),
            "cpr_router_conservative": acc("cons_router_correct"),
            "oracle_best_of": acc("oracle_correct"),
        },
        "conservative_vs_raw": round(acc("cons_router_correct") - acc("raw_correct"), 4),
        "argmax_vs_raw": round(acc("cpr_router_correct") - acc("raw_correct"), 4),
        "switching_argmax": {"n_switched": len(switched), "helped": sw_help, "hurt": sw_hurt},
        "switching_conservative": {"n_switched": len(cons_sw), "helped": cons_help, "hurt": cons_hurt},
        "selective_nm_cpr_router": {
            f"cov{c}": _sel_nm(rows, "cpr_router_conf", "cpr_router_correct", c)
            for c in (0.25, 0.5, 0.75, 1.0)
        },
        "reliability_signal_auroc": {
            "legacy_value_only": round(_auroc([r["sig_legacy"] for r in rows], [r["raw_correct_flag"] for r in rows]), 4),
            "raw_kg_agreement": round(_auroc([r["sig_agreement"] for r in rows], [r["raw_correct_flag"] for r in rows]), 4),
            "cpr": round(_auroc([r["sig_cpr"] for r in rows], [r["raw_correct_flag"] for r in rows]), 4),
            "cpr_plus_agreement": round(_auroc([r["sig_cpr_plus_agreement"] for r in rows], [r["raw_correct_flag"] for r in rows]), 4),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/generation_system_q35_s400/{ds}_predictions.jsonl")
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/cpr_router")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for ds in args.datasets:
        r = run(ds, args.pred.format(ds=ds), args.retr.format(ds=ds), args.limit)
        allres[ds] = r
        (out / f"{ds}.json").write_text(json.dumps(r, indent=2))
        nm = r["number_match"]
        print(f"\n=== {ds} (n={r['n']}) ===")
        print(f"  NM: raw={nm['raw_only']} kg={nm['kg_only']} | oracle={nm['oracle_best_of']}")
        print(f"  cpr-router(argmax)={nm['cpr_router_argmax']} ({r['argmax_vs_raw']:+.4f})  "
              f"cpr-router(conservative)={nm['cpr_router_conservative']} ({r['conservative_vs_raw']:+.4f})")
        sc = r["switching_conservative"]
        print(f"  conservative switches {sc['n_switched']}: helped {sc['helped']}, hurt {sc['hurt']}")
        print(f"  selective NM (CPR-router): {r['selective_nm_cpr_router']}")
        sa = r["reliability_signal_auroc"]
        print(f"  reliability AUROC: value-only={sa['legacy_value_only']}  agreement={sa['raw_kg_agreement']}  "
              f"CPR={sa['cpr']}  CPR+agree={sa['cpr_plus_agreement']}")
    (out / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
