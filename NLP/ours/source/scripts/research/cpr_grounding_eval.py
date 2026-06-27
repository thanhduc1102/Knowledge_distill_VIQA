#!/usr/bin/env python3
"""Offline replay: does Concept-Period-Role (CPR) grounding separate correct answers
better than the legacy value-only grounding?

No GPU / no LLM. Re-verifies cached generation predictions against the retrieved
fact ledgers using (a) the legacy value-only verifier and (b) the CPR verifier, and
reports the faithfulness/selective-prediction diagnostics that matter for the paper:

  * separation  = acc(supported) - acc(unsupported)         [higher = better signal]
  * AUROC       = confidence vs answer-correct               [higher = better calibration]
  * over-firing = supported-but-wrong count                  [lower = fewer grounded_wrong]
  * downgrade precision = of legacy supported-wrong, how many CPR moves to unsupported.
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

PRED_DEFAULT = "outputs/research/generation_system_7b/{ds}_predictions.jsonl"
RETR_DEFAULT = "outputs/final_retrieval/{ds}/retrieval_top3.jsonl"


def _union_ledger(pack) -> FactLedger | None:
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    merged = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        merged.facts.extend(d.ledger.facts)
    return merged


def _auroc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based AUROC with tie handling. labels True = positive (correct)."""
    pos = sum(1 for l in labels if l)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _paired_bootstrap(rows, n_boot=2000, seed=0):
    """Paired bootstrap CIs for legacy/CPR AUROC, separation, and their deltas.

    Resampling the same indices for both methods makes the delta CI a valid paired test:
    if the 95% CI of (CPR - legacy) excludes 0, the improvement is significant.
    """
    import random
    rng = random.Random(seed)
    n = len(rows)
    idx_all = list(range(n))
    keys = ("legacy_auroc", "cpr_auroc", "d_auroc", "legacy_sep", "cpr_sep", "d_sep")
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = [rng.choice(idx_all) for _ in range(n)]
        sub = [rows[i] for i in idx]
        corr = [r["correct"] for r in sub]
        la = _auroc([r["legacy_conf"] for r in sub], corr)
        ca = _auroc([r["cpr_conf"] for r in sub], corr)
        ls = _sep(sub, "legacy_supported")["separation"]
        cs = _sep(sub, "cpr_supported")["separation"]
        if la != la or ca != ca:  # nan guard (degenerate resample)
            continue
        samples["legacy_auroc"].append(la); samples["cpr_auroc"].append(ca)
        samples["d_auroc"].append(ca - la)
        samples["legacy_sep"].append(ls); samples["cpr_sep"].append(cs)
        samples["d_sep"].append(cs - ls)

    def ci(vals):
        if not vals:
            return [None, None]
        s = sorted(vals)
        lo = s[int(0.025 * len(s))]; hi = s[min(len(s) - 1, int(0.975 * len(s)))]
        return [round(lo, 4), round(hi, 4)]

    out = {k: ci(v) for k, v in samples.items()}
    out["d_auroc_p_gt_0"] = round(sum(1 for v in samples["d_auroc"] if v > 0) / max(len(samples["d_auroc"]), 1), 4)
    out["d_sep_p_gt_0"] = round(sum(1 for v in samples["d_sep"] if v > 0) / max(len(samples["d_sep"]), 1), 4)
    return out


def _risk_coverage(rows, conf_key, correct_key="correct"):
    scored = sorted(rows, key=lambda r: r[conf_key], reverse=True)
    n = len(scored)
    curve = []
    for cov in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(round(n * cov)))
        kept = scored[:k]
        acc = sum(r[correct_key] for r in kept) / max(len(kept), 1)
        curve.append({"coverage": cov, "accuracy": round(acc, 4), "risk": round(1 - acc, 4)})
    # area under accuracy-coverage (selective accuracy AUC; higher is better)
    aacc = sum(c["accuracy"] for c in curve) / len(curve)
    return {"curve": curve, "selective_acc_auc": round(aacc, 4)}


def _sep(rows, supported_key, correct_key="correct"):
    sup = [r for r in rows if r[supported_key]]
    uns = [r for r in rows if not r[supported_key]]
    a_s = sum(r[correct_key] for r in sup) / max(len(sup), 1)
    a_u = sum(r[correct_key] for r in uns) / max(len(uns), 1)
    return {
        "supported_n": len(sup),
        "acc_supported": round(a_s, 4),
        "unsupported_n": len(uns),
        "acc_unsupported": round(a_u, 4),
        "separation": round(a_s - a_u, 4),
        "supported_but_wrong": sum(1 for r in sup if not r[correct_key]),
    }


def run(ds: str, pred_path: str, retr_path: str, limit: int | None) -> dict:
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}

    rows = []
    missing = 0
    for p in preds:
        rec = by_id.get(str(p.get("query_id"))) or by_q.get(p.get("query"))
        if rec is None:
            missing += 1
            continue
        retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
        if not retrieved:
            missing += 1
            continue
        query = p["query"]
        gold = p.get("gold")
        pred_text = p.get("raw_prediction") or p.get("prediction") or ""
        if "raw_correct" in p:
            correct = bool(p.get("raw_correct"))
        else:
            pv = extract_final_number(pred_text)
            correct = bool(pv is not None and gold is not None and number_match(pv, gold))
        pack = build_evidence_pack(query, retrieved, top_n_facts=8)
        ledger = _union_ledger(pack)
        if ledger is None:
            missing += 1
            continue
        vr = verify(pred_text, ledger, query, gold=gold)
        cpr = verify_cpr(pred_text, ledger, query, gold=gold, selected_facts=pack.selected_facts)
        rows.append({
            "correct": correct,
            "legacy_supported": bool(vr.grounded or vr.derivable),
            "legacy_conf": float(
                1.0 if vr.grounded else (0.85 if vr.derivable else (0.1 if vr.pred_value is not None else 0.0))
            ),
            "cpr_supported": cpr.supported,
            "cpr_conf": cpr.confidence,
            "cpr_level": cpr.level,
            "value_only_grounded": cpr.value_only_grounded,
            "value_only_derivable": cpr.value_only_derivable,
            "concept_consistency": round(cpr.concept_consistency, 4),
            "period_consistency": round(cpr.period_consistency, 4),
            "cpr_grounded": cpr.grounded,
            "cpr_derivable": cpr.derivable,
        })

    n = len(rows)
    legacy = _sep(rows, "legacy_supported")
    cpr = _sep(rows, "cpr_supported")
    legacy["auroc"] = round(_auroc([r["legacy_conf"] for r in rows], [r["correct"] for r in rows]), 4)
    cpr["auroc"] = round(_auroc([r["cpr_conf"] for r in rows], [r["correct"] for r in rows]), 4)
    legacy["risk_coverage"] = _risk_coverage(rows, "legacy_conf")
    cpr["risk_coverage"] = _risk_coverage(rows, "cpr_conf")
    boot = _paired_bootstrap(rows)

    legacy_sup_wrong = [r for r in rows if r["legacy_supported"] and not r["correct"]]
    downgraded = [r for r in legacy_sup_wrong if not r["cpr_supported"]]
    # did we wrongly drop any legacy-supported-correct?
    legacy_sup_right = [r for r in rows if r["legacy_supported"] and r["correct"]]
    dropped_right = [r for r in legacy_sup_right if not r["cpr_supported"]]

    level_dist: dict[str, int] = {}
    for r in rows:
        level_dist[r["cpr_level"]] = level_dist.get(r["cpr_level"], 0) + 1

    summary = {
        "dataset": ds,
        "n": n,
        "missing_join": missing,
        "raw_accuracy": round(sum(r["correct"] for r in rows) / max(n, 1), 4),
        "legacy_value_only": legacy,
        "cpr": cpr,
        "bootstrap_ci95": boot,
        "over_firing_reduction": {
            "legacy_supported_wrong": len(legacy_sup_wrong),
            "cpr_downgraded_of_those": len(downgraded),
            "downgrade_precision": round(len(downgraded) / max(len(legacy_sup_wrong), 1), 4),
            "legacy_supported_correct": len(legacy_sup_right),
            "wrongly_dropped_correct": len(dropped_right),
        },
        "cpr_level_distribution": level_dist,
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default=PRED_DEFAULT)
    ap.add_argument("--retr", default=RETR_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/cpr_grounding")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    allres = {}
    for ds in args.datasets:
        res, rows = run(ds, args.pred.format(ds=ds), args.retr.format(ds=ds), args.limit)
        allres[ds] = res
        (out_dir / f"{ds}.json").write_text(json.dumps(res, indent=2))
        with open(out_dir / f"{ds}_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        L, C = res["legacy_value_only"], res["cpr"]
        print(f"\n=== {ds} (n={res['n']}, raw_acc={res['raw_accuracy']}) ===")
        print(f"  legacy: sep={L['separation']:+.3f} auroc={L['auroc']:.3f} "
              f"sup={L['supported_n']} sup_wrong={L['supported_but_wrong']}")
        print(f"  CPR   : sep={C['separation']:+.3f} auroc={C['auroc']:.3f} "
              f"sup={C['supported_n']} sup_wrong={C['supported_but_wrong']}")
        ofr = res["over_firing_reduction"]
        print(f"  over-firing: downgraded {ofr['cpr_downgraded_of_those']}/{ofr['legacy_supported_wrong']} "
              f"legacy-wrong (prec={ofr['downgrade_precision']}), dropped {ofr['wrongly_dropped_correct']} correct")
        b = res["bootstrap_ci95"]
        print(f"  bootstrap: dAUROC={b['d_auroc']} (P(>0)={b['d_auroc_p_gt_0']}) "
              f"dSep={b['d_sep']} (P(>0)={b['d_sep_p_gt_0']})")
    (out_dir / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out_dir}/")


if __name__ == "__main__":
    main()
