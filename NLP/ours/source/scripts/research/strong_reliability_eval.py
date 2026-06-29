#!/usr/bin/env python3
"""Strong-generator reliability evaluation (Gemini 2.5 Flash).

Head-to-head of four annotation-free reliability signals predicting raw-answer
correctness, on a strong generator (the regime Blind Spot #3 demanded):

  1. value_only      - legacy verifier (number present / any-pair derivable)
  2. cpr             - Concept-Period-Role structure grounding (ours)
  3. self_consistency- fraction of k sampled answers agreeing with the temp-0 answer
  4. verbalized      - the model's own stated confidence in [0,1]

Metrics: AUROC, selective-accuracy AUC (risk-coverage), coverage at fixed accuracy
bars, separation, and paired-bootstrap CIs for (CPR - each baseline) AUROC.

No GPU. Rebuilds the fact ledger from the retrieval records, identical to the
deployed pipeline, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import verify, extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, numbers_close
from gsr_cacl.research.cpr_verifier import verify_cpr

RETR = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}
SIGNALS = ["value_only", "cpr", "self_consistency", "verbalized"]


def _auroc(scores, labels):
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
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _selective_acc_auc(conf, correct, points=(0.25, 0.5, 0.75, 1.0)):
    order = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
    n = len(order)
    accs, curve = [], []
    for cov in points:
        k = max(1, int(round(n * cov)))
        kept = order[:k]
        acc = sum(correct[i] for i in kept) / len(kept)
        accs.append(acc)
        curve.append({"coverage": cov, "accuracy": round(acc, 4)})
    return sum(accs) / len(accs), curve


def _coverage_at_acc(conf, correct, bars=(0.6, 0.7, 0.8)):
    """Max coverage achievable with selective accuracy >= bar (answer top-conf first)."""
    order = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
    n = len(order)
    out = {}
    for bar in bars:
        best_cov = 0.0
        cum_correct = 0
        for k in range(1, n + 1):
            cum_correct += int(correct[order[k - 1]])
            acc = cum_correct / k
            if acc >= bar:
                best_cov = k / n
        out[f"cov@acc>={bar}"] = round(best_cov, 4)
    return out


def _sep(conf, correct, thr):
    sup = [i for i in range(len(conf)) if conf[i] >= thr]
    uns = [i for i in range(len(conf)) if conf[i] < thr]
    a_s = sum(correct[i] for i in sup) / max(len(sup), 1)
    a_u = sum(correct[i] for i in uns) / max(len(uns), 1)
    return {"acc_supported": round(a_s, 4), "acc_unsupported": round(a_u, 4),
            "separation": round(a_s - a_u, 4), "supported_n": len(sup),
            "supported_but_wrong": sum(1 for i in sup if not correct[i])}


def _cv_logistic(rows, feat_keys, out_key, folds=5, seed=0):
    """Out-of-fold logistic-regression fusion of signals -> calibrated reliability score.

    Writes r[out_key] = OOF P(correct). Tests whether *combining* structure (CPR) with
    model-internal signals (self-consistency, verbalized) beats any single signal.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except Exception:
        for r in rows:
            r[out_key] = sum(r[k] for k in feat_keys) / len(feat_keys)
        return
    n = len(rows)
    X = np.array([[float(r[k]) for k in feat_keys] for r in rows])
    y = np.array([1 if r["correct"] else 0 for r in rows])
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    fold_of = {idx[i]: i % folds for i in range(n)}
    preds = [0.5] * n
    if y.sum() == 0 or y.sum() == n:
        for r in rows:
            r[out_key] = 0.5
        return
    for f in range(folds):
        tr = [i for i in range(n) if fold_of[i] != f]
        te = [i for i in range(n) if fold_of[i] == f]
        if not te or len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        for j, i in enumerate(te):
            preds[i] = float(p[j])
    for i, r in enumerate(rows):
        r[out_key] = preds[i]


def _paired_bootstrap_auroc(rows, signals, ref, others, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(rows)
    deltas = {o: [] for o in others}
    abs_auroc = {s: [] for s in signals}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        corr = [rows[i]["correct"] for i in idx]
        a = {s: _auroc([rows[i][s] for i in idx], corr) for s in signals}
        if a.get(ref) != a.get(ref):
            continue
        for s in signals:
            if a[s] == a[s]:
                abs_auroc[s].append(a[s])
        for o in others:
            if a[o] == a[o]:
                deltas[o].append(a[ref] - a[o])

    def ci(vals):
        if not vals:
            return [None, None]
        s = sorted(vals)
        return [round(s[int(0.025 * len(s))], 4), round(s[min(len(s) - 1, int(0.975 * len(s)))], 4)]

    out = {"ref": ref, "auroc_ci": {s: ci(v) for s, v in abs_auroc.items()},
           f"delta_vs_{ref}": {}}
    for o in others:
        d = deltas[o]
        out[f"delta_vs_{ref}"][o] = {
            "ci95": ci(d),
            "p_ref_better": round(sum(1 for v in d if v > 0) / max(len(d), 1), 4),
        }
    return out


def _union_ledger(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    merged = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        merged.facts.extend(d.ledger.facts)
    return merged


def _self_consistency_conf(raw_pred, sc_samples, rel_tol=1e-2):
    rv = extract_final_number(raw_pred)
    svals = [extract_final_number(s) for s in (sc_samples or [])]
    svals = [v for v in svals if v is not None]
    if not svals:
        return 0.0
    if rv is None:
        # fall back to modal agreement
        best = 0
        for v in svals:
            best = max(best, sum(1 for u in svals if numbers_close(u, v, rel_tol)))
        return best / len(svals)
    return sum(1 for v in svals if numbers_close(v, rv, rel_tol)) / len(svals)


def run(ds, pred_path, retr_path, limit=None):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}
    if limit:
        preds = preds[:limit]

    rows, missing = [], 0
    for p in preds:
        rec = by_id.get(str(p.get("query_id"))) or by_q.get(p.get("query"))
        if rec is None or not (rec.get("retrieved") or rec.get("retrieved_docs")):
            missing += 1
            continue
        query = p.get("query") or p.get("question")
        gold = p.get("gold")
        raw_pred = p.get("raw_pred") or p.get("prediction") or ""
        correct = bool(p["raw_correct"]) if p.get("raw_correct") is not None else \
            bool(extract_final_number(raw_pred) is not None and gold is not None and number_match(raw_pred, gold))

        pack = build_evidence_pack(query, rec.get("retrieved") or rec.get("retrieved_docs"), top_n_facts=8)
        ledger = _union_ledger(pack)
        if ledger is None:
            missing += 1
            continue
        vr = verify(raw_pred, ledger, query, gold=gold)
        cpr = verify_cpr(raw_pred, ledger, query, gold=gold, selected_facts=pack.selected_facts)
        vo_conf = 1.0 if vr.grounded else (0.85 if vr.derivable else (0.1 if vr.pred_value is not None else 0.0))
        sc_conf = _self_consistency_conf(raw_pred, p.get("sc_samples"))
        vb = p.get("verbalized_conf")
        vb_conf = 0.5 if vb is None else float(vb)
        rows.append({
            "correct": correct,
            "value_only": vo_conf,
            "cpr": cpr.confidence,
            "self_consistency": sc_conf,
            "verbalized": vb_conf,
            "cpr_level": cpr.level,
        })

    # ---- derived combination signals (test complementarity of structure vs model-internal) ----
    def _minmax(key):
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        for r in rows:
            r[f"_n_{key}"] = (r[key] - lo) / rng
    for s in SIGNALS:
        _minmax(s)
    for r in rows:
        r["cpr+sc"] = 0.5 * (r["_n_cpr"] + r["_n_self_consistency"])
        r["cpr+verb"] = 0.5 * (r["_n_cpr"] + r["_n_verbalized"])
        r["sc+verb"] = 0.5 * (r["_n_self_consistency"] + r["_n_verbalized"])
    _cv_logistic(rows, ["cpr", "self_consistency", "verbalized", "value_only"], "fusion_all")
    _cv_logistic(rows, ["self_consistency", "verbalized"], "fusion_internal")

    eval_signals = SIGNALS + ["cpr+sc", "cpr+verb", "sc+verb", "fusion_internal", "fusion_all"]
    n = len(rows)
    base_acc = sum(r["correct"] for r in rows) / max(n, 1)
    result = {"dataset": ds, "n": n, "missing": missing, "base_accuracy": round(base_acc, 4),
              "signals": {}}
    thr = {s: 0.5 for s in eval_signals}
    for s in eval_signals:
        conf = [r[s] for r in rows]
        corr = [r["correct"] for r in rows]
        sa_auc, curve = _selective_acc_auc(conf, corr)
        result["signals"][s] = {
            "auroc": round(_auroc(conf, corr), 4),
            "selective_acc_auc": round(sa_auc, 4),
            "risk_coverage": curve,
            **_coverage_at_acc(conf, corr),
            **_sep(conf, corr, thr[s]),
        }
    # (1) CPR's classic claim: does CPR beat value-only? (2) Complementarity: does the
    # structure+model-internal fusion beat the best single signal?
    result["bootstrap_cpr"] = _paired_bootstrap_auroc(
        rows, eval_signals, ref="cpr", others=["value_only"])
    result["bootstrap_fusion"] = _paired_bootstrap_auroc(
        rows, eval_signals, ref="fusion_all", others=["verbalized", "self_consistency", "cpr", "fusion_internal"])

    # Does structure catch errors the model is OVER-confident about? Among the answers the
    # model itself rates most reliable (top-half by sc+verb), can CPR still separate
    # correct from wrong? If yes, structure is orthogonal value that model-internal misses.
    order_int = sorted(range(n), key=lambda i: rows[i]["sc+verb"], reverse=True)
    top = order_int[: max(2, n // 2)]
    top_rows = [rows[i] for i in top]
    n_conf_wrong = sum(1 for r in top_rows if not r["correct"])
    cpr_in_top = _auroc([r["cpr"] for r in top_rows], [r["correct"] for r in top_rows])
    vo_in_top = _auroc([r["value_only"] for r in top_rows], [r["correct"] for r in top_rows])
    result["overconfidence_audit"] = {
        "high_model_conf_n": len(top_rows),
        "confident_but_wrong": n_conf_wrong,
        "confident_but_wrong_rate": round(n_conf_wrong / max(len(top_rows), 1), 4),
        "cpr_auroc_within_confident": round(cpr_in_top, 4) if cpr_in_top == cpr_in_top else None,
        "value_only_auroc_within_confident": round(vo_in_top, 4) if vo_in_top == vo_in_top else None,
    }
    # CPR level distribution
    dist = {}
    for r in rows:
        dist[r["cpr_level"]] = dist.get(r["cpr_level"], 0) + 1
    result["cpr_level_dist"] = dist
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/gemini_gen/{ds}_predictions.jsonl")
    ap.add_argument("--out", default="outputs/research/strong_reliability/report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    all_res = {}
    for ds in args.datasets:
        pred_path = args.pred.format(ds=ds)
        if not Path(pred_path).exists():
            print(f"[skip] {ds}: no predictions at {pred_path}")
            continue
        res = run(ds, pred_path, RETR[ds], args.limit)
        all_res[ds] = res
        print(f"\n=== {ds}  n={res['n']}  base_acc={res['base_accuracy']} ===")
        print(f"{'signal':18s} {'AUROC':>7s} {'sel_AUC':>8s} {'cov@.6':>7s} {'cov@.7':>7s} {'sep':>7s}")
        for s in res["signals"]:
            g = res["signals"][s]
            print(f"{s:18s} {g['auroc']:7.4f} {g['selective_acc_auc']:8.4f} "
                  f"{g.get('cov@acc>=0.6',0):7.3f} {g.get('cov@acc>=0.7',0):7.3f} {g['separation']:7.4f}")
        bc = res["bootstrap_cpr"]["delta_vs_cpr"]["value_only"]
        print(f"CPR vs value_only:  ΔAUROC CI95={bc['ci95']} P(CPR better)={bc['p_ref_better']}")
        print("Fusion(all) vs single signals (ΔAUROC, P(fusion better)):")
        for o, d in res["bootstrap_fusion"]["delta_vs_fusion_all"].items():
            print(f"   vs {o:16s} CI95={d['ci95']}  P={d['p_ref_better']}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(all_res, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
