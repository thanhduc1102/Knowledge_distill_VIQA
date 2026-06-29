#!/usr/bin/env python3
"""Iterative CPR optimization on strong-generator outputs.

The baseline study found that, for a strong generator, model-internal signals
(self-consistency, verbalized) dominate hand-weighted CPR, and CPR's marginal value
appears only on ConvFinQA. This script tries to *close that gap* by:

  1. COMPONENT SWEEP: which CPR component set maximises AUROC per dataset
     (full / -period / -3op / role-only / concept+role / ...).
  2. LEARNED CPR (CPR-cal): a 5-fold CV logistic over CPR's OWN structural features
     (confidence, concept/period consistency, level, grounded/derivable flags), i.e.
     replacing CPR's hand-designed max() weighting with learned weights — the upgrade
     the design docs said was needed.
  3. COMPLEMENTARITY RE-TEST: does [CPR-cal, self_consistency, verbalized] beat
     [self_consistency, verbalized] (model-internal only)? Paired bootstrap.

No GPU/API. Reuses the same ledgers as the deployed pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, numbers_close
from gsr_cacl.research.cpr_verifier import verify_cpr, LEVEL_CONFIDENCE

RETR = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}

COMPONENT_SETS = {
    "full": frozenset({"concept", "period", "role", "3op"}),
    "no_period": frozenset({"concept", "role", "3op"}),
    "no_3op": frozenset({"concept", "period", "role"}),
    "concept_role": frozenset({"concept", "role"}),
    "role_only": frozenset({"role"}),
    "concept_period": frozenset({"concept", "period"}),
}


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
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _union_ledger(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    merged = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        merged.facts.extend(d.ledger.facts)
    return merged


def _sc_conf(raw, samples, tol=1e-2):
    rv = extract_final_number(raw)
    sv = [extract_final_number(s) for s in (samples or [])]
    sv = [v for v in sv if v is not None]
    if not sv:
        return 0.0
    if rv is None:
        return max((sum(1 for u in sv if numbers_close(u, v, tol)) for v in sv), default=0) / len(sv)
    return sum(1 for v in sv if numbers_close(v, rv, tol)) / len(sv)


def _cpr_features(cpr):
    return [
        cpr.confidence,
        cpr.concept_consistency,
        cpr.period_consistency,
        1.0 if cpr.grounded else 0.0,
        1.0 if cpr.derivable else 0.0,
        1.0 if cpr.value_only_grounded else 0.0,
        1.0 if cpr.value_only_derivable else 0.0,
        LEVEL_CONFIDENCE.get(cpr.level, 0.0),
    ]


def _cv_logistic(X, y, folds=5, seed=0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    n = len(y)
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    fold = {idx[i]: i % folds for i in range(n)}
    preds = [0.5] * n
    X = np.array(X); y = np.array(y)
    if y.sum() in (0, n):
        return preds
    for f in range(folds):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        if not te or len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=600, C=1.0).fit(X[tr], y[tr])
        for j, i in enumerate(te):
            preds[i] = float(clf.predict_proba([X[i]])[0, 1])
    return preds


def _paired_delta(rows, a_key, b_key, n_boot=2000, seed=0):
    """AUROC(a) - AUROC(b) paired bootstrap."""
    rng = random.Random(seed); n = len(rows); deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        corr = [rows[i]["correct"] for i in idx]
        aa = _auroc([rows[i][a_key] for i in idx], corr)
        bb = _auroc([rows[i][b_key] for i in idx], corr)
        if aa == aa and bb == bb:
            deltas.append(aa - bb)
    s = sorted(deltas)
    return {"delta_auroc": round(_auroc([r[a_key] for r in rows], [r["correct"] for r in rows]) -
                                _auroc([r[b_key] for r in rows], [r["correct"] for r in rows]), 4),
            "ci95": [round(s[int(0.025 * len(s))], 4), round(s[min(len(s) - 1, int(0.975 * len(s)))], 4)],
            "p_a_better": round(sum(1 for v in deltas if v > 0) / max(len(deltas), 1), 4)}


def run(ds, pred_path, retr_path):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    rows = []
    for p in preds:
        rec = by_id.get(str(p.get("query_id")))
        if rec is None or not rec.get("retrieved"):
            continue
        query = p.get("query") or p.get("question")
        gold = p.get("gold")
        raw = p.get("raw_pred") or ""
        correct = bool(p["raw_correct"]) if p.get("raw_correct") is not None else \
            bool(extract_final_number(raw) is not None and gold is not None and number_match(raw, gold))
        pack = build_evidence_pack(query, rec["retrieved"], top_n_facts=8)
        ledger = _union_ledger(pack)
        if ledger is None:
            continue
        row = {"correct": correct,
               "sc": _sc_conf(raw, p.get("sc_samples")),
               "verb": 0.5 if p.get("verbalized_conf") is None else float(p["verbalized_conf"])}
        # component sweep
        for name, comps in COMPONENT_SETS.items():
            cpr = verify_cpr(raw, ledger, query, gold=gold,
                             selected_facts=pack.selected_facts, components=comps)
            row[f"cpr_{name}"] = cpr.confidence
            if name == "full":
                row["_feat"] = _cpr_features(cpr)
        rows.append(row)

    y = [1 if r["correct"] else 0 for r in rows]
    # learned CPR (calibrated over its own features)
    cpr_cal = _cv_logistic([r["_feat"] for r in rows], y)
    for i, r in enumerate(rows):
        r["cpr_cal"] = cpr_cal[i]

    # min-max for fusion
    def mm(key):
        vs = [r[key] for r in rows]; lo, hi = min(vs), max(vs); rng = (hi - lo) or 1.0
        for r in rows:
            r["_n_" + key] = (r[key] - lo) / rng
    for k in ("cpr_cal", "cpr_full", "sc", "verb"):
        mm(k)
    # fusions
    internal = _cv_logistic([[r["sc"], r["verb"]] for r in rows], y)
    full_cal = _cv_logistic([[r["cpr_cal"], r["sc"], r["verb"]] for r in rows], y)
    full_raw = _cv_logistic([[r["cpr_full"], r["sc"], r["verb"]] for r in rows], y)
    for i, r in enumerate(rows):
        r["fusion_internal"] = internal[i]
        r["fusion_cprcal"] = full_cal[i]
        r["fusion_cprraw"] = full_raw[i]

    res = {"dataset": ds, "n": len(rows), "base_acc": round(sum(y) / len(y), 4), "auroc": {}}
    for key in (list(f"cpr_{n}" for n in COMPONENT_SETS) +
                ["cpr_cal", "sc", "verb", "fusion_internal", "fusion_cprcal", "fusion_cprraw"]):
        res["auroc"][key] = round(_auroc([r[key] for r in rows], y), 4)
    res["best_component_set"] = max(COMPONENT_SETS, key=lambda n: res["auroc"][f"cpr_{n}"])
    res["complementarity"] = {
        "cprcal_vs_rawcpr": _paired_delta(rows, "cpr_cal", "cpr_full"),
        "fusion_cprcal_vs_internal": _paired_delta(rows, "fusion_cprcal", "fusion_internal"),
        "fusion_cprcal_vs_cprraw": _paired_delta(rows, "fusion_cprcal", "fusion_cprraw"),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/gemini_gen/{ds}_predictions.jsonl")
    ap.add_argument("--out", default="outputs/research/cpr_optimize/report.json")
    args = ap.parse_args()
    allres = {}
    for ds in args.datasets:
        pp = args.pred.format(ds=ds)
        if not Path(pp).exists():
            print(f"[skip] {ds}"); continue
        r = run(ds, pp, RETR[ds]); allres[ds] = r
        print(f"\n=== {ds} n={r['n']} base_acc={r['base_acc']} best_comp={r['best_component_set']} ===")
        a = r["auroc"]
        print(f"  cpr_full={a['cpr_full']}  cpr_cal={a['cpr_cal']}  (sc={a['sc']} verb={a['verb']})")
        print(f"  fusion_internal={a['fusion_internal']}  fusion_cprcal={a['fusion_cprcal']}  fusion_cprraw={a['fusion_cprraw']}")
        c = r["complementarity"]
        print(f"  CPR-cal vs raw-CPR:        Δ={c['cprcal_vs_rawcpr']['delta_auroc']} CI={c['cprcal_vs_rawcpr']['ci95']} P={c['cprcal_vs_rawcpr']['p_a_better']}")
        print(f"  fusion(CPRcal) vs internal: Δ={c['fusion_cprcal_vs_internal']['delta_auroc']} CI={c['fusion_cprcal_vs_internal']['ci95']} P={c['fusion_cprcal_vs_internal']['p_a_better']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allres, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
