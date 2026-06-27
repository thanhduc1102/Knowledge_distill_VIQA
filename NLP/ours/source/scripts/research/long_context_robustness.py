"""Long-context / evidence-dilution robustness (controlled DocFinQA substitute).

DocFinQA's HF loader is currently broken, so we test the *same* failure mode it targets —
relevant evidence buried in a long, noisy context — in a controlled way: inject `m` distractor
facts (sampled from OTHER documents) into each query's ledger and measure how the reliability
AUROC of value-only grounding vs CPR degrades as `m` grows.

Hypothesis (TCEP): value-only grounding over-fires more as the ledger grows (more coincidental
value matches), while CPR's concept/period typing filters distractors, so CPR should degrade
*more gracefully* — i.e. structure-grounded reliability is robust to context dilution.
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
from gsr_cacl.ledger.fact import Fact, FactLedger
from gsr_cacl.ledger.numeric import number_match
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


def _legacy_conf(vr):
    if vr is None or vr.pred_value is None:
        return 0.0
    return 1.0 if vr.grounded else (0.85 if vr.derivable else 0.1)


def _union(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    m = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def run(ds, pred_path, retr_path, ms, limit, seed):
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in retr}
    by_q = {r.get("query"): r for r in retr}
    if limit:
        preds = preds[:limit]
    rng = random.Random(seed)

    items = []  # (query, gold, pred_text, correct, base_ledger)
    distractor_pool: list[Fact] = []
    for p in preds:
        rec = by_id.get(str(p.get("query_id"))) or by_q.get(p.get("query"))
        if not rec:
            continue
        retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
        if not retrieved:
            continue
        pred_text = p.get("raw_prediction") or p.get("prediction") or ""
        if extract_final_number(pred_text) is None:
            continue
        pack = build_evidence_pack(p["query"], retrieved, top_n_facts=8)
        ledger = _union(pack)
        if ledger is None or not ledger.numeric_facts():
            continue
        correct = bool(p.get("raw_correct") if "raw_correct" in p else number_match(extract_final_number(pred_text), p.get("gold")))
        items.append((p["query"], p.get("gold"), pred_text, correct, ledger, pack.selected_facts))
        distractor_pool.extend(ledger.numeric_facts())

    results = {}
    for m in ms:
        leg_conf, cpr_conf, labels = [], [], []
        for query, gold, pred_text, correct, ledger, sel in items:
            if m > 0 and distractor_pool:
                extra = [rng.choice(distractor_pool) for _ in range(m)]
                diluted = FactLedger(doc_id="dil", facts=list(ledger.facts) + extra, meta=ledger.meta)
            else:
                diluted = ledger
            vr = verify(pred_text, diluted, query, gold=gold)
            cpr = verify_cpr(pred_text, diluted, query, gold=gold, selected_facts=sel)
            leg_conf.append(_legacy_conf(vr)); cpr_conf.append(cpr.confidence); labels.append(correct)
        results[f"m={m}"] = {
            "legacy_auroc": round(_auroc(leg_conf, labels), 4),
            "cpr_auroc": round(_auroc(cpr_conf, labels), 4),
        }
    return {"dataset": ds, "n": len(items), "by_distractors": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/generation_system_q35_s400/{ds}_predictions.jsonl")
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--distractors", nargs="+", type=int, default=[0, 25, 75, 150])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/long_context_robustness")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for ds in args.datasets:
        r = run(ds, args.pred.format(ds=ds), args.retr.format(ds=ds), args.distractors, args.limit, args.seed)
        allres[ds] = r
        print(f"\n=== {ds} (n={r['n']}) — AUROC vs injected distractor facts ===")
        print(f"    {'distractors':>12s} " + " ".join(f"{k:>9s}" for k in r['by_distractors']))
        print(f"    {'legacy':>12s} " + " ".join(f"{v['legacy_auroc']:>9.4f}" for v in r['by_distractors'].values()))
        print(f"    {'CPR':>12s} " + " ".join(f"{v['cpr_auroc']:>9.4f}" for v in r['by_distractors'].values()))
    (out / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
