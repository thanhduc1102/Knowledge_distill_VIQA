#!/usr/bin/env python3
"""Why does cpr+verbalized beat each alone? Error-set disjointness analysis.

For each reliability signal we define "flagged" = confidence below a threshold tuned
so the signal abstains on the same fraction (operating point) across signals. Among the
WRONG answers (the ones we should abstain on), we measure how much CPR and verbalized
catch the SAME vs DIFFERENT errors. CPR-only catches are confident-but-ungrounded
hallucinations a strong model rates as reliable -> the orthogonal value of structure.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, numbers_close
from gsr_cacl.research.cpr_verifier import verify_cpr

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}


def _union(pack):
    if not pack.ranked:
        return None
    m = FactLedger(doc_id="u", facts=list(pack.ranked[0].ledger.facts), meta=dict(pack.ranked[0].ledger.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _sc(raw, samples, tol=1e-2):
    rv = extract_final_number(raw); sv = [extract_final_number(s) for s in (samples or [])]
    sv = [v for v in sv if v is not None]
    if not sv:
        return 0.0
    if rv is None:
        return 0.0
    return sum(1 for v in sv if numbers_close(v, rv, tol)) / len(sv)


def run(ds, pred, retr, abstain_frac=0.4):
    preds = [json.loads(l) for l in open(pred) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in (json.loads(l) for l in open(retr) if l.strip())}
    rows = []
    for p in preds:
        rec = by_id.get(str(p.get("query_id")))
        if not rec or not rec.get("retrieved"):
            continue
        q = p.get("query"); gold = p.get("gold"); raw = p.get("raw_pred") or ""
        correct = bool(p["raw_correct"]) if p.get("raw_correct") is not None else bool(number_match(raw, gold))
        pack = build_evidence_pack(q, rec["retrieved"], top_n_facts=8); led = _union(pack)
        if led is None:
            continue
        cpr = verify_cpr(raw, led, q, gold=gold, selected_facts=pack.selected_facts)
        rows.append({"correct": correct, "cpr": cpr.confidence,
                     "verb": 0.5 if p.get("verbalized_conf") is None else float(p["verbalized_conf"]),
                     "sc": _sc(raw, p.get("sc_samples"))})
    n = len(rows)
    wrong = [i for i in range(n) if not rows[i]["correct"]]

    def flagged(key):
        # abstain on the lowest-confidence abstain_frac of ALL answers
        order = sorted(range(n), key=lambda i: rows[i][key])
        k = int(round(abstain_frac * n))
        return set(order[:k])

    fc, fv, fs = flagged("cpr"), flagged("verb"), flagged("sc")
    W = set(wrong)
    cc, cv, cs = fc & W, fv & W, fs & W
    return {"dataset": ds, "n": n, "n_wrong": len(wrong), "abstain_frac": abstain_frac,
            "caught_by_cpr": len(cc), "caught_by_verb": len(cv), "caught_by_sc": len(cs),
            "cpr_and_verb": len(cc & cv), "cpr_or_verb": len(cc | cv),
            "cpr_only_not_verb": len(cc - cv), "verb_only_not_cpr": len(cv - cc),
            "union_recall": round(len(cc | cv) / max(len(wrong), 1), 4),
            "verb_recall": round(len(cv) / max(len(wrong), 1), 4),
            "cpr_marginal_recall": round(len(cc - cv) / max(len(wrong), 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/gemini_gen/{ds}_predictions.jsonl")
    ap.add_argument("--abstain-frac", type=float, default=0.4)
    ap.add_argument("--out", default="outputs/research/error_disjointness/report.json")
    args = ap.parse_args()
    allr = {}
    for ds in args.datasets:
        pp = args.pred.format(ds=ds)
        if not Path(pp).exists():
            continue
        r = run(ds, pp, RETR[ds], args.abstain_frac); allr[ds] = r
        print(f"{ds}: wrong={r['n_wrong']} | verb catches {r['caught_by_verb']} "
              f"({r['verb_recall']}) | cpr catches {r['caught_by_cpr']} | "
              f"cpr-only (verb misses) {r['cpr_only_not_verb']} (+{r['cpr_marginal_recall']} recall) | "
              f"union {r['cpr_or_verb']} ({r['union_recall']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
