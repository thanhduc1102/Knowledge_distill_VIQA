#!/usr/bin/env python3
"""Retrieval -> Number-Match linkage: does better retrieval drive end-to-end NM?

Joins the strong-generator predictions (gemini_gen) with the retrieval records, computes the
gold document's rank in each query's retrieved list, and stratifies Number-Match accuracy by
that rank. If NM is much higher when the gold doc is rank-1 than when it is absent, retrieval
quality is causal for the end output — the central claim that retrieval is the lever.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.ledger.numeric import number_match

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}


def gold_rank(rec):
    gid = str(rec.get("ground_truth_id") or "")
    for i, d in enumerate(rec.get("retrieved", []), 1):
        if str(d.get("context_id") or d.get("id")) == gid:
            return i
    return -1


def run(ds):
    retr = {str(r.get("query_id")): r for r in (json.loads(l) for l in open(RETR[ds]) if l.strip())}
    preds = [json.loads(l) for l in open(f"outputs/research/gemini_gen/{ds}_predictions.jsonl") if l.strip()]
    buckets = {"rank1": [], "rank2_3": [], "absent": []}
    for p in preds:
        rec = retr.get(str(p.get("query_id")))
        if not rec:
            continue
        r = gold_rank(rec)
        correct = bool(p["raw_correct"]) if p.get("raw_correct") is not None else bool(number_match(p.get("raw_pred", ""), p.get("gold")))
        key = "rank1" if r == 1 else ("rank2_3" if 2 <= r <= 3 else "absent")
        buckets[key].append(correct)
    def acc(b):
        return (round(sum(b) / len(b), 4), len(b)) if b else (None, 0)
    out = {"dataset": ds, "n": len(preds)}
    for k, b in buckets.items():
        a, nn = acc(b)
        out[k] = {"nm_accuracy": a, "n": nn}
    # overall + lift
    r1 = out["rank1"]["nm_accuracy"]; ab = out["absent"]["nm_accuracy"]
    out["nm_lift_rank1_vs_absent"] = round((r1 - ab), 4) if (r1 is not None and ab is not None) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--out", default="outputs/research/retrieval_nm_linkage/report.json")
    args = ap.parse_args()
    allr = {}
    for ds in args.datasets:
        r = run(ds); allr[ds] = r
        print(f"{ds}: NM | gold@rank1={r['rank1']['nm_accuracy']}(n={r['rank1']['n']}) "
              f"rank2-3={r['rank2_3']['nm_accuracy']}(n={r['rank2_3']['n']}) "
              f"absent={r['absent']['nm_accuracy']}(n={r['absent']['n']}) "
              f"| lift={r['nm_lift_rank1_vs_absent']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
