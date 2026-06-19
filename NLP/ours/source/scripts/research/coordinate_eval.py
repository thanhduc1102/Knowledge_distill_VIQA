#!/usr/bin/env python3
"""Isolated grounding test: 2D coordinate grounding vs current heuristic on the GOLD doc.

To separate cell-grounding from doc-selection and retrieval, we run both answerers on the
ledger of the GOLD document (when it is in the retrieved top-3) and compare exact Number-Match
by task. This directly measures whether Tabular Coordinate Grounding closes the lookup/sum gap.

Usage:  PYTHONPATH=src python scripts/research/coordinate_eval.py --dataset finqa
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.coordinate import coordinate_answer
from gsr_cacl.ledger.select import infer_task_type, calculation_plan, select_facts
from gsr_cacl.ledger.numeric import number_match, numbers_close

DEFAULTS = {
    "finqa": "outputs/strong_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/strong_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/strong_retrieval/tat-dqa/retrieval_top3.jsonl",
}
COORD_TASKS = {"lookup", "difference", "percent_change"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="finqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default="outputs/research/coordinate_eval")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.input or DEFAULTS[args.dataset]) if l.strip()]

    # per-task tallies: n, current-correct, coord-correct (on gold doc)
    tally = defaultdict(lambda: {"n": 0, "cur": 0, "coord": 0, "either": 0})
    agree = {"n_total": 0, "n_agree": 0, "agree_nm": 0}   # multi-path agreement ensemble
    for rec in recs:
        gt = str(rec.get("ground_truth_id") or "")
        gold = rec.get("gold")
        gdoc = next((d for d in (rec.get("retrieved") or []) if str(d.get("context_id") or d.get("id")) == gt), None)
        if gdoc is None:
            continue  # gold not retrieved → grounding not testable
        ledger = extract_ledger_from_table(gdoc.get("table") or gdoc.get("page_content", ""),
                                           doc_id=gt, meta=gdoc.get("meta") or {})
        if not ledger.numeric_facts():
            continue
        q = rec["query"]
        task = infer_task_type(q)
        if task not in COORD_TASKS:
            continue
        t = tally[task]
        t["n"] += 1

        cur = calculation_plan(q, select_facts(q, [ledger], top_n=6))
        cur_ok = cur.get("answer") is not None and number_match(cur["answer"], gold)
        t["cur"] += int(cur_ok)

        co = coordinate_answer(q, ledger, task)
        co_ok = co is not None and number_match(co["answer"], gold)
        t["coord"] += int(co_ok)
        t["either"] += int(cur_ok or co_ok)

        # multi-path agreement: both methods produce an answer AND they agree on the value
        agree["n_total"] += 1
        if cur.get("answer") is not None and co is not None and \
           numbers_close(float(cur["answer"]), float(co["answer"])):
            agree["n_agree"] += 1
            agree["agree_nm"] += int(cur_ok)   # cur_ok == co_ok here (same value)

    print(f"\n=== Coordinate grounding vs current (on GOLD doc) — {args.dataset} ===")
    print(f"{'task':<16}{'n':>6}{'current':>10}{'coord-2D':>10}{'either':>9}")
    tot = {"n": 0, "cur": 0, "coord": 0, "either": 0}
    out = {}
    for task in ("lookup", "difference", "percent_change"):
        t = tally.get(task)
        if not t or not t["n"]:
            continue
        n = t["n"]
        print(f"{task:<16}{n:>6}{t['cur']/n:>10.3f}{t['coord']/n:>10.3f}{t['either']/n:>9.3f}")
        out[task] = {"n": n, "current": round(t["cur"]/n, 4), "coord": round(t["coord"]/n, 4),
                     "either": round(t["either"]/n, 4)}
        for k in tot:
            tot[k] += t[k]
    if tot["n"]:
        n = tot["n"]
        print(f"{'-'*51}")
        print(f"{'ALL (coord tasks)':<16}{n:>6}{tot['cur']/n:>10.3f}{tot['coord']/n:>10.3f}{tot['either']/n:>9.3f}")
        out["all"] = {"n": n, "current": round(tot["cur"]/n, 4), "coord": round(tot["coord"]/n, 4),
                      "either": round(tot["either"]/n, 4)}
    if agree["n_total"]:
        cov = agree["n_agree"] / agree["n_total"]
        nm = agree["agree_nm"] / max(agree["n_agree"], 1)
        print(f"\nMULTI-PATH AGREEMENT (current ≡ coord): coverage={cov:.3f}  "
              f"NM_when_agree={nm:.3f}  (n_agree={agree['n_agree']}/{agree['n_total']})")
        out["agreement"] = {"coverage": round(cov, 4), "nm_when_agree": round(nm, 4),
                            "n_agree": agree["n_agree"], "n_total": agree["n_total"]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset+'.json')}")


if __name__ == "__main__":
    main()
