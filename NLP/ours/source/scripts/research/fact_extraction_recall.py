#!/usr/bin/env python3
"""Fact-extraction recall: can the auto-extracted ledger even *contain* the answer?

CPR (and any ledger verifier) is upper-bounded by extraction: if the gold answer value is not in
the ledger (and not derivable from it), no grounding check can certify it. This reports, per
dataset and per task type, the fraction of queries whose GOLD answer is:
  * grounded  : present as a ledger cell (numbers_close),
  * derivable : a simple op over two ledger cells,
  * certifiable = grounded OR derivable  (the auditable ceiling).

Runs on the retrieval top-k (gold + tables) — independent of any generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import is_grounded, is_derivable
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import extract_numbers
from gsr_cacl.ledger.select import infer_task_type
from gsr_cacl.research.derivation import derivation_depth


def _union(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    m = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _gold_value(gold):
    if gold is None:
        return None
    if isinstance(gold, list):
        for g in gold:
            ns = extract_numbers(str(g))
            if ns:
                return ns[0]
        return None
    ns = extract_numbers(str(gold))
    return ns[0] if ns else None


def run(ds, retr_path, limit, gold_doc_only, multi_op):
    retr = [json.loads(l) for l in open(retr_path) if l.strip()]
    if limit:
        retr = retr[:limit]
    keys = ["grounded", "derivable", "certifiable", "cert3op"]
    by_task = defaultdict(lambda: {"n": 0, **{k: 0 for k in keys}})
    tot = {"n": 0, **{k: 0 for k in keys}}
    for rec in retr:
        gv = _gold_value(rec.get("gold"))
        if gv is None:
            continue
        retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
        if gold_doc_only:
            gid = str(rec.get("ground_truth_id") or "")
            retrieved = [d for d in retrieved if str(d.get("context_id") or d.get("id")) == gid] or retrieved[:1]
        if not retrieved:
            continue
        pack = build_evidence_pack(rec.get("query", ""), retrieved, top_n_facts=8)
        ledger = _union(pack)
        if ledger is None or not ledger.numeric_facts():
            tot["n"] += 1
            continue
        g = is_grounded(gv, ledger) is not None
        d = (not g) and (is_derivable(gv, ledger) is not None)
        cert3 = False
        if multi_op and not (g or d):
            vals = [f.value for f in ledger.numeric_facts() if f.value is not None]
            cert3 = derivation_depth(gv, vals, max_ops=3) == "3op"
        task = infer_task_type(rec.get("query", ""))
        for bucket in (tot, by_task[task]):
            bucket["n"] += 1
            bucket["grounded"] += int(g)
            bucket["derivable"] += int(d)
            bucket["certifiable"] += int(g or d)
            bucket["cert3op"] += int(g or d or cert3)

    def rates(b):
        n = max(b["n"], 1)
        return {"n": b["n"], "grounded": round(b["grounded"]/n, 4),
                "derivable": round(b["derivable"]/n, 4), "certifiable": round(b["certifiable"]/n, 4),
                "certifiable_3op": round(b["cert3op"]/n, 4)}

    return {"dataset": ds, "gold_doc_only": gold_doc_only, "multi_op": multi_op, "overall": rates(tot),
            "by_task": {k: rates(v) for k, v in sorted(by_task.items(), key=lambda kv: -kv[1]["n"])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gold-doc-only", action="store_true",
                    help="restrict ledger to the gold document (isolates extraction from retrieval)")
    ap.add_argument("--multi-op", action="store_true", help="also report 3-operand certifiable ceiling")
    ap.add_argument("--out", default="outputs/research/fact_extraction_recall")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    tag = "gold-doc" if args.gold_doc_only else "top-k"
    print(f"Extraction recall ({tag}): grounded / derivable / certifiable(2op) / certifiable(3op)")
    for ds in args.datasets:
        r = run(ds, args.retr.format(ds=ds), args.limit, args.gold_doc_only, args.multi_op)
        allres[ds] = r
        o = r["overall"]
        print(f"\n{ds} (n={o['n']}): grounded={o['grounded']} derivable={o['derivable']} "
              f"certifiable_2op={o['certifiable']} certifiable_3op={o['certifiable_3op']}")
        for t, v in list(r["by_task"].items())[:5]:
            print(f"    {t:14s} n={v['n']:4d}  2op={v['certifiable']} 3op={v['certifiable_3op']}")
    (out / ("summary_golddoc.json" if args.gold_doc_only else "summary.json")).write_text(json.dumps(allres, indent=2))
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
