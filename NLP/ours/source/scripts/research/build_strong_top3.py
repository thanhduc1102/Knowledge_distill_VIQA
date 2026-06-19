#!/usr/bin/env python3
"""Regenerate retrieval top-3 with the STRONG retriever (company-pool + pool-local IDF).

The saved `outputs/final_retrieval/*/retrieval_top3.jsonl` come from the weaker FULL+C3
arm (gold-in-top3 ≈ 87/93/62%), which confounds the generation diagnosis with retrieval
misses. This rebuilds top-3 from the verified strong regime — company-complete pool +
loclex(+concept+cell+meta) training-free fusion — and writes records in the format the
KG bridge / eval scripts consume (table + meta + gold + question).

Usage:  PYTHONPATH=src python scripts/research/build_strong_top3.py --dataset FinQA --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np
from datasets import load_dataset

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.experts.base import minmax
from gsr_cacl.experts.local_lexical import LocalLexicalExpert
from gsr_cacl.experts.concept import ConceptExpert
from gsr_cacl.experts.cell import CellExpert
from gsr_cacl.experts.meta_retriever import MetadataRetriever

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=list(SPLITS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="outputs/strong_retrieval")
    args = ap.parse_args()
    split = SPLITS[args.dataset]

    data = load_t2ragbench_split(args.dataset, split=split, sample_size=(args.sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    # gold answers + table/context maps from the raw dataset
    qa = load_dataset("G4KMU/t2-ragbench", args.dataset, split=split).to_pandas()
    if args.sample:
        qa = qa.sample(n=args.sample, random_state=42).reset_index(drop=True)
    def _coerce(r):
        try:
            return float(r)
        except (ValueError, TypeError):
            return r
    gold_ans = [_coerce(r) for r in qa["program_answer"].tolist()]
    orig_ans = qa["original_answer"].tolist()
    cid2table, cid2ctx = {}, {}
    for cfg_split in set(["train", "dev", "test", split]):
        try:
            d = load_dataset("G4KMU/t2-ragbench", args.dataset, split=cfg_split).to_pandas()
        except Exception:
            continue
        for _, r in d.iterrows():
            cid = str(r.get("context_id", ""))
            if cid and cid not in cid2table:
                cid2table[cid] = str(r.get("table", "") or "")
                cid2ctx[cid] = str(r.get("context", "") or "")

    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]

    # experts: meta seeds the company pool; loclex/concept/cell re-score within it
    meta = MetadataRetriever(company_pool=True, max_add=200)
    loc = LocalLexicalExpert(abbr_expand=True)
    con = ConceptExpert()
    cel = CellExpert()
    for ex in (meta, loc, con, cel):
        ex.prepare(corpus, doc_metas); ex.set_queries(raw_q, metas)

    out_path = Path(args.out, args.dataset.lower()); out_path.mkdir(parents=True, exist_ok=True)
    fout = open(out_path / "retrieval_top3.jsonl", "w")
    hit3 = 0; n = 0
    for qi in range(len(raw_q)):
        pool = meta.get_candidates(qi, 200)
        if not pool:
            continue
        n += 1
        s = (minmax(loc.score_pool(qi, pool)) + minmax(con.score_pool(qi, pool))
             + minmax(cel.score_pool(qi, pool)))
        order = [pool[j] for j in np.argsort(-s)[:3]]
        gi = gold_idx[qi]
        hit3 += int(gi in order)
        retrieved = []
        for rank, di in enumerate(order, 1):
            cid = str(corpus[di].id)
            retrieved.append({
                "rank": rank, "id": cid, "context_id": cid,
                "meta": doc_metas[di],
                "table": cid2table.get(cid, ""),
                "page_content": cid2ctx.get(cid) or corpus[di].page_content,
                "score": float(s[pool.index(di)]),
            })
        fout.write(json.dumps({
            "query": raw_q[qi], "raw_question": raw_q[qi],
            "ground_truth_id": str(gts[qi]),
            "gold": [gold_ans[qi], str(orig_ans[qi])],
            "retrieved": retrieved,
        }) + "\n")
    fout.close()
    print(f"{args.dataset}: wrote {n} records | gold-in-top3 (R@3) = {hit3/max(n,1):.4f} → {out_path}/retrieval_top3.jsonl")


if __name__ == "__main__":
    main()
