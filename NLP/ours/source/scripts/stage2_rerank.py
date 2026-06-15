#!/usr/bin/env python3
"""Stage-2 reranking experiment (honest) — completes A.

Stage-1 = the validated honest retriever: BM25 + gated period + gated cell-match.
We take its top-N candidate pool and rerank with a cross-encoder (BAAI/bge-reranker-base),
then compare ways of combining the two stages. This tests the literature's single biggest
lever for text-and-table RAG (cross-encoder reranking, +12pp R@5 in the BM25→CRAG benchmark).

Arms:
  stage1_only            BM25 + period + cellmatch (no rerank)
  ce_only                pure cross-encoder over the stage-1 top-N pool
  rrf(stage1, ce)        reciprocal-rank fusion of the two orderings
  weighted(stage1+w·ce)  stage-1 RRF score + w · normalized CE score

Usage: python scripts/stage2_rerank.py --sample 0 --pool 20
"""
from __future__ import annotations

import argparse, json, os, re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.core import RetrievalResult
from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the","a","an","of","in","for","to","and","or","was","is","are","what","how","much",
         "many","did","does","do","as","by","on","at","year","fiscal","report","reported",
         "company","value","during","between","change","total"}
DATASETS = [("FinQA", "test"), ("ConvFinQA", "turn_0"), ("TAT-DQA", "test")]


def toks(text): return {t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1}
def bm_toks(text): return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]
def jacc(a, b): return len(a & b) / len(a | b) if (a and b) else 0.0


def doc_facts(pc):
    tmd = extract_table(pc)
    if not tmd:
        return set(extract_years(pc)), []
    L = extract_ledger_from_table(tmd)
    yrs, cells = set(), []
    for f in L.facts:
        p = None
        if f.period:
            try: p = int(float(str(f.period))); yrs.add(p)
            except (ValueError, TypeError): pass
        cells.append((toks(f.concept), p))
    if not yrs: yrs = set(extract_years(tmd))
    return yrs, cells


def cell_match(qc, qy, cells, dy):
    if not qc or not cells: return 0.0
    best = 0.0
    for ct, p in cells:
        ov = jacc(qc, ct)
        if ov <= 0: continue
        pf = 1.0 if (p is not None and p in qy) else (0.6 if qy & dy else 0.3)
        best = max(best, ov * pf)
    return best


def rrf_order(ranklist, k=60):
    return {idx: 1.0 / (k + r + 1) for r, idx in enumerate(ranklist)}


def evaluate(model, ce, cfg, split, sample, pool, rrf_k=60):
    from rank_bm25 import BM25Okapi
    data = load_t2ragbench_split(cfg, split=split, sample_size=(sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c)+1:].lstrip() if c and q.startswith(c + ":") else q)

    bm25 = BM25Okapi([bm_toks(d.page_content) for d in corpus])
    df = [doc_facts(d.page_content) for d in corpus]
    doc_years = [a for a, _ in df]; doc_cells = [b for _, b in df]
    q_years = [set(extract_years(q)) for q in raw_q]; q_content = [toks(q) for q in raw_q]

    # ---- stage-1 fused pool ----
    stage1_pool, stage1_score = [], []
    for i in range(len(raw_q)):
        scores = bm25.get_scores(bm_toks(raw_q[i]))
        cpool = list(np.argsort(-scores)[:50])
        fused = {int(idx): 1.0 / (rrf_k + r + 1) for r, idx in enumerate(cpool)}
        keys = list(fused.keys()); qy, qct = q_years[i], q_content[i]
        if qy:
            ms = [idx for idx in keys if doc_years[idx] & qy]
            if ms and len(ms)/len(keys) <= 0.4:
                for idx in ms: fused[idx] += 0.05/rrf_k
        if qct:
            cm = {idx: cell_match(qct, qy, doc_cells[idx], doc_years[idx]) for idx in keys}
            hi = [idx for idx, c in cm.items() if c >= 0.34]
            if hi and len(hi)/len(keys) <= 0.4:
                for idx in hi: fused[idx] += 0.3*cm[idx]/rrf_k
        order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:pool]
        stage1_pool.append([i2 for i2, _ in order])
        stage1_score.append({i2: s for i2, s in order})

    # ---- cross-encoder scores over the pool ----
    pairs, owner = [], []
    for i, pl in enumerate(stage1_pool):
        for idx in pl:
            pairs.append((raw_q[i], corpus[idx].page_content[:2000])); owner.append((i, idx))
    ce_scores_flat = ce.predict(pairs, batch_size=128, show_progress_bar=False)
    ce_by_q = [dict() for _ in range(len(raw_q))]
    for (i, idx), s in zip(owner, ce_scores_flat):
        ce_by_q[i][idx] = float(s)

    def mk(order_fn):
        res = []
        for i in range(len(raw_q)):
            order = order_fn(i)[:5]
            res.append(RetrievalResult(query=raw_q[i], retrieved_docs=[corpus[idx] for idx in order],
                                       ground_truth_id=gts[i], meta_data=metas[i]))
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    def o_stage1(i): return stage1_pool[i]
    def o_ce(i): return sorted(stage1_pool[i], key=lambda idx: ce_by_q[i][idx], reverse=True)
    def o_rrf(i):
        s1 = rrf_order(stage1_pool[i])
        ce_ord = sorted(stage1_pool[i], key=lambda idx: ce_by_q[i][idx], reverse=True)
        s2 = rrf_order(ce_ord)
        comb = {idx: s1.get(idx, 0) + s2.get(idx, 0) for idx in stage1_pool[i]}
        return sorted(stage1_pool[i], key=lambda idx: comb[idx], reverse=True)
    def o_weighted(i, w=1.0):
        ces = np.array([ce_by_q[i][idx] for idx in stage1_pool[i]], float)
        if len(ces) > 1 and np.ptp(ces) > 0:
            cen = (ces - ces.min()) / np.ptp(ces)
        else:
            cen = np.zeros(len(ces))
        comb = {idx: stage1_score[i].get(idx, 0) + w * cen[j] / 60.0 for j, idx in enumerate(stage1_pool[i])}
        return sorted(stage1_pool[i], key=lambda idx: comb[idx], reverse=True)

    return len(raw_q), {
        "stage1_only": mk(o_stage1),
        "ce_only": mk(o_ce),
        "rrf(stage1,ce)": mk(o_rrf),
        "weighted(stage1+1.0ce)": mk(lambda i: o_weighted(i, 1.0)),
        "weighted(stage1+2.0ce)": mk(lambda i: o_weighted(i, 2.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--out", default="outputs/stage2_rerank")
    args = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.embed_model, device=dev)  # kept for parity; stage-1 uses BM25 only
    ce = CrossEncoder(args.reranker, max_length=512, device=dev)

    per_ds, counts, names = {}, {}, None
    for cfg, split in DATASETS:
        print(f"\n### {cfg} ({split}) — rerank pool={args.pool} ###")
        n, res = evaluate(model, ce, cfg, split, args.sample, args.pool)
        per_ds[cfg], counts[cfg] = res, n; names = list(res.keys())
        print(f"{'arm':<26}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
        for nm, mm in res.items():
            print(f"{nm:<26}{mm['MRR@3']:>8}{mm['Recall@1']:>8}{mm['Recall@3']:>8}{mm['Recall@5']:>8}{mm['NDCG@3']:>8}")
    total = sum(counts.values())
    wavg = {nm: round(sum(per_ds[c][nm]["MRR@3"] * counts[c] for c in counts) / total, 4) for nm in names}
    best = max(wavg.items(), key=lambda x: x[1])
    print(f"\n### weighted-avg MRR@3 (n={total}) ###")
    for nm, v in sorted(wavg.items(), key=lambda x: -x[1]): print(f"  {nm:<26}{v:>8}")
    print(f"\nBEST: {best[0]} (w.avg MRR@3={best[1]})")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "stage2_rerank.json").write_text(json.dumps(
        {"counts": counts, "per_dataset": per_ds, "weighted_avg_mrr3": wavg, "best": best[0],
         "reranker": args.reranker, "pool": args.pool}, indent=2))
    print(f"\nSaved to {args.out}/stage2_rerank.json")


if __name__ == "__main__":
    main()
