#!/usr/bin/env python3
"""Optimized Hybrid BM25 + period retrieval on FinQA (honest: no gold metadata injected).

Motivation (measured in scripts/header_retrieval.py):
  full-text BM25 = 0.665 MRR@3 (strongest, honest)  >>  dense e5 = 0.394
  naive RRF(dense, bm25) = 0.534  HURT bm25 because dense drags it down.

So we (1) use WEIGHTED multi-channel RRF that down-weights the weak dense channel, and
(2) add a PERIOD channel — the honest disambiguator headers could not provide: boost
documents whose table report-periods contain the year mentioned in the QUESTION.

Channels (all honest — query text only):
  bm25    full-text BM25 over context tokens
  dense   e5 cosine
  period  doc gets top rank iff its table periods ∩ query years ≠ ∅

Weighted RRF:  score(d) = Σ_c  w_c / (k + rank_c(d) + 1)

Arms compared + a small weight grid for the period bonus and dense weight.

Usage:
  python scripts/hybrid_period_retrieval.py --sample 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
sys.path.insert(0, "src")

import numpy as np
from tqdm import tqdm

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.core import RetrievalResult
from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are",
         "what", "how", "much", "many", "did", "does", "do", "as", "by", "on", "at",
         "year", "fiscal", "report", "reported", "company", "value", "during", "between"}


def toks(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def doc_period_set(page_content: str) -> set[int]:
    """Report periods of a document = years on the table column headers (precise),
    falling back to any years in the table text."""
    tmd = extract_table(page_content)
    if not tmd:
        return set(extract_years(page_content))
    ledger = extract_ledger_from_table(tmd)
    yrs: set[int] = set()
    for f in ledger.facts:
        if f.period:
            try:
                yrs.add(int(float(str(f.period))))
            except (ValueError, TypeError):
                pass
    if not yrs:  # fallback: years anywhere in the table
        yrs = set(extract_years(tmd))
    return yrs


def ranks_from_scores(scores: np.ndarray, cand: int) -> dict[int, int]:
    top = np.argsort(-scores)[:cand]
    return {int(idx): r for r, idx in enumerate(top)}


def weighted_rrf(channels: list[tuple[float, dict[int, int]]], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = {}
    for w, rl in channels:
        if w == 0:
            continue
        for idx, r in rl.items():
            out[idx] = out.get(idx, 0.0) + w / (k + r + 1)
    return out


def topk(score_dict: dict[int, float], corpus, k: int = 5):
    order = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    return [corpus[idx] for idx, _ in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_t2ragbench_split(args.dataset, split=args.split, sample_size=(args.sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data

    # honest questions (strip injected "company: ")
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    # ---- channels precompute ----
    model = SentenceTransformer(args.embed_model, device=device)
    INSTR = "Given a question about a company, retrieve relevant passages that answer the query."
    print("Encoding corpus ...")
    doc_emb = np.asarray(model.encode([d.page_content for d in corpus], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=True, batch_size=64), "float32")
    q_emb = np.asarray(model.encode([f"Instruct: {INSTR}\nQuery: {q}" for q in raw_q],
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False, batch_size=64), "float32")

    print("BM25 + doc periods ...")
    bm25 = BM25Okapi([toks(d.page_content) for d in corpus])
    doc_years = [doc_period_set(d.page_content) for d in tqdm(corpus, desc="periods")]
    q_years = [set(extract_years(q)) for q in raw_q]
    n_q_with_year = sum(1 for y in q_years if y)

    # ---- precompute per-query channel ranks ONCE (so arms are cheap) ----
    print("Precomputing channel ranks ...")
    bm_ranks, de_ranks = [], []
    for i in tqdm(range(len(raw_q))):
        bm_ranks.append(ranks_from_scores(bm25.get_scores(toks(raw_q[i])), args.cand))
        de_ranks.append(ranks_from_scores(doc_emb @ q_emb[i], args.cand))

    # same-company-cluster subset: queries whose gold doc shares its company with >1 doc
    # (this is where a period tie-break SHOULD matter — pure text can't separate years)
    comp_to_docs: dict[str, int] = {}
    for d in corpus:
        c = str(d.meta_data.get("company_name", "")).lower().strip()
        comp_to_docs[c] = comp_to_docs.get(c, 0) + 1
    cluster_idx = [i for i in range(len(raw_q))
                   if comp_to_docs.get(str(metas[i].get("company_name", "")).lower().strip(), 0) > 1]

    # period is a BOUNDED re-rank: only boosts docs already in the content pool (never
    # introduces a same-year wrong-company doc), with a small bonus = pure tie-break.
    def evaluate(w_bm25, w_dense, period_bonus, subset=None):
        idxs = subset if subset is not None else range(len(raw_q))
        res = []
        for i in idxs:
            fused = weighted_rrf([(w_bm25, bm_ranks[i]), (w_dense, de_ranks[i])])
            qy = q_years[i]
            if period_bonus and qy:
                for idx in list(fused.keys()):
                    if doc_years[idx] & qy:
                        fused[idx] += period_bonus / 60.0
            res.append(RetrievalResult(query=raw_q[i], retrieved_docs=topk(fused, corpus),
                                       ground_truth_id=gts[i], meta_data=metas[i]))
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    arms = {
        "dense_only":                  (0.0, 1.0, 0.0),
        "bm25_only":                   (1.0, 0.0, 0.0),
        "bm25+dense(0.3)":             (1.0, 0.3, 0.0),
        "bm25+dense(1.0)":             (1.0, 1.0, 0.0),
        "bm25+period(0.05)":           (1.0, 0.0, 0.05),
        "bm25+period(0.1)":            (1.0, 0.0, 0.1),
        "bm25+period(0.2)":            (1.0, 0.0, 0.2),
        "bm25+dense(0.3)+period(0.05)":(1.0, 0.3, 0.05),
        "dense+period(0.1)":           (0.0, 1.0, 0.1),
    }
    print("Evaluating arms ...")
    results = {name: evaluate(*w) for name, w in tqdm(arms.items())}

    report = {"dataset": args.dataset, "split": args.split, "n_queries": len(raw_q),
              "n_corpus": len(corpus), "embed_model": args.embed_model,
              "queries_with_year": n_q_with_year, "weights(bm25,dense,period)": arms, "arms": results}
    out_dir = Path(args.out or f"outputs/hybrid_period/{args.dataset.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hybrid_period.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== {args.dataset} (n={len(raw_q)}, corpus={len(corpus)}, "
          f"q_with_year={n_q_with_year}) ===")
    print(f"{'arm':<26}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
    for name, mm in results.items():
        print(f"{name:<26}{mm['MRR@3']:>8}{mm['Recall@1']:>8}{mm['Recall@3']:>8}"
              f"{mm['Recall@5']:>8}{mm['NDCG@3']:>8}")
    print(f"\nSaved to {out_dir}/hybrid_period.json")


if __name__ == "__main__":
    main()
