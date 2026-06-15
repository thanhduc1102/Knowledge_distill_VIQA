#!/usr/bin/env python3
"""Header-aware retrieval prototype on FinQA.

Tests whether matching the query against a document's TABLE HEADERS (row labels +
column headers) is a strong, *honest* retrieval signal — and quantifies the leak from
injecting the gold company name into the query.

Arms (all share one e5 encoding of the corpus):
  dense_honest          dense, query = question only            (no metadata injected)
  dense_leaky           dense, query = "company: question"      (current system → LEAK)
  fulltext_bm25         BM25 over full context, honest query
  header_bm25           BM25 over (row labels + column headers), honest query
  dense_honest+fulltext RRF( dense_honest , fulltext_bm25 )     (paper-style Hybrid, honest)
  dense_honest+header   RRF( dense_honest , header_bm25 )       (our proposal, honest)
  dense_leaky+header    RRF( dense_leaky  , header_bm25 )       (best-effort upper signal)

Reading:
  dense_leaky − dense_honest          = magnitude of the company-injection leak
  (dense+header) − dense_honest       = honest value added by header matching
  header_bm25  vs fulltext_bm25       = does restricting BM25 to headers help?

Usage:
  python scripts/header_retrieval.py --sample 0   # 0 = all 1147 queries
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
from gsr_cacl.core import RetrievalResult
from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are",
         "what", "how", "much", "many", "did", "does", "do", "as", "by", "on", "at",
         "year", "fiscal", "report", "reported", "company", "value", "during", "between"}


def toks(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def header_blob(page_content: str) -> str:
    """Row labels + column headers extracted from the document's first table."""
    tmd = extract_table(page_content)
    if not tmd:
        return ""
    ledger = extract_ledger_from_table(tmd)
    rows = {f.concept for f in ledger.facts if f.concept}
    cols = {f.column_header for f in ledger.facts if f.column_header}
    return " ".join(list(rows) + list(cols))


def rrf(rank_lists: list[dict[int, int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over {doc_idx: rank} dicts (rank starts at 0)."""
    out: dict[int, float] = {}
    for rl in rank_lists:
        for idx, r in rl.items():
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + r + 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--cand", type=int, default=50, help="candidate pool per channel for fusion")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_t2ragbench_split(args.dataset, split=args.split,
                                 sample_size=(args.sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    id_to_idx = {d.id: i for i, d in enumerate(corpus)}

    # honest questions = strip the "company: " prefix the loader injects
    raw_questions = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_questions.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    # ---- dense encoding (shared) ----
    model = SentenceTransformer(args.embed_model, device=device)
    INSTR = "Given a question about a company, retrieve relevant passages that answer the query."

    def enc_queries(qs):
        qs = [f"Instruct: {INSTR}\nQuery: {q}" for q in qs]
        return np.asarray(model.encode(qs, normalize_embeddings=True, convert_to_numpy=True,
                                       show_progress_bar=False, batch_size=64), dtype="float32")

    print("Encoding corpus ...")
    doc_emb = np.asarray(model.encode([d.page_content for d in corpus], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=True, batch_size=64),
                         dtype="float32")
    q_honest = enc_queries(raw_questions)
    q_leaky = enc_queries(data.queries)

    # ---- BM25 channels ----
    print("Building BM25 (full-text + headers) ...")
    full_corpus = [toks(d.page_content) for d in corpus]
    head_corpus = [toks(header_blob(d.page_content)) for d in tqdm(corpus, desc="headers")]
    bm25_full = BM25Okapi(full_corpus)
    bm25_head = BM25Okapi([h or ["<empty>"] for h in head_corpus])

    n_empty_head = sum(1 for h in head_corpus if not h)

    def dense_ranks(qvec_i):
        scores = doc_emb @ qvec_i
        top = np.argsort(-scores)[: args.cand]
        return {int(idx): r for r, idx in enumerate(top)}

    def bm25_ranks(bm25, query_tokens):
        scores = bm25.get_scores(query_tokens)
        top = np.argsort(-scores)[: args.cand]
        return {int(idx): r for r, idx in enumerate(top)}

    def topk_from_scores(score_dict, k=5):
        order = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        return [corpus[idx] for idx, _ in order]

    def topk_from_dense(qvec_i, k=5):
        scores = doc_emb @ qvec_i
        return [corpus[int(idx)] for idx in np.argsort(-scores)[:k]]

    def topk_from_bm25(bm25, qtok, k=5):
        scores = bm25.get_scores(qtok)
        return [corpus[int(idx)] for idx in np.argsort(-scores)[:k]]

    arms: dict[str, list[RetrievalResult]] = {a: [] for a in [
        "dense_honest", "dense_leaky", "fulltext_bm25", "header_bm25",
        "dense_honest+fulltext", "dense_honest+header", "dense_leaky+header"]}

    print("Scoring queries ...")
    for i in tqdm(range(len(raw_questions))):
        gt = gts[i]
        qh_tok = toks(raw_questions[i])
        m = metas[i]

        docs_dh = topk_from_dense(q_honest[i])
        docs_dl = topk_from_dense(q_leaky[i])
        docs_ft = topk_from_bm25(bm25_full, qh_tok)
        docs_hd = topk_from_bm25(bm25_head, qh_tok)

        dh_r = dense_ranks(q_honest[i]); dl_r = dense_ranks(q_leaky[i])
        ft_r = bm25_ranks(bm25_full, qh_tok); hd_r = bm25_ranks(bm25_head, qh_tok)

        fused_dhft = topk_from_scores(rrf([dh_r, ft_r]))
        fused_dhhd = topk_from_scores(rrf([dh_r, hd_r]))
        fused_dlhd = topk_from_scores(rrf([dl_r, hd_r]))

        for name, docs in [
            ("dense_honest", docs_dh), ("dense_leaky", docs_dl),
            ("fulltext_bm25", docs_ft), ("header_bm25", docs_hd),
            ("dense_honest+fulltext", fused_dhft),
            ("dense_honest+header", fused_dhhd),
            ("dense_leaky+header", fused_dlhd),
        ]:
            arms[name].append(RetrievalResult(query=raw_questions[i], retrieved_docs=docs,
                                              ground_truth_id=gt, meta_data=m))

    def metrics(res):
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    results = {name: metrics(res) for name, res in arms.items()}
    report = {"dataset": args.dataset, "split": args.split, "n_queries": len(raw_questions),
              "n_corpus": len(corpus), "embed_model": args.embed_model,
              "empty_header_docs": n_empty_head, "arms": results}

    out_dir = Path(args.out or f"outputs/header_retrieval/{args.dataset.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "header_retrieval.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== {args.dataset} (n={len(raw_questions)}, corpus={len(corpus)}, "
          f"empty_headers={n_empty_head}) ===")
    print(f"{'arm':<26}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
    for name, mm in results.items():
        print(f"{name:<26}{mm['MRR@3']:>8}{mm['Recall@1']:>8}{mm['Recall@3']:>8}"
              f"{mm['Recall@5']:>8}{mm['NDCG@3']:>8}")
    print(f"\nSaved to {out_dir}/header_retrieval.json")


if __name__ == "__main__":
    main()
