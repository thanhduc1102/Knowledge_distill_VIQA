#!/usr/bin/env python3
"""External benchmark: FinanceBench evidence retrieval.

This deliberately evaluates outside T²-RAGBench to control the reviewer concern that the
main result is tied to company/year artifacts.  FinanceBench open-source rows provide
gold evidence snippets.  We build a retrieval corpus from unique evidence pages/snippets
and test whether a query retrieves one of its annotated evidence snippets.

The corpus is not a full-PDF crawl; the output labels this as an evidence-retrieval
setting.  It is still a genuine external benchmark with different documents, question
style, and annotation source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
from datasets import load_dataset

from gsr_cacl.research.simple_ir import SimpleBM25, minmax, rank_of_gold, ranking_metrics, tokenize
from gsr_cacl.ontology.aliases import normalize_company


def _evidence_id(row: dict, ev: dict, k: int) -> str:
    doc = ev.get("doc_name") or row.get("doc_name") or "doc"
    page = ev.get("evidence_page_num", k)
    return f"{doc}::p{page}::{k}"


def load_financebench(sample: int = 0):
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = [dict(r) for r in ds]
    if sample:
        rows = rows[:sample]

    corpus, seen = [], {}
    queries = []
    for qi, r in enumerate(rows):
        gold_ids = []
        for k, ev in enumerate(r.get("evidence") or []):
            text = ev.get("evidence_text") or ev.get("evidence_text_full_page") or ""
            if not text.strip():
                continue
            eid = _evidence_id(r, ev, k)
            if eid not in seen:
                seen[eid] = len(corpus)
                corpus.append({
                    "id": eid,
                    "text": text,
                    "company": str(r.get("company") or ""),
                    "doc_name": str(ev.get("doc_name") or r.get("doc_name") or ""),
                    "doc_period": str(r.get("doc_period") or ""),
                })
            gold_ids.append(seen[eid])
        if gold_ids:
            queries.append({
                "id": str(r.get("financebench_id") or qi),
                "question": str(r.get("question") or ""),
                "company": str(r.get("company") or ""),
                "doc_period": str(r.get("doc_period") or ""),
                "gold": gold_ids,
            })
    return queries, corpus


def _local_bm25_scores(query: str, pool: list[int], corpus_tokens: list[list[str]]) -> np.ndarray:
    bm25 = SimpleBM25([corpus_tokens[i] for i in pool])
    return bm25.scores(tokenize(query))


def _mask_query(query: str, company: str, doc_period: str) -> str:
    toks = tokenize(query)
    company_toks = set(tokenize(company))
    year_toks = {str(doc_period)} if doc_period else set()
    keep = [t for t in toks if t not in company_toks and t not in year_toks]
    return " ".join(keep)


def _bootstrap_delta(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray([1.0 / (r + 1) if r < 3 else 0.0 for r in a], dtype=np.float64)
    b = np.asarray([1.0 / (r + 1) if r < 3 else 0.0 for r in b], dtype=np.float64)
    diff = b - a
    boots = []
    n = len(diff)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diff[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta_mrr3": round(float(diff.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_delta_le_0": round(float(np.mean(np.asarray(boots) <= 0.0)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--with-reranker", action="store_true",
                    help="Optionally add a cross-encoder reranker if the model is available.")
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/external_financebench")
    args = ap.parse_args()

    queries, corpus = load_financebench(args.sample)
    texts = [d["text"] for d in corpus]
    bm25 = SimpleBM25.from_texts(texts)
    corpus_tokens = [tokenize(t) for t in texts]

    comp2idx: dict[str, list[int]] = {}
    period2idx: dict[tuple[str, str], list[int]] = {}
    for i, d in enumerate(corpus):
        ck = normalize_company(d["company"])
        comp2idx.setdefault(ck, []).append(i)
        period2idx.setdefault((ck, d["doc_period"]), []).append(i)

    reranker = None
    if args.with_reranker:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(args.reranker_model)
        except Exception as e:
            print(f"[warn] reranker unavailable: {e}", flush=True)

    rng = np.random.default_rng(args.seed)
    ranks = {
        "bm25": [],
        "bm25_masked_company_year": [],
        "company_random": [],
        "company_year_random": [],
        "company_loclex": [],
        "company_year_loclex": [],
        "bm25_rerank": [],
        "company_loclex_rerank": [],
    }
    pool_stats = {"company": [], "company_year": []}
    examples = []

    for q in queries:
        scores = bm25.scores(q["question"])
        full_order = np.argsort(-scores)
        ranks["bm25"].append(min(rank_of_gold(full_order, g) for g in q["gold"]))

        masked_scores = bm25.scores(_mask_query(q["question"], q["company"], q["doc_period"]))
        masked_order = np.argsort(-masked_scores)
        ranks["bm25_masked_company_year"].append(min(rank_of_gold(masked_order, g) for g in q["gold"]))

        ck = normalize_company(q["company"])
        comp_pool = comp2idx.get(ck, list(range(len(corpus))))
        pool_stats["company"].append(len(comp_pool))
        comp_random = comp_pool.copy()
        rng.shuffle(comp_random)
        ranks["company_random"].append(min(rank_of_gold(comp_random, g) for g in q["gold"]))
        loc_scores = _local_bm25_scores(q["question"], comp_pool, corpus_tokens)
        comp_order = [comp_pool[i] for i in np.argsort(-loc_scores)]
        ranks["company_loclex"].append(min(rank_of_gold(comp_order, g) for g in q["gold"]))

        cypool = period2idx.get((ck, q["doc_period"]), comp_pool)
        pool_stats["company_year"].append(len(cypool))
        cy_random = cypool.copy()
        rng.shuffle(cy_random)
        ranks["company_year_random"].append(min(rank_of_gold(cy_random, g) for g in q["gold"]))
        cy_scores = _local_bm25_scores(q["question"], cypool, corpus_tokens)
        cy_order = [cypool[i] for i in np.argsort(-cy_scores)]
        ranks["company_year_loclex"].append(min(rank_of_gold(cy_order, g) for g in q["gold"]))

        if reranker is not None:
            cand = full_order[: min(30, len(full_order))].tolist()
            pairs = [(q["question"], corpus[i]["text"][:3000]) for i in cand]
            rr = np.asarray(reranker.predict(pairs), dtype=np.float64)
            fused = minmax(scores[cand]) + minmax(rr)
            order = [cand[i] for i in np.argsort(-fused)]
            ranks["bm25_rerank"].append(min(rank_of_gold(order, g) for g in q["gold"]))

            pairs_local = [(q["question"], corpus[i]["text"][:3000]) for i in comp_pool]
            rr_local = np.asarray(reranker.predict(pairs_local), dtype=np.float64)
            fused_local = minmax(loc_scores) + minmax(rr_local)
            local_order = [comp_pool[i] for i in np.argsort(-fused_local)]
            ranks["company_loclex_rerank"].append(min(rank_of_gold(local_order, g) for g in q["gold"]))

        if len(examples) < 5:
            examples.append({
                "query": q["question"],
                "gold_ids": [corpus[g]["id"] for g in q["gold"]],
                "bm25_top1": corpus[int(full_order[0])]["id"],
                "company_loclex_top1": corpus[int(comp_order[0])]["id"] if comp_order else None,
            })

    out = {
        "benchmark": "FinanceBench open-source evidence retrieval",
        "setting_note": "Corpus is built from annotated evidence snippets/pages, not full PDFs.",
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "avg_company_pool": round(float(np.mean(pool_stats["company"])), 2),
        "avg_company_year_pool": round(float(np.mean(pool_stats["company_year"])), 2),
        "metrics": {k: ranking_metrics(v) for k, v in ranks.items() if v},
        "bootstrap": {
            "company_loclex_vs_bm25": _bootstrap_delta(ranks["bm25"], ranks["company_loclex"], seed=args.seed),
            "company_year_loclex_vs_company_random": _bootstrap_delta(
                ranks["company_year_random"], ranks["company_year_loclex"], seed=args.seed
            ),
            "company_year_loclex_vs_company_loclex": _bootstrap_delta(
                ranks["company_loclex"], ranks["company_year_loclex"], seed=args.seed
            ),
        },
        "per_query_ranks": [
            {name: int(vals[i]) for name, vals in ranks.items() if vals}
            for i in range(len(queries))
        ],
        "examples": examples,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "financebench_retrieval.json").write_text(json.dumps(out, indent=2))

    print("\n=== FinanceBench evidence retrieval ===")
    print(f"queries={out['n_queries']} corpus={out['n_corpus']} "
          f"avg_company_pool={out['avg_company_pool']}")
    for name, m in out["metrics"].items():
        print(f"{name:<24} MRR@3={m['MRR@3']:.4f} R@1={m['R@1']:.4f} "
              f"R@3={m['R@3']:.4f} R@5={m['R@5']:.4f}")
    print("bootstrap:")
    for k, v in out["bootstrap"].items():
        print(f"  {k:<40} Δ={v['delta_mrr3']:+.4f} CI={v['ci95']} p<=0={v['p_delta_le_0']}")
    print(f"Saved -> {Path(args.out) / 'financebench_retrieval.json'}")


if __name__ == "__main__":
    main()
