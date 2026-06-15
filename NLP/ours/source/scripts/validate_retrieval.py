#!/usr/bin/env python3
"""Validate the honest retrieval recipe across all 3 T²-RAGBench subsets and pick optima.

Runs a weight grid {bm25, dense, period} on FinQA / ConvFinQA / TAT-DQA (honest queries —
years read from the question, no gold metadata injected), prints per-dataset tables, and
reports the query-count-weighted average MRR@3 per arm to choose a single best method.

Usage:
  python scripts/validate_retrieval.py --sample 0     # 0 = full
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

DATASETS = [("FinQA", "test"), ("ConvFinQA", "turn_0"), ("TAT-DQA", "test")]

# (w_bm25, w_dense, period_bonus, gate_thr, adaptive)
#   gate_thr  = apply period only if <= this fraction of the candidate pool matches the
#               query year (period must be DISCRIMINATIVE). 1.0 = always (old behavior).
#   adaptive  = scale bonus by (1 - pool_match_fraction): an IDF-style down-weight when
#               most candidates share the year (the TAT-DQA failure mode: 80% multi-year).
ARMS = {
    "dense_only":                    (0.0, 1.0, 0.0, 1.0, False),
    "bm25_only":                     (1.0, 0.0, 0.0, 1.0, False),
    "bm25+period_fixed(0.05)":       (1.0, 0.0, 0.05, 1.0, False),
    "bm25+period_gated(0.05,thr.6)": (1.0, 0.0, 0.05, 0.6, False),
    "bm25+period_gated(0.05,thr.4)": (1.0, 0.0, 0.05, 0.4, False),
    "bm25+period_adaptive(0.1)":     (1.0, 0.0, 0.10, 1.0, True),
    "bm25+period_adaptive(0.2)":     (1.0, 0.0, 0.20, 1.0, True),
    "bm25+period_adapt_gated(0.2,.6)":(1.0, 0.0, 0.20, 0.6, True),
}


def toks(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def doc_period_set(page_content: str) -> set[int]:
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
    return yrs or set(extract_years(tmd))


def ranks_from_scores(scores: np.ndarray, cand: int) -> dict[int, int]:
    return {int(idx): r for r, idx in enumerate(np.argsort(-scores)[:cand])}


def evaluate_dataset(model, config_name, split, sample, cand, rrf_k=60):
    from rank_bm25 import BM25Okapi

    data = load_t2ragbench_split(config_name, split=split, sample_size=(sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    INSTR = "Given a question about a company, retrieve relevant passages that answer the query."
    doc_emb = np.asarray(model.encode([d.page_content for d in corpus], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False, batch_size=64), "float32")
    q_emb = np.asarray(model.encode([f"Instruct: {INSTR}\nQuery: {q}" for q in raw_q],
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False, batch_size=64), "float32")
    bm25 = BM25Okapi([toks(d.page_content) for d in corpus])
    doc_years = [doc_period_set(d.page_content) for d in corpus]
    q_years = [set(extract_years(q)) for q in raw_q]

    bm_ranks = [ranks_from_scores(bm25.get_scores(toks(raw_q[i])), cand) for i in range(len(raw_q))]
    de_ranks = [ranks_from_scores(doc_emb @ q_emb[i], cand) for i in range(len(raw_q))]

    def run(w_bm25, w_dense, period_bonus, gate_thr=1.0, adaptive=False):
        res = []
        for i in range(len(raw_q)):
            fused: dict[int, float] = {}
            for w, rl in ((w_bm25, bm_ranks[i]), (w_dense, de_ranks[i])):
                if w == 0:
                    continue
                for idx, r in rl.items():
                    fused[idx] = fused.get(idx, 0.0) + w / (rrf_k + r + 1)
            qy = q_years[i]
            if period_bonus and qy and fused:
                matchers = [idx for idx in fused if doc_years[idx] & qy]
                frac = len(matchers) / len(fused)
                if frac <= gate_thr:                    # period must be discriminative
                    eff = period_bonus * (1.0 - frac) if adaptive else period_bonus
                    for idx in matchers:
                        fused[idx] += eff / rrf_k
            order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]
            res.append(RetrievalResult(query=raw_q[i], retrieved_docs=[corpus[i2] for i2, _ in order],
                                       ground_truth_id=gts[i], meta_data=metas[i]))
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    return len(raw_q), {name: run(*w) for name, w in ARMS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--out", default="outputs/validate_retrieval")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.embed_model, device=device)

    per_ds = {}
    counts = {}
    for cfg, split in DATASETS:
        print(f"\n### {cfg} ({split}) ###")
        n, res = evaluate_dataset(model, cfg, split, args.sample, args.cand)
        per_ds[cfg] = res
        counts[cfg] = n
        print(f"{'arm':<32}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
        for name, mm in res.items():
            print(f"{name:<32}{mm['MRR@3']:>8}{mm['Recall@1']:>8}{mm['Recall@3']:>8}"
                  f"{mm['Recall@5']:>8}{mm['NDCG@3']:>8}")

    total = sum(counts.values())
    wavg = {}
    for name in ARMS:
        wavg[name] = round(sum(per_ds[c][name]["MRR@3"] * counts[c] for c in counts) / total, 4)
    best = max(wavg.items(), key=lambda x: x[1])

    print(f"\n### Query-count-weighted avg MRR@3 (total n={total}) ###")
    for name, v in sorted(wavg.items(), key=lambda x: -x[1]):
        print(f"  {name:<32}{v:>8}")
    print(f"\nBEST METHOD: {best[0]}  (w.avg MRR@3 = {best[1]})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validate_retrieval.json").write_text(json.dumps(
        {"counts": counts, "per_dataset": per_ds, "weighted_avg_mrr3": wavg,
         "best_method": best[0], "best_wavg_mrr3": best[1], "arms": ARMS}, indent=2))
    print(f"\nSaved to {out_dir}/validate_retrieval.json")


if __name__ == "__main__":
    main()
