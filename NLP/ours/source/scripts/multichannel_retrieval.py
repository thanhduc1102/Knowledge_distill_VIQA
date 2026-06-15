#!/usr/bin/env python3
"""Multi-channel honest hybrid retrieval: text (BM25) + dense + metadata(period) + table-structure.

Synthesis of three retrieval families, fused by weighted RRF + bounded additive boosts:
  * TEXT      — full-text BM25 (the measured backbone; dense is weak here)
  * METADATA  — period (year) match, GATED to stay discriminative (the validated signal)
  * TABLE/GRAPH — concept-coverage: does the document's Fact-Ledger cover the canonical
                  concept(s) + period the QUESTION asks about? This is the structure signal
                  that targets the dominant text-and-table failure mode (≈73% of retrieval
                  errors are table-structure mismatch — markdown cells don't embed well).

All signals are HONEST: concepts/years/entities are read from the question, never from gold
metadata. Tests whether the table-structure channel adds on top of the BM25+period optimum.

Usage:
  python scripts/multichannel_retrieval.py --sample 0
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

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.scoring.concept_coverage import concept_coverage_score, expand_derivable
from gsr_cacl.core import RetrievalResult
from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are",
         "what", "how", "much", "many", "did", "does", "do", "as", "by", "on", "at",
         "year", "fiscal", "report", "reported", "company", "value", "during", "between"}

DATASETS = [("FinQA", "test"), ("ConvFinQA", "turn_0"), ("TAT-DQA", "test")]


def toks(text: str):
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def doc_concepts_periods(page_content: str):
    tmd = extract_table(page_content)
    if not tmd:
        return set(), set(extract_years(page_content))
    L = extract_ledger_from_table(tmd)
    concepts = {f.concept_canonical for f in L.facts if f.concept_canonical}
    yrs = set()
    for f in L.facts:
        if f.period:
            try:
                yrs.add(int(float(str(f.period))))
            except (ValueError, TypeError):
                pass
    if not yrs:
        yrs = set(extract_years(tmd))
    return concepts, yrs


def ranks_from_scores(scores, cand):
    return {int(idx): r for r, idx in enumerate(np.argsort(-scores)[:cand])}


def evaluate_dataset(model, cfg, split, sample, cand=50, rrf_k=60):
    from rank_bm25 import BM25Okapi

    data = load_t2ragbench_split(cfg, split=split, sample_size=(sample or None))
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

    doc_cp = [doc_concepts_periods(d.page_content) for d in corpus]
    doc_concepts = [c for c, _ in doc_cp]
    doc_years = [y for _, y in doc_cp]
    q_years = [set(extract_years(q)) for q in raw_q]
    q_concepts = [concepts_in_text(q) for q in raw_q]

    bm_ranks = [ranks_from_scores(bm25.get_scores(toks(q)), cand) for q in raw_q]
    de_ranks = [ranks_from_scores(doc_emb @ q_emb[i], cand) for i in range(len(raw_q))]

    def run(w_bm25, w_dense, period_bonus, period_gate, concept_bonus):
        res = []
        for i in range(len(raw_q)):
            fused = {}
            for w, rl in ((w_bm25, bm_ranks[i]), (w_dense, de_ranks[i])):
                if w == 0:
                    continue
                for idx, r in rl.items():
                    fused[idx] = fused.get(idx, 0.0) + w / (rrf_k + r + 1)
            qy, qc = q_years[i], q_concepts[i]
            # metadata: period (gated, discriminative)
            if period_bonus and qy and fused:
                matchers = [idx for idx in fused if doc_years[idx] & qy]
                if len(matchers) / len(fused) <= period_gate:
                    for idx in matchers:
                        fused[idx] += period_bonus / rrf_k
            # table/graph: concept-coverage of the question's concept+period by the ledger
            if concept_bonus and qc and fused:
                for idx in list(fused.keys()):
                    cov = concept_coverage_score(qc, qy, doc_concepts[idx], doc_years[idx])
                    fused[idx] += concept_bonus * cov / rrf_k
            order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]
            res.append(RetrievalResult(query=raw_q[i], retrieved_docs=[corpus[i2] for i2, _ in order],
                                       ground_truth_id=gts[i], meta_data=metas[i]))
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    # (w_bm25, w_dense, period_bonus, period_gate, concept_bonus)
    arms = {
        "bm25_only":                 (1.0, 0.0, 0.0, 0.4, 0.0),
        "bm25+period":               (1.0, 0.0, 0.05, 0.4, 0.0),
        "bm25+period+concept(0.1)":  (1.0, 0.0, 0.05, 0.4, 0.1),
        "bm25+period+concept(0.3)":  (1.0, 0.0, 0.05, 0.4, 0.3),
        "bm25+period+concept(0.5)":  (1.0, 0.0, 0.05, 0.4, 0.5),
        "bm25+dense+period+concept": (1.0, 0.3, 0.05, 0.4, 0.3),
    }
    cov_q = sum(1 for c in q_concepts if c) / len(q_concepts)
    return len(raw_q), cov_q, {name: run(*w) for name, w in arms.items()}, list(arms.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--out", default="outputs/multichannel_retrieval")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model, device="cuda" if torch.cuda.is_available() else "cpu")

    per_ds, counts, covq = {}, {}, {}
    arm_names = None
    for cfg, split in DATASETS:
        print(f"\n### {cfg} ({split}) ###")
        n, cq, res, arm_names = evaluate_dataset(model, cfg, split, args.sample)
        per_ds[cfg] = res; counts[cfg] = n; covq[cfg] = round(cq, 3)
        print(f"(queries with a canonical concept: {cq:.1%})")
        print(f"{'arm':<30}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
        for name, mm in res.items():
            print(f"{name:<30}{mm['MRR@3']:>8}{mm['Recall@1']:>8}{mm['Recall@3']:>8}"
                  f"{mm['Recall@5']:>8}{mm['NDCG@3']:>8}")

    total = sum(counts.values())
    wavg = {name: round(sum(per_ds[c][name]["MRR@3"] * counts[c] for c in counts) / total, 4)
            for name in arm_names}
    best = max(wavg.items(), key=lambda x: x[1])
    print(f"\n### weighted-avg MRR@3 (n={total}) ###")
    for name, v in sorted(wavg.items(), key=lambda x: -x[1]):
        print(f"  {name:<30}{v:>8}")
    print(f"\nBEST: {best[0]}  (w.avg MRR@3 = {best[1]})")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "multichannel_retrieval.json").write_text(json.dumps(
        {"counts": counts, "queries_with_concept": covq, "per_dataset": per_ds,
         "weighted_avg_mrr3": wavg, "best": best[0]}, indent=2))
    print(f"\nSaved to {args.out}/multichannel_retrieval.json")


if __name__ == "__main__":
    main()
