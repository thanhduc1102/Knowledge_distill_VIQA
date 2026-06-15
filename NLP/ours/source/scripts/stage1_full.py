#!/usr/bin/env python3
"""Stage-1 full multi-signal retrieval (honest) — completes B (gated concept) + C (cell match).

Channels, all read from the QUESTION only (no gold metadata):
  TEXT     bm25       full-text BM25 (backbone)
  META     period     year match, GATED (apply only when ≤ gate_thr of pool matches)
  STRUCT   concept    canonical concept-coverage (ledger), GATED
  TABLE    cellmatch  cell-level: best ledger row-label token overlap × period factor
                      (TableRAG/FT-RAG-style; uses RAW row labels, not just canonical concepts,
                      so it covers the ~86% of facts the small ontology misses)

cellmatch(q, d) = max_f [ jaccard(q_content_tokens, tokens(f.concept))
                          × (1.0 if f.period∈q_years else 0.6 if q_year∈doc_years else 0.3) ]

All structure signals are bounded additive boosts in the BM25 candidate pool (never introduce
a new doc) and gated to stay discriminative (the fix that saved period on TAT-DQA).

Usage: python scripts/stage1_full.py --sample 0
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
from gsr_cacl.ontology.concepts import concepts_in_text
from gsr_cacl.scoring.concept_coverage import concept_coverage_score
from gsr_cacl.core import RetrievalResult
from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the","a","an","of","in","for","to","and","or","was","is","are","what","how","much",
         "many","did","does","do","as","by","on","at","year","fiscal","report","reported",
         "company","value","during","between","change","total"}
DATASETS = [("FinQA", "test"), ("ConvFinQA", "turn_0"), ("TAT-DQA", "test")]


def toks(text: str) -> set[str]:
    return {t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def bm_toks(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def doc_facts(page_content: str):
    """Per-doc: canonical concepts, periods, and list of (row-label tokens, period) for cell match."""
    tmd = extract_table(page_content)
    if not tmd:
        return set(), set(extract_years(page_content)), []
    L = extract_ledger_from_table(tmd)
    concepts = {f.concept_canonical for f in L.facts if f.concept_canonical}
    yrs, cells = set(), []
    for f in L.facts:
        p = None
        if f.period:
            try:
                p = int(float(str(f.period))); yrs.add(p)
            except (ValueError, TypeError):
                pass
        cells.append((toks(f.concept), p))
    if not yrs:
        yrs = set(extract_years(tmd))
    return concepts, yrs, cells


def ranks_from_scores(scores, cand):
    return {int(idx): r for r, idx in enumerate(np.argsort(-scores)[:cand])}


def cell_match(q_content: set, q_years: set, cells, doc_years: set) -> float:
    if not q_content or not cells:
        return 0.0
    best = 0.0
    for ctoks, p in cells:
        ov = jaccard(q_content, ctoks)
        if ov <= 0:
            continue
        if p is not None and p in q_years:
            pf = 1.0
        elif q_years & doc_years:
            pf = 0.6
        else:
            pf = 0.3
        best = max(best, ov * pf)
    return best


def evaluate_dataset(model, cfg, split, sample, cand=50, rrf_k=60):
    from rank_bm25 import BM25Okapi
    data = load_t2ragbench_split(cfg, split=split, sample_size=(sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c)+1:].lstrip() if c and q.startswith(c + ":") else q)

    INSTR = "Given a question about a company, retrieve relevant passages that answer the query."
    doc_emb = np.asarray(model.encode([d.page_content for d in corpus], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False, batch_size=64), "float32")
    q_emb = np.asarray(model.encode([f"Instruct: {INSTR}\nQuery: {q}" for q in raw_q],
                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=64), "float32")
    bm25 = BM25Okapi([bm_toks(d.page_content) for d in corpus])

    dfacts = [doc_facts(d.page_content) for d in corpus]
    doc_concepts = [a for a, _, _ in dfacts]
    doc_years = [b for _, b, _ in dfacts]
    doc_cells = [c for _, _, c in dfacts]
    q_years = [set(extract_years(q)) for q in raw_q]
    q_concepts = [concepts_in_text(q) for q in raw_q]
    q_content = [toks(q) for q in raw_q]
    bm_ranks = [ranks_from_scores(bm25.get_scores(bm_toks(q)), cand) for q in raw_q]
    de_ranks = [ranks_from_scores(doc_emb @ q_emb[i], cand) for i in range(len(raw_q))]

    def run(w_dense, period_b, concept_b, cell_b, gate=0.4):
        res = []
        for i in range(len(raw_q)):
            fused = {}
            for w, rl in ((1.0, bm_ranks[i]), (w_dense, de_ranks[i])):
                if w == 0:
                    continue
                for idx, r in rl.items():
                    fused[idx] = fused.get(idx, 0.0) + w / (rrf_k + r + 1)
            qy, qc, qct = q_years[i], q_concepts[i], q_content[i]
            keys = list(fused.keys())
            # META period (gated)
            if period_b and qy:
                ms = [idx for idx in keys if doc_years[idx] & qy]
                if ms and len(ms) / len(keys) <= gate:
                    for idx in ms:
                        fused[idx] += period_b / rrf_k
            # STRUCT concept-coverage (gated)
            if concept_b and qc:
                covs = {idx: concept_coverage_score(qc, qy, doc_concepts[idx], doc_years[idx]) for idx in keys}
                hi = [idx for idx, c in covs.items() if c >= 0.5]
                if hi and len(hi) / len(keys) <= gate:
                    for idx in hi:
                        fused[idx] += concept_b * covs[idx] / rrf_k
            # TABLE cell-match (graded; gate on discriminativeness)
            if cell_b and qct:
                cm = {idx: cell_match(qct, qy, doc_cells[idx], doc_years[idx]) for idx in keys}
                hi = [idx for idx, c in cm.items() if c >= 0.34]
                if hi and len(hi) / len(keys) <= gate:
                    for idx in hi:
                        fused[idx] += cell_b * cm[idx] / rrf_k
            order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]
            res.append(RetrievalResult(query=raw_q[i], retrieved_docs=[corpus[i2] for i2, _ in order],
                                       ground_truth_id=gts[i], meta_data=metas[i]))
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    arms = {
        "bm25+period":                  (0.0, 0.05, 0.0, 0.0),
        "+concept_gated(0.1)":          (0.0, 0.05, 0.1, 0.0),
        "+cellmatch(0.3)":              (0.0, 0.05, 0.0, 0.3),
        "+cellmatch(0.5)":              (0.0, 0.05, 0.0, 0.5),
        "+concept+cellmatch(0.1,0.3)":  (0.0, 0.05, 0.1, 0.3),
        "+concept+cellmatch(0.1,0.5)":  (0.0, 0.05, 0.1, 0.5),
    }
    return len(raw_q), {n: run(*w) for n, w in arms.items()}, list(arms.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--out", default="outputs/stage1_full")
    args = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model, device="cuda" if torch.cuda.is_available() else "cpu")

    per_ds, counts, names = {}, {}, None
    for cfg, split in DATASETS:
        print(f"\n### {cfg} ({split}) ###")
        n, res, names = evaluate_dataset(model, cfg, split, args.sample)
        per_ds[cfg], counts[cfg] = res, n
        print(f"{'arm':<32}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
        for nm, mm in res.items():
            print(f"{nm:<32}{mm['MRR@3']:>8}{mm['Recall@1']:>8}{mm['Recall@3']:>8}{mm['Recall@5']:>8}{mm['NDCG@3']:>8}")
    total = sum(counts.values())
    wavg = {nm: round(sum(per_ds[c][nm]["MRR@3"] * counts[c] for c in counts) / total, 4) for nm in names}
    best = max(wavg.items(), key=lambda x: x[1])
    print(f"\n### weighted-avg MRR@3 (n={total}) ###")
    for nm, v in sorted(wavg.items(), key=lambda x: -x[1]):
        print(f"  {nm:<32}{v:>8}")
    print(f"\nBEST: {best[0]} (w.avg MRR@3={best[1]})")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "stage1_full.json").write_text(json.dumps(
        {"counts": counts, "per_dataset": per_ds, "weighted_avg_mrr3": wavg, "best": best[0]}, indent=2))
    print(f"\nSaved to {args.out}/stage1_full.json")


if __name__ == "__main__":
    main()
