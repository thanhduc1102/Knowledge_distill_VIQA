#!/usr/bin/env python3
"""C1 — Artifact-Controlled Difficulty Decomposition.

The honest diagnostic behind the AAAI-27 framing (docs/RESEARCH_AAAI27.md §2, C1).
It separates the part of retrieval performance that is a *benchmark artifact* (gold
always shares the query's company → company-scoping trivialises recall) from the
*residual content-only difficulty* that actually generalises.

Regimes measured per query (all use sparse signals only — no LLM, no neural):
  A  corpus-BM25            rank the WHOLE corpus by global BM25.
  A' corpus-BM25 (masked)   same, but the company's tokens are removed from the query
                            → estimates how much of A is pure metadata/lexical company leak.
  B  company-pool + BM25    restrict to the query company's chunks, rank by GLOBAL BM25.
  C  company-pool + loclex  restrict to the company's chunks, rank by POOL-LOCAL IDF.
  D  within-(company,year)  restrict to company AND year, rank by loclex (the hard residual).

Outputs MRR@3 / R@1/3/5 per regime + pool recall + coverage stats, to
``outputs/research/difficulty/<dataset>.json``.

Usage:  PYTHONPATH=src python scripts/research/difficulty_decomposition.py --dataset FinQA --sample 0
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.experts.lexical import LexicalExpert, _toks
from gsr_cacl.experts.local_lexical import LocalLexicalExpert
from gsr_cacl.experts.meta_retriever import MetadataRetriever
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.aliases import normalize_company, significant_tokens

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}


def mrr_metrics(ranks: list[int]) -> dict:
    ranks = [r for r in ranks]
    return {
        "MRR@3": round(float(np.mean([1.0 / (r + 1) if r < 3 else 0.0 for r in ranks])), 4),
        "R@1": round(float(np.mean([r < 1 for r in ranks])), 4),
        "R@3": round(float(np.mean([r < 3 for r in ranks])), 4),
        "R@5": round(float(np.mean([r < 5 for r in ranks])), 4),
        "n": len(ranks),
    }


def rank_in(scores: np.ndarray, pool: list[int], gold_idx: int) -> int:
    """Rank (0-based) of gold within ``pool`` ordered by ``scores`` (aligned to pool). ∞ if absent."""
    if gold_idx not in pool:
        return 10**9
    order = np.argsort(-scores)
    gp = pool.index(gold_idx)
    hit = np.where(order == gp)[0]
    return int(hit[0]) if hit.size else 10**9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=list(SPLITS))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/difficulty")
    args = ap.parse_args()

    split = SPLITS[args.dataset]
    data = load_t2ragbench_split(args.dataset, split=split, sample_size=(args.sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data

    # Honest question text: strip the "company:" prefix (as in modular_retrieval).
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]

    # ---- shared experts --------------------------------------------------------
    lex = LexicalExpert(abbr_expand=True)
    lex.prepare(corpus, doc_metas); lex.set_queries(raw_q, metas)
    loc = LocalLexicalExpert(abbr_expand=True)
    loc.prepare(corpus, doc_metas); loc.set_queries(raw_q, metas)
    meta = MetadataRetriever(company_pool=True, max_add=10000)  # full company set
    meta.prepare(corpus, doc_metas); meta.set_queries(raw_q, metas)

    # Company-masked query tokens (regime A'): drop the company's significant tokens.
    full_pool = list(range(len(corpus)))
    Q = len(raw_q)

    # doc year per corpus doc (for regime D)
    doc_year = meta._doc_year  # reuse MetadataRetriever's parsed years

    ranks = {k: [] for k in ("A", "A_masked", "B", "C", "D")}
    pool_in = {k: [] for k in ("B", "C", "D")}
    comp_sizes, comp_year_sizes = [], []
    q_has_company, q_has_year = 0, 0
    d_eligible = 0

    for qi in range(Q):
        gi = gold_idx[qi]
        if gi < 0:
            continue
        # --- A: corpus BM25 ---
        s_lex_full = lex.full_scores(qi)
        ranks["A"].append(rank_in(s_lex_full[full_pool], full_pool, gi))

        # --- A': corpus BM25 with company tokens masked from the query ---
        comp = str((metas[qi] or {}).get("company_name", "")).strip()
        comp_tokens = set(significant_tokens(comp)) if comp else set()
        masked_q_tokens = [t for t in _toks(raw_q[qi]) if t not in comp_tokens]
        s_masked = np.asarray(lex._bm25.get_scores(masked_q_tokens), dtype=np.float64)
        ranks["A_masked"].append(rank_in(s_masked[full_pool], full_pool, gi))

        # --- company pool ---
        company_pool = meta.get_candidates(qi, 10000)
        has_company = bool(meta._q_company_key[qi])
        q_has_company += int(has_company)
        if company_pool:
            comp_sizes.append(len(company_pool))
            # B: global BM25 within company pool
            ranks["B"].append(rank_in(s_lex_full[company_pool], company_pool, gi))
            pool_in["B"].append(gi in company_pool)
            # C: pool-local IDF within company pool
            s_loc = loc.score_pool(qi, company_pool)
            ranks["C"].append(rank_in(s_loc, company_pool, gi))
            pool_in["C"].append(gi in company_pool)
        else:
            for k in ("B", "C"):
                ranks[k].append(10**9); pool_in[k].append(False)

        # --- D: within (company, year) ---
        q_yrs = meta._q_years[qi]
        q_has_year += int(bool(q_yrs))
        if company_pool and q_yrs:
            cy_pool = [d for d in company_pool if doc_year[d] in q_yrs]
            if cy_pool:
                d_eligible += 1
                comp_year_sizes.append(len(cy_pool))
                s_loc_cy = loc.score_pool(qi, cy_pool)
                ranks["D"].append(rank_in(s_loc_cy, cy_pool, gi))
                pool_in["D"].append(gi in cy_pool)
            else:
                ranks["D"].append(10**9); pool_in["D"].append(False)
        else:
            ranks["D"].append(10**9); pool_in["D"].append(False)

    out = {
        "dataset": args.dataset,
        "n_queries": len([g for g in gold_idx if g >= 0]),
        "corpus_size": len(corpus),
        "coverage": {
            "q_has_company_frac": round(q_has_company / max(Q, 1), 4),
            "q_has_year_frac": round(q_has_year / max(Q, 1), 4),
            "avg_company_pool": round(float(np.mean(comp_sizes)), 2) if comp_sizes else 0.0,
            "avg_company_year_pool": round(float(np.mean(comp_year_sizes)), 2) if comp_year_sizes else 0.0,
            "D_eligible_frac": round(d_eligible / max(Q, 1), 4),
        },
        "pool_recall": {k: round(float(np.mean(v)), 4) for k, v in pool_in.items()},
        "regimes": {
            "A_corpus_bm25": mrr_metrics(ranks["A"]),
            "A_corpus_bm25_company_masked": mrr_metrics(ranks["A_masked"]),
            "B_companypool_bm25": mrr_metrics(ranks["B"]),
            "C_companypool_loclex": mrr_metrics(ranks["C"]),
            "D_within_company_year_loclex": mrr_metrics(ranks["D"]),
        },
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset.lower()}.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== C1 Difficulty Decomposition — {args.dataset} "
          f"(N={out['n_queries']}, corpus={len(corpus)}) ===")
    print(f"company in Q: {out['coverage']['q_has_company_frac']:.1%} | "
          f"year in Q: {out['coverage']['q_has_year_frac']:.1%} | "
          f"avg company pool: {out['coverage']['avg_company_pool']} | "
          f"avg (company,year) pool: {out['coverage']['avg_company_year_pool']}")
    print(f"{'regime':<38}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'recall':>9}")
    rc = out["pool_recall"]
    for key, label, rec in [
        ("A_corpus_bm25", "A  corpus BM25", 1.0),
        ("A_corpus_bm25_company_masked", "A' corpus BM25 (company masked)", 1.0),
        ("B_companypool_bm25", "B  company-pool + global BM25", rc["B"]),
        ("C_companypool_loclex", "C  company-pool + loclex", rc["C"]),
        ("D_within_company_year_loclex", "D  within (company,year) + loclex", rc["D"]),
    ]:
        m = out["regimes"][key]
        print(f"{label:<38}{m['MRR@3']:>8}{m['R@1']:>8}{m['R@3']:>8}{rec:>9.4f}")
    print(f"\nArtifact magnitude (C − A'): "
          f"{out['regimes']['C_companypool_loclex']['MRR@3'] - out['regimes']['A_corpus_bm25_company_masked']['MRR@3']:+.4f}")
    print(f"Saved → {Path(args.out) / (args.dataset.lower() + '.json')}")


if __name__ == "__main__":
    main()
