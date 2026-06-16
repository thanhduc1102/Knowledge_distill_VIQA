#!/usr/bin/env python3
"""Phase-A honest retrieval — abbreviation unification + company self-query.

Backbone is the validated stage-1 recipe (BM25 + gated period + gated cell-match),
re-implemented BM25-only (no dense; the winning stage-1 arms all set w_dense=0, so
e5 is never needed here — this runs in seconds with no model download).

Two NEW honest channels are layered on top and ablated separately:

  ABBR     concept sentinels appended to BM25 tokens (query AND docs) so abbreviation
           vs full-form ("EBITDA" / "earnings before interest...") match — EDA #5.
  COMPANY  alias-robust company detected FROM THE QUESTION; soft, gated boost to docs
           of that company. No-op when the question names no company — EDA #4 + the
           ~20-36% no-company fairness case.

All signals are bounded additive boosts inside the BM25 candidate pool (never add a
new doc) and gated to stay discriminative. Honest: years/company/concepts are read
from the QUESTION only; the injected ``company:`` prefix is stripped first.

Usage:  PYTHONPATH=src python scripts/phaseA_retrieval.py --sample 0
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
from gsr_cacl.retrieval.normalize import concept_sentinels
from gsr_cacl.retrieval.self_query import CompanyIndex

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
    """(periods, [(row-label tokens, period)]) for cell-level match."""
    tmd = extract_table(page_content)
    if not tmd:
        return set(extract_years(page_content)), []
    L = extract_ledger_from_table(tmd)
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
    return yrs, cells


def cell_match(q_content: set, q_years: set, cells, doc_years: set) -> float:
    if not q_content or not cells:
        return 0.0
    best = 0.0
    for ctoks, p in cells:
        ov = jaccard(q_content, ctoks)
        if ov <= 0:
            continue
        pf = 1.0 if (p is not None and p in q_years) else (0.6 if (q_years & doc_years) else 0.3)
        best = max(best, ov * pf)
    return best


def ranks_from_scores(scores, cand):
    return {int(idx): r for r, idx in enumerate(np.argsort(-scores)[:cand])}


def evaluate_dataset(cfg, split, sample, cand=50, rrf_k=60):
    from rank_bm25 import BM25Okapi
    data = load_t2ragbench_split(cfg, split=split, sample_size=(sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data

    # Honest: strip the injected "company:" prefix the loader prepends.
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    # ---- BM25 backbones: plain vs abbreviation-expanded -------------------------
    base_doc_tok = [bm_toks(d.page_content) for d in corpus]
    abbr_doc_tok = [base_doc_tok[i] + concept_sentinels(corpus[i].page_content)
                    for i in range(len(corpus))]
    bm25_base = BM25Okapi(base_doc_tok)
    bm25_abbr = BM25Okapi(abbr_doc_tok)

    # ---- per-doc structure caches ----------------------------------------------
    dfacts = [doc_facts(d.page_content) for d in corpus]
    doc_years = [a for a, _ in dfacts]
    doc_cells = [b for _, b in dfacts]
    comp_index = CompanyIndex(metas)

    # ---- per-query caches -------------------------------------------------------
    q_years = [set(extract_years(q)) for q in raw_q]
    q_content = [toks(q) for q in raw_q]
    q_company = [comp_index.detect(q) for q in raw_q]           # honest, from question
    q_comp_docs = [comp_index.docs_for(k) for k in q_company]

    base_ranks = [ranks_from_scores(bm25_base.get_scores(bm_toks(q)), cand) for q in raw_q]
    abbr_ranks = [ranks_from_scores(bm25_abbr.get_scores(bm_toks(q) + concept_sentinels(q)), cand)
                  for q in raw_q]

    company_hit_rate = round(sum(1 for k in q_company if k) / max(len(q_company), 1), 3)

    def run(use_abbr, period_b, cell_b, company_b, coyear_b, gate=0.4):
        ranks = abbr_ranks if use_abbr else base_ranks
        res = []
        for i in range(len(raw_q)):
            fused = {idx: 1.0 / (rrf_k + r + 1) for idx, r in ranks[i].items()}
            qy, qct = q_years[i], q_content[i]
            keys = list(fused.keys())
            if not keys:
                res.append(RetrievalResult(query=raw_q[i], retrieved_docs=[],
                                           ground_truth_id=gts[i], meta_data=metas[i]))
                continue
            # META period (gated)
            if period_b and qy:
                ms = [idx for idx in keys if doc_years[idx] & qy]
                if ms and len(ms) / len(keys) <= gate:
                    for idx in ms:
                        fused[idx] += period_b / rrf_k
            # TABLE cell-match (graded; gated)
            if cell_b and qct:
                cm = {idx: cell_match(qct, qy, doc_cells[idx], doc_years[idx]) for idx in keys}
                hi = [idx for idx, c in cm.items() if c >= 0.34]
                if hi and len(hi) / len(keys) <= gate:
                    for idx in hi:
                        fused[idx] += cell_b * cm[idx] / rrf_k
            # COMPANY self-query, flat (gated; honest no-op when no company in question)
            if company_b and q_company[i]:
                cdocs = q_comp_docs[i]
                ms = [idx for idx in keys if idx in cdocs]
                if ms and len(ms) / len(keys) <= gate:
                    for idx in ms:
                        fused[idx] += company_b / rrf_k
            # COMPANY×YEAR joint (gated): boost only docs matching company AND year —
            # the discriminative metadata signal that disambiguates same-company-wrong-year.
            if coyear_b and q_company[i] and qy:
                cdocs = q_comp_docs[i]
                ms = [idx for idx in keys if idx in cdocs and (doc_years[idx] & qy)]
                if ms and len(ms) / len(keys) <= gate:
                    for idx in ms:
                        fused[idx] += coyear_b / rrf_k
            order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]
            res.append(RetrievalResult(query=raw_q[i], retrieved_docs=[corpus[i2] for i2, _ in order],
                                       ground_truth_id=gts[i], meta_data=metas[i]))
        return {"MRR@3": round(compute_mrr(res, 3), 4), "Recall@1": round(compute_recall(res, 1), 4),
                "Recall@3": round(compute_recall(res, 3), 4), "Recall@5": round(compute_recall(res, 5), 4),
                "NDCG@3": round(compute_ndcg(res, 3), 4)}

    arms = {
        # (use_abbr, period_b, cell_b, company_b, coyear_b)
        "stage1 (bm25+period+cell)":        (False, 0.05, 0.3, 0.0,  0.0),
        "+abbr":                            (True,  0.05, 0.3, 0.0,  0.0),
        "+abbr+coyear(0.05)":               (True,  0.05, 0.3, 0.0,  0.05),
        "+abbr+coyear(0.1)":                (True,  0.05, 0.3, 0.0,  0.10),
        "+abbr+coyear(0.2)":                (True,  0.05, 0.3, 0.0,  0.20),
        "+abbr+company_flat(0.05)":         (True,  0.05, 0.3, 0.05, 0.0),
    }
    return len(raw_q), {n: run(*w) for n, w in arms.items()}, list(arms.keys()), company_hit_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="outputs/phaseA")
    args = ap.parse_args()

    per_ds, counts, hitrate, names = {}, {}, {}, None
    for cfg, split in DATASETS:
        print(f"\n### {cfg} ({split}) ###")
        n, res, names, hr = evaluate_dataset(cfg, split, args.sample)
        per_ds[cfg], counts[cfg], hitrate[cfg] = res, n, hr
        print(f"  company-in-question rate: {hr:.1%}")
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
    Path(args.out, "phaseA.json").write_text(json.dumps(
        {"counts": counts, "company_in_question_rate": hitrate,
         "per_dataset": per_ds, "weighted_avg_mrr3": wavg, "best": best[0]}, indent=2))
    print(f"\nSaved to {args.out}/phaseA.json")


if __name__ == "__main__":
    main()
