#!/usr/bin/env python3
"""Metadata-aware BM25 — leaderboard-faithful SOTA replication for T2-RAGBench.

The T2-RAGBench #1 system is "GPT-5.4 + Metadata-aware BM25". T2-RAGBench is a retrieval
benchmark where each query is accompanied by the target filing's company/year as METADATA;
exploiting it is the sanctioned, standard setting (not leakage). This script reproduces and
analyses that approach transparently under THREE settings, so the paper can report both the
SOTA number and the honest content-only number with no ambiguity:

  1. content_only       : pure BM25 (+abbr sentinel). No metadata. The hard setting.
  2. meta_question      : company/year extracted from the QUESTION TEXT only (legitimate
                          even under the strict honest contract). Filter+rank by metadata.
  3. meta_provided      : company/year taken from the provided query metadata (the standard
                          T2-RAGBench / leaderboard setting). Upper bound of metadata value.

Also reports metadata RECOVERABILITY (how often company/year are present in the question),
which is the legitimacy argument for setting 2/3.

Ranking (settings 2/3): score(d) = BM25(d) + BIG*[company match] + MED*[year match],
so company-matching docs float to the top, year breaks ties within company — exactly the
"metadata-aware BM25" recipe. Falls back to pure BM25 when no company is detected.

CPU only. Full test sets. Honest MRR@3 / Recall@{1,3,5}.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.aliases import normalize_company
from gsr_cacl.retrieval.self_query import CompanyIndex
from gsr_cacl.retrieval.normalize import concept_sentinels

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "was", "is", "are", "what",
         "how", "much", "many", "did", "does", "do", "as", "by", "on", "at", "year", "fiscal",
         "report", "reported", "company", "value", "during", "between", "change", "total"}
BIG, MED = 1e6, 1e3


def _toks(t):
    return [x for x in _TOK.findall((t or "").lower()) if x not in _STOP and len(x) > 1]


def _doc_toks(t):
    return _toks(t) + concept_sentinels(t)


def _metrics(ranks):
    ranks = np.array(ranks, dtype=float)
    valid = ranks > 0
    mrr3 = np.mean([1.0 / r if (r > 0 and r <= 3) else 0.0 for r in ranks])
    return {"MRR@3": round(float(mrr3), 4),
            "R@1": round(float(np.mean((ranks == 1))), 4),
            "R@3": round(float(np.mean((ranks >= 1) & (ranks <= 3))), 4),
            "R@5": round(float(np.mean((ranks >= 1) & (ranks <= 5))), 4),
            "found_rate": round(float(np.mean(valid)), 4)}


def _rank_of_gold(scores, gold_idx):
    if gold_idx < 0:
        return -1
    order = np.argsort(-scores, kind="stable")
    pos = np.where(order == gold_idx)[0]
    return int(pos[0] + 1) if len(pos) else -1


def run(ds):
    from rank_bm25 import BM25Okapi
    data = load_t2ragbench_split(ds, split=SPLITS[ds])
    corpus, gts, qmetas = data.corpus, data.ground_truth_ids, data.meta_data
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]
    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    n = len(corpus)

    # strip the "{company}: " prefix to get the genuine question text
    raw_q = []
    for q, m in zip(data.queries, qmetas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    bm25 = BM25Okapi([_doc_toks(d.page_content) for d in corpus])
    comp_index = CompanyIndex(doc_metas)
    doc_company = [normalize_company(str((m or {}).get("company_name", "")).strip()) for m in doc_metas]
    doc_sector = [str((m or {}).get("company_sector", "")).strip().lower() for m in doc_metas]
    doc_year = []
    for m in doc_metas:
        yr = (m or {}).get("report_year") or (m or {}).get("year")
        try:
            doc_year.append(int(float(str(yr))) if yr else None)
        except (ValueError, TypeError):
            doc_year.append(None)

    # metadata recoverability from the question
    n_co_q = n_yr_q = n_co_gold = 0
    q_company_detected, q_company_provided, q_years, q_sector = [], [], [], []
    for q, m in zip(raw_q, qmetas):
        det = comp_index.detect(q)
        prov = normalize_company(str((m or {}).get("company_name", "")).strip()) or None
        yrs = set(extract_years(q))
        q_company_detected.append(det)
        q_company_provided.append(prov)
        q_years.append(yrs)
        q_sector.append(str((m or {}).get("company_sector", "")).strip().lower())
        n_co_q += int(bool(det)); n_yr_q += int(bool(yrs)); n_co_gold += int(bool(prov))

    SECT = 5.0  # small sector prior (<< BIG company, << MED year): only a tie-breaker

    def eval_setting(use_meta, company_src, use_sector=False):
        ranks = []
        for qi in range(len(raw_q)):
            s = np.asarray(bm25.get_scores(_doc_toks(raw_q[qi])), dtype=np.float64)
            if use_meta:
                co = company_src[qi]
                yrs = q_years[qi]
                if use_sector and q_sector[qi]:
                    smask = np.array([1.0 if doc_sector[d] == q_sector[qi] else 0.0 for d in range(n)])
                    s = s + SECT * smask  # coarse sector prior over all docs
                if co:
                    cmask = np.array([1.0 if doc_company[d] == co else 0.0 for d in range(n)])
                    s = s + BIG * cmask
                    if yrs:
                        ymask = np.array([1.0 if (doc_year[d] in yrs) else 0.0 for d in range(n)])
                        s = s + MED * ymask * cmask  # year bonus only within company
            ranks.append(_rank_of_gold(s, gold_idx[qi]))
        return _metrics(ranks)

    return {
        "dataset": ds, "n_queries": len(raw_q), "n_corpus": n,
        "recoverability": {
            "company_in_question": round(n_co_q / len(raw_q), 4),
            "year_in_question": round(n_yr_q / len(raw_q), 4),
            "company_provided_meta": round(n_co_gold / len(raw_q), 4),
        },
        "content_only": eval_setting(False, None),
        "meta_question": eval_setting(True, q_company_detected),
        "meta_provided": eval_setting(True, q_company_provided),
        "meta_provided_3field": eval_setting(True, q_company_provided, use_sector=True),
        "sector_only": eval_setting(True, [None] * len(raw_q), use_sector=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["FinQA", "ConvFinQA", "TAT-DQA"])
    ap.add_argument("--out", default="outputs/research/metadata_bm25/report.json")
    args = ap.parse_args()
    allr = {}
    for ds in args.datasets:
        r = run(ds); allr[ds] = r
        rec = r["recoverability"]
        print(f"\n=== {ds} (n={r['n_queries']}, corpus={r['n_corpus']}) ===")
        print(f"  recoverability: company_in_Q={rec['company_in_question']} "
              f"year_in_Q={rec['year_in_question']} company_provided={rec['company_provided_meta']}")
        print(f"  {'setting':20s} {'MRR@3':>7s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s}")
        for s in ("content_only", "sector_only", "meta_question", "meta_provided", "meta_provided_3field"):
            g = r[s]
            print(f"  {s:20s} {g['MRR@3']:7.4f} {g['R@1']:6.3f} {g['R@3']:6.3f} {g['R@5']:6.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
