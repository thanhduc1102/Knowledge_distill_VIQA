#!/usr/bin/env python3
"""Cross-encoder rerank on the retrieval pool (push toward leaderboard #1).

The MMER fusion ranks with cheap experts; a cross-encoder (query x doc) is a stronger but
costlier re-scorer. We test it on two pools:
  * BM25 top-N (content-only)                -> does CE rerank lift content-only retrieval?
  * provided company+year pool (metadata)    -> CE rerank WITHIN the metadata pool (the
                                                "right place to exploit" — strongest setting).
Reports MRR@3 / R@{1,3,5} for: bm25, bm25+CE, meta-pool first-doc(bm25), meta-pool+CE.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.aliases import normalize_company
from gsr_cacl.retrieval.normalize import concept_sentinels

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "is", "are", "what", "how",
         "much", "many", "did", "year", "during", "between", "change", "total"}


def _toks(t):
    return [x for x in _TOK.findall((t or "").lower()) if x not in _STOP and len(x) > 1]


def _doc_toks(t):
    return _toks(t) + concept_sentinels(t)


def _metrics(ranks):
    r = np.array(ranks, float)
    return {"MRR@3": round(float(np.mean([1.0 / x if (0 < x <= 3) else 0.0 for x in r])), 4),
            "R@1": round(float(np.mean(r == 1)), 4),
            "R@3": round(float(np.mean((r >= 1) & (r <= 3))), 4),
            "R@5": round(float(np.mean((r >= 1) & (r <= 5))), 4)}


def _rank_in(order_global, gold):
    if gold < 0:
        return -1
    pos = [i for i, d in enumerate(order_global) if d == gold]
    return pos[0] + 1 if pos else -1


def run(ds, ce, pool_n=50, max_chars=400):
    from rank_bm25 import BM25Okapi
    data = load_t2ragbench_split(ds, split=SPLITS[ds])
    corpus, gts, qmetas = data.corpus, data.ground_truth_ids, data.meta_data
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]
    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    texts = [d.page_content for d in corpus]
    raw_q = []
    for q, m in zip(data.queries, qmetas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)
    bm = BM25Okapi([_doc_toks(t) for t in texts])
    doc_co = [normalize_company(str((m or {}).get("company_name", "")).strip()) for m in doc_metas]
    doc_yr = []
    for m in doc_metas:
        y = (m or {}).get("report_year")
        try:
            doc_yr.append(int(float(str(y))) if y else None)
        except (ValueError, TypeError):
            doc_yr.append(None)
    q_co = [normalize_company(str((m or {}).get("company_name", "")).strip()) or None for m in qmetas]

    r_bm25, r_bm25ce, r_meta, r_metace = [], [], [], []
    for qi in range(len(raw_q)):
        s = np.asarray(bm.get_scores(_doc_toks(raw_q[qi])))
        topN = list(np.argsort(-s)[:pool_n])
        r_bm25.append(_rank_in(topN, gold_idx[qi]))
        # CE rerank of BM25 pool
        pairs = [(raw_q[qi], texts[d][:max_chars]) for d in topN]
        ce_s = ce.predict(pairs, batch_size=128, show_progress_bar=False)
        order = [topN[j] for j in np.argsort(-ce_s)]
        r_bm25ce.append(_rank_in(order, gold_idx[qi]))
        # metadata (provided company+year) pool
        co = q_co[qi]; yrs = set(extract_years(raw_q[qi]))
        ym = (qmetas[qi] or {}).get("report_year")
        if ym:
            try:
                yrs.add(int(float(str(ym))))
            except (ValueError, TypeError):
                pass
        if co:
            pool = [d for d in range(len(corpus)) if doc_co[d] == co]
            pool = sorted(pool, key=lambda d: (doc_yr[d] in yrs, s[d]), reverse=True)[:pool_n] or topN
        else:
            pool = topN
        # meta baseline: rank by (year-match, bm25)
        r_meta.append(_rank_in(pool, gold_idx[qi]))
        pairs = [(raw_q[qi], texts[d][:max_chars]) for d in pool]
        ce_s = ce.predict(pairs, batch_size=128, show_progress_bar=False)
        order = [pool[j] for j in np.argsort(-ce_s)]
        r_metace.append(_rank_in(order, gold_idx[qi]))
    return {"dataset": ds, "n": len(raw_q),
            "bm25": _metrics(r_bm25), "bm25+CE": _metrics(r_bm25ce),
            "meta_pool": _metrics(r_meta), "meta_pool+CE": _metrics(r_metace)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["FinQA", "ConvFinQA", "TAT-DQA"])
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--out", default="outputs/research/cross_encoder/report.json")
    args = ap.parse_args()
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.model, device=args.device, max_length=384)
    allr = {}
    for ds in args.datasets:
        t0 = time.time()
        r = run(ds, ce, pool_n=args.pool)
        allr[ds] = r
        print(f"\n=== {ds} (n={r['n']}, {time.time()-t0:.0f}s) ===")
        for k in ("bm25", "bm25+CE", "meta_pool", "meta_pool+CE"):
            g = r[k]
            print(f"  {k:14s} MRR@3={g['MRR@3']:.4f} R@1={g['R@1']:.3f} R@3={g['R@3']:.3f} R@5={g['R@5']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
