"""Export a metadata-aware retrieval_top3.jsonl (company+year provided pool + BM25 within).

TAT-DQA's dominant NM failure is RETRIEVAL (31% gold-doc miss, error_decomposition.py). The
final_retrieval/tatqa top3 has only R@3 0.62; the metadata-aware ranking reaches R@3 0.72.
This script writes a stronger top3 (same schema as final_retrieval) so the generator can be
re-run on it to convert the retrieval gain into Number-Match gain.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.aliases import normalize_company
from gsr_cacl.retrieval.normalize import concept_sentinels

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}
DSKEY = {"FinQA": "finqa", "ConvFinQA": "convfinqa", "TAT-DQA": "tatqa"}
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "is", "are", "what", "how",
         "much", "many", "did", "year", "during", "between", "change", "total"}


def _toks(t):
    return [x for x in _TOK.findall((t or "").lower()) if x not in _STOP and len(x) > 1]


def _dtoks(t):
    return _toks(t) + concept_sentinels(t)


def main():
    from rank_bm25 import BM25Okapi
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="TAT-DQA", choices=list(SPLITS))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="outputs/meta_retrieval/{ds}/retrieval_top3.jsonl")
    args = ap.parse_args()
    data = load_t2ragbench_split(args.dataset, split=SPLITS[args.dataset])
    corpus, gts, qmetas = data.corpus, data.ground_truth_ids, data.meta_data
    texts = [d.page_content for d in corpus]
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]
    doc_co = [normalize_company(str((m or {}).get("company_name", "")).strip()) for m in doc_metas]
    doc_yr = []
    for m in doc_metas:
        y = (m or {}).get("report_year")
        try:
            doc_yr.append(int(float(str(y))) if y else None)
        except (ValueError, TypeError):
            doc_yr.append(None)
    bm = BM25Okapi([_dtoks(t) for t in texts])
    raw_q = []
    for q, m in zip(data.queries, qmetas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)

    outp = Path(args.out.format(ds=DSKEY[args.dataset]))
    outp.parent.mkdir(parents=True, exist_ok=True)
    hit = 0
    with outp.open("w") as fh:
        for qi in range(len(raw_q)):
            s = np.asarray(bm.get_scores(_dtoks(raw_q[qi])))
            co = normalize_company(str((qmetas[qi] or {}).get("company_name", "")).strip()) or None
            yrs = set(extract_years(raw_q[qi]))
            ym = (qmetas[qi] or {}).get("report_year")
            if ym:
                try:
                    yrs.add(int(float(str(ym))))
                except (ValueError, TypeError):
                    pass
            if co:
                pool = [d for d in range(len(corpus)) if doc_co[d] == co]
                pool = sorted(pool, key=lambda d: ((doc_yr[d] in yrs), s[d]), reverse=True)
            else:
                pool = list(np.argsort(-s))
            top = pool[: args.topk]
            gid = str(gts[qi])
            if gid in [str(corpus[d].id) for d in top]:
                hit += 1
            retrieved = [{"rank": r + 1, "context_id": str(corpus[d].id), "id": str(corpus[d].id),
                          "score": float(s[d]), "meta": doc_metas[d],
                          "table": texts[d]} for r, d in enumerate(top)]
            fh.write(json.dumps({
                "query_id": qi, "query": data.queries[qi], "raw_question": raw_q[qi],
                "query_meta": qmetas[qi], "ground_truth_id": gid,
                "gold": None,  # joined later by question from gemini_gen (which carries gold)
                "retrieved": retrieved,
            }) + "\n")
    print(f"{args.dataset}: wrote {outp}  R@{args.topk}={hit/len(raw_q):.4f} (n={len(raw_q)})")


if __name__ == "__main__":
    main()
