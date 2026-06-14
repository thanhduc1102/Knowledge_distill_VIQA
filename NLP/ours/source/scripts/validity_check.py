#!/usr/bin/env python3
"""Validity / leakage diagnostics for the metadata retrieval channel (Experiment #0).

Answers three questions that decide whether the big metadata-driven gain (0.38→0.73 MRR@3)
is a legitimate, reportable contribution or a near-oracle artefact:

  1. (company, year) UNIQUENESS — if a (company, year) pair almost always maps to a single
     corpus document, then filtering by metadata is ≈ oracle (reviewers would reject it).
  2. METADATA-ONLY recall ceiling — recall of the gold doc using ONLY the metadata filter
     (no text), exact vs alias-aware; and how often the filtered set is a singleton == gold.
  3. RECOVERABLE-FROM-TEXT — can company / year be recovered from the *raw question text*
     (not the gold metadata field)? If yes, using metadata is legitimate (it's in the query);
     if no, the structured field is side information that won't exist at deployment.

No GPU needed. Saves outputs/validity/<dataset>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CFG = {"finqa": ("FinQA", "test"), "convfinqa": ("ConvFinQA", "turn_0"), "tatqa": ("TAT-DQA", "test")}
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def analyse(dataset: str, year_window: int = 1) -> dict:
    from datasets import load_dataset, DatasetDict
    import pandas as pd
    from gsr_cacl.ontology import normalize_company, company_match, significant_tokens

    cfgname, split = CFG[dataset]
    qa = load_dataset("G4KMU/t2-ragbench", cfgname, split=split).to_pandas()
    full = load_dataset("G4KMU/t2-ragbench", cfgname)
    parts = [full[s].to_pandas() for s in full.keys()] if isinstance(full, DatasetDict) else [full.to_pandas()]
    corpus_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="context_id").reset_index(drop=True)

    cids = corpus_df["context_id"].astype(str).tolist()
    id_to_pos = {c: i for i, c in enumerate(cids)}

    def yr(v):
        try:
            return int(float(str(v)))
        except (ValueError, TypeError):
            return None

    ccomp_raw = [str(r.get("company_name", "") or "") for _, r in corpus_df.iterrows()]
    ccomp = [normalize_company(c) for c in ccomp_raw]
    cyear = [yr(r.get("report_year")) for _, r in corpus_df.iterrows()]

    # ---- 1. (company, year) uniqueness ----
    cy_groups = Counter(zip(ccomp, cyear))
    sizes = list(cy_groups.values())
    n_docs = len(cids)
    singleton_docs = sum(s for s in sizes if s == 1)
    uniqueness = {
        "n_corpus": n_docs,
        "n_unique_companies": len(set(ccomp)),
        "n_(company,year)_groups": len(cy_groups),
        "mean_docs_per_(company,year)": round(sum(sizes) / len(sizes), 3),
        "max_docs_per_(company,year)": max(sizes),
        "pct_docs_in_singleton_(company,year)": round(100 * singleton_docs / n_docs, 1),
    }

    # company -> idx (exact + alias)
    exact_idx, alias_idx = {}, {}
    for i, (raw, can) in enumerate(zip(ccomp_raw, ccomp)):
        exact_idx.setdefault(raw.lower().strip(), []).append(i)
        alias_idx.setdefault(can, []).append(i)
    alias_keys = list(alias_idx.keys())

    # ---- 2 & 3. per query ----
    n = len(qa)
    hit_exact = hit_alias = singleton_correct = 0
    year_in_text = year_text_matches_gold = 0
    comp_in_text = 0
    cand_sizes_alias = []
    for _, r in qa.iterrows():
        gid = str(r.get("context_id", ""))
        gpos = id_to_pos.get(gid, -1)
        gcomp_raw = str(r.get("company_name", "") or "")
        gcomp = normalize_company(gcomp_raw)
        gy = yr(r.get("report_year"))
        question = str(r.get("question", ""))

        def filt(idx_map, key):
            out = []
            for j in idx_map.get(key, []):
                if gy is None or cyear[j] is None or abs(cyear[j] - gy) <= year_window:
                    out.append(j)
            return out

        ce = filt(exact_idx, gcomp_raw.lower().strip())
        ca = filt(alias_idx, gcomp)
        if not ca:  # alias fuzzy fallback
            for ck in alias_keys:
                if ck and company_match(gcomp, ck):
                    ca.extend(filt(alias_idx, ck))
        if gpos in ce:
            hit_exact += 1
        if gpos in ca:
            hit_alias += 1
            cand_sizes_alias.append(len(set(ca)))
            if len(set(ca)) == 1:
                singleton_correct += 1

        # recoverable from RAW question text
        yrs_in_q = [int(m.group(0)) for m in _YEAR_RE.finditer(question)]
        if yrs_in_q:
            year_in_text += 1
            if gy is not None and any(abs(y - gy) <= 0 for y in yrs_in_q):
                year_text_matches_gold += 1
        # company tokens present in question text (≥1 significant token, len≥4)
        toks = [t for t in significant_tokens(gcomp_raw) if len(t) >= 4]
        ql = question.lower()
        if toks and sum(t in ql for t in toks) >= max(1, len(toks) // 2):
            comp_in_text += 1

    metadata_ceiling = {
        "recall_company+year_exact": round(hit_exact / n, 3),
        "recall_company+year_alias": round(hit_alias / n, 3),
        "pct_queries_metadata_singleton==gold": round(100 * singleton_correct / n, 1),
        "mean_alias_candidate_set_size": round(sum(cand_sizes_alias) / max(len(cand_sizes_alias), 1), 2),
    }
    recoverable = {
        "pct_queries_with_year_in_text": round(100 * year_in_text / n, 1),
        "pct_queries_year_in_text==gold_year": round(100 * year_text_matches_gold / n, 1),
        "pct_queries_company_in_text": round(100 * comp_in_text / n, 1),
    }
    return {"dataset": cfgname, "n_queries": n, "uniqueness": uniqueness,
            "metadata_only_ceiling": metadata_ceiling, "recoverable_from_text": recoverable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CFG))
    ap.add_argument("--year-window", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path("outputs/validity"); out_dir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        rep = analyse(ds, args.year_window)
        (out_dir / f"{ds}.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
