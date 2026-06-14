#!/usr/bin/env python3
"""Integrated retrieval evaluation (E1/E2 entity + C3 concept-coverage structural signal).

Shares one text-embedding pass across all arms. Saves outputs/ledger_full2/<dataset>/
ablation.{json,md} and retrieval_top3.jsonl (FULL+C3 arm, with evidence block).

Arms:
  dense                         s = cos_text
  +entity                       s = cos_text + β·cos(e_Q,e_D)            (rerank, no filter)
  FULL (entity+meta-filter)     metadata-aware candidates + cos_text + β·entity   [current best]
  FULL + C3 (concept-coverage)  ... + δ·concept_coverage(Q,D)            [does structure help?]

Also prints coverage diagnostics: %docs with ≥1 canonical concept, %queries with ≥1 concept.

Usage:  python scripts/full_eval2.py --dataset finqa --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CFG = {"finqa": ("FinQA", "test"), "convfinqa": ("ConvFinQA", "turn_0"), "tatqa": ("TAT-DQA", "test")}


def main():
    import numpy as np
    import torch
    import faiss
    from datasets import load_dataset, DatasetDict
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from gsr_cacl.entity.train import train_entity_embedder
    from gsr_cacl.ontology import normalize_company, company_match
    from gsr_cacl.ledger import extract_ledger, build_evidence_block
    from gsr_cacl.scoring.concept_coverage import (
        query_concepts, query_periods, doc_periods_from_ledger, concept_coverage_score)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--year-window", type=int, default=1)
    ap.add_argument("--entity-epochs", type=int, default=12)
    ap.add_argument("--fact-top-n", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfgname, split = CFG[args.dataset]
    out_dir = Path(args.out or f"outputs/ledger_full2/{args.dataset}"); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    qa = load_dataset("G4KMU/t2-ragbench", cfgname, split=split).to_pandas()
    full = load_dataset("G4KMU/t2-ragbench", cfgname)
    parts = [full[s].to_pandas() for s in full.keys()] if isinstance(full, DatasetDict) else [full.to_pandas()]
    corpus_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="context_id").reset_index(drop=True)

    def meta(r):
        return {k: str(r.get(k, "") or "") for k in
                ("company_name", "report_year", "company_sector", "company_industry", "company_symbol")}

    cids = corpus_df["context_id"].astype(str).tolist()
    ctx = corpus_df["context"].astype(str).tolist()
    tables = corpus_df["table"].astype(str).tolist() if "table" in corpus_df.columns else ["" for _ in cids]
    cmetas = [meta(r) for _, r in corpus_df.iterrows()]
    N = len(cids)
    id_to_pos = {c: i for i, c in enumerate(cids)}

    queries = [(f"{r.get('company_name')}: {r['question']}" if r.get("company_name") else str(r["question"]))
               for _, r in qa.iterrows()]
    raw_q = qa["question"].astype(str).tolist()
    qmetas = [meta(r) for _, r in qa.iterrows()]
    gt = qa["context_id"].astype(str).tolist()
    golds = [[r.get("program_answer"), r.get("original_answer")] for _, r in qa.iterrows()]
    Q = len(queries)
    print(f"[{args.dataset}] queries={Q} corpus={N} ({time.time()-t0:.0f}s)")

    st = SentenceTransformer(args.embed_model, device=args.device)
    doc_emb = st.encode(ctx, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    q_emb = st.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    ent = train_entity_embedder(cmetas, epochs=args.entity_epochs, device=args.device, embedder="ontology")
    with torch.no_grad():
        d_ent = ent.encode(cmetas).cpu().numpy().astype("float32")
        q_ent = ent.encode(qmetas).cpu().numpy().astype("float32")

    # ---- per-doc ledger concept/period sets (C2/C3) ----
    doc_concepts: list[set] = []
    doc_periods: list[set] = []
    n_doc_with_concept = 0
    for i in range(N):
        led = extract_ledger(table_md=tables[i], context=ctx[i], doc_id=cids[i], meta=cmetas[i])
        cs = led.concept_set()
        doc_concepts.append(cs)
        doc_periods.append(doc_periods_from_ledger(led))
        if cs:
            n_doc_with_concept += 1
    # ---- per-query concept/period ----
    q_concepts = [query_concepts(q) for q in raw_q]
    q_periods = [query_periods(q) for q in raw_q]
    n_q_with_concept = sum(1 for c in q_concepts if c)
    diag = {"pct_docs_with_canonical_concept": round(100 * n_doc_with_concept / N, 1),
            "pct_queries_with_canonical_concept": round(100 * n_q_with_concept / Q, 1)}
    print(f"[{args.dataset}] coverage diag: {diag} ({time.time()-t0:.0f}s)")

    # ---- metadata index (alias-aware) ----
    dyears = []
    for m in cmetas:
        try:
            dyears.append(int(float(m["report_year"])))
        except (ValueError, TypeError):
            dyears.append(None)
    alias_idx = {}
    for i, m in enumerate(cmetas):
        alias_idx.setdefault(normalize_company(m["company_name"]), []).append(i)
    alias_keys = list(alias_idx.keys())

    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(doc_emb)
    _, I = index.search(q_emb, args.cand)

    def company_hits(qi):
        key = normalize_company(qmetas[qi]["company_name"])
        hits = list(alias_idx.get(key, []))
        if hits:
            return hits
        for ck in alias_keys:
            if ck and company_match(key, ck):
                hits.extend(alias_idx[ck])
        return hits

    def cand_set(qi, use_meta):
        s = set(int(x) for x in I[qi] if x >= 0)
        if use_meta:
            try:
                qy = int(float(qmetas[qi]["report_year"]))
            except (ValueError, TypeError):
                qy = None
            for j in company_hits(qi):
                if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= args.year_window:
                    s.add(j)
        return list(s)

    def run_arm(beta, use_meta, delta, want_rank=False):
        mrr = r1 = r3 = r5 = ndcg = 0.0
        ranks_out = []
        for qi in range(Q):
            cidx = cand_set(qi, use_meta)
            if not cidx:
                ranks_out.append([]); continue
            ca = np.array(cidx)
            score = (doc_emb[ca] @ q_emb[qi]).astype("float64")
            if beta:
                score = score + beta * (d_ent[ca] @ q_ent[qi])
            if delta:
                qc, qp = q_concepts[qi], q_periods[qi]
                cov = np.array([concept_coverage_score(qc, qp, doc_concepts[j], doc_periods[j])
                                for j in cidx])
                score = score + delta * cov
            order = np.argsort(-score)
            ranked = [cidx[o] for o in order[:5]]
            ranks_out.append(ranked)
            gpos = id_to_pos.get(gt[qi], -1)
            for rank, p in enumerate(ranked[:3], 1):
                if p == gpos:
                    mrr += 1.0 / rank; ndcg += 1.0 / math.log2(rank + 1); break
            if gpos in ranked[:1]: r1 += 1
            if gpos in ranked[:3]: r3 += 1
            if gpos in ranked[:5]: r5 += 1
        m = {"MRR@3": mrr / Q, "Recall@1": r1 / Q, "Recall@3": r3 / Q, "Recall@5": r5 / Q, "NDCG@3": ndcg / Q}
        return (m, ranks_out) if want_rank else m

    arms = {
        "dense": run_arm(0.0, False, 0.0),
        "+entity (rerank)": run_arm(args.beta, False, 0.0),
        "FULL (entity+meta-filter)": run_arm(args.beta, True, 0.0),
    }
    # small sweep on the C3 weight
    best = None
    for delta in (0.1, 0.2, 0.3, 0.5):
        m = run_arm(args.beta, True, delta)
        arms[f"FULL + C3 (δ={delta})"] = m
        if best is None or m["MRR@3"] > best[1]["MRR@3"]:
            best = (delta, m)
    full_c3, full_ranks = run_arm(args.beta, True, best[0], want_rank=True)

    payload = {"dataset": cfgname, "split": split, "n_queries": Q, "n_corpus": N,
               "embed_model": args.embed_model, "beta": args.beta, "best_C3_delta": best[0],
               "coverage_diag": diag, "arms": arms}
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2))
    lines = [f"# Integrated retrieval ablation — {cfgname} (n={Q}, corpus={N})",
             f"_coverage: {diag}_", "",
             "| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |", "|---|---|---|---|---|---|"]
    for k, m in arms.items():
        lines.append(f"| {k} | {m['MRR@3']:.3f} | {m['Recall@1']:.3f} | {m['Recall@3']:.3f} "
                     f"| {m['Recall@5']:.3f} | {m['NDCG@3']:.3f} |")
    (out_dir / "ablation.md").write_text("\n".join(lines))

    with open(out_dir / "retrieval_top3.jsonl", "w") as f:
        for qi in range(Q):
            top = full_ranks[qi][:3]
            ledgers = [extract_ledger(table_md=tables[p], context=ctx[p], doc_id=cids[p], meta=cmetas[p])
                       for p in top]
            evidence = build_evidence_block(queries[qi], ledgers, top_n=args.fact_top_n)
            f.write(json.dumps({"query": queries[qi], "query_meta": qmetas[qi],
                                "ground_truth_id": gt[qi], "gold": golds[qi],
                                "retrieved": [{"id": cids[p], "meta": cmetas[p], "table": tables[p],
                                               "page_content": ctx[p]} for p in top],
                                "evidence_block": evidence}) + "\n")
    print("\n".join(lines))
    print(f"[{args.dataset}] DONE in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
