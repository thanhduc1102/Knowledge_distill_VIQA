#!/usr/bin/env python3
"""Entity-channel ablation: hash vs GICS/alias ontology embedder (contributions E1+E2).

Isolates the *entity* contribution on the full T²-RAGBench test set. Text embeddings are
computed ONCE and shared across all arms (no re-encoding); only the entity embedder and the
company-matching rule change. Saves under ``outputs/entity_ablation/<dataset>/``:
  - ablation.json / ablation.md

Arms (text cosine is always present):
  dense                         : s = cos_text                              (baseline)
  + hash-entity (rerank)        : s = cos_text + β·cos(e_Q,e_D)   [HashMetadataEmbedder]
  + ontology-entity (rerank)    : s = cos_text + β·cos(e_Q,e_D)   [OntologyMetadataEmbedder]
  FULL hash (exact filter)      : metadata-aware candidates (exact company) + hash entity
  FULL ontology (alias filter)  : metadata-aware candidates (alias/GICS)   + ontology entity

Usage:
  python scripts/entity_ablation.py --dataset finqa --device cuda:0
"""
from __future__ import annotations

import argparse
import json
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--year-window", type=int, default=1)
    ap.add_argument("--entity-epochs", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfgname, split = CFG[args.dataset]
    out_dir = Path(args.out or f"outputs/entity_ablation/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
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
    cmetas = [meta(r) for _, r in corpus_df.iterrows()]
    N = len(cids)
    id_to_pos = {c: i for i, c in enumerate(cids)}

    queries = [(f"{r.get('company_name')}: {r['question']}" if r.get("company_name") else str(r["question"]))
               for _, r in qa.iterrows()]
    qmetas = [meta(r) for _, r in qa.iterrows()]
    gt = qa["context_id"].astype(str).tolist()
    Q = len(queries)
    print(f"[{args.dataset}] queries={Q} corpus={N} ({time.time()-t0:.0f}s)")

    st = SentenceTransformer(args.embed_model, device=args.device)
    doc_emb = st.encode(ctx, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    q_emb = st.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    # entity embedders: baseline hash vs ontology (E1+E2)
    ent_hash = train_entity_embedder(cmetas, epochs=args.entity_epochs, device=args.device, embedder="hash")
    ent_onto = train_entity_embedder(cmetas, epochs=args.entity_epochs, device=args.device, embedder="ontology")
    with torch.no_grad():
        d_hash = ent_hash.encode(cmetas).cpu().numpy().astype("float32")
        q_hash = ent_hash.encode(qmetas).cpu().numpy().astype("float32")
        d_onto = ent_onto.encode(cmetas).cpu().numpy().astype("float32")
        q_onto = ent_onto.encode(qmetas).cpu().numpy().astype("float32")

    # metadata indices: exact vs alias-canonical company key
    dyears = []
    for m in cmetas:
        try:
            dyears.append(int(float(m["report_year"])))
        except (ValueError, TypeError):
            dyears.append(None)
    exact_idx, alias_idx = {}, {}
    for i, m in enumerate(cmetas):
        raw = m["company_name"].lower().strip()
        exact_idx.setdefault(raw, []).append(i)
        alias_idx.setdefault(normalize_company(raw), []).append(i)
    alias_keys = list(alias_idx.keys())

    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(doc_emb)
    _, I = index.search(q_emb, args.cand)

    def company_hits(qi, alias):
        comp = qmetas[qi]["company_name"]
        if not alias:
            return exact_idx.get(comp.lower().strip(), [])
        key = normalize_company(comp)
        hits = list(alias_idx.get(key, []))
        if hits:
            return hits
        for ck in alias_keys:
            if ck and company_match(key, ck):
                hits.extend(alias_idx[ck])
        return hits

    def cand_set(qi, use_meta, alias):
        s = set(int(x) for x in I[qi] if x >= 0)
        if use_meta:
            try:
                qy = int(float(qmetas[qi]["report_year"]))
            except (ValueError, TypeError):
                qy = None
            for j in company_hits(qi, alias):
                if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= args.year_window:
                    s.add(j)
        return list(s)

    def run_arm(beta, d_ent, q_ent, use_meta, alias):
        import math
        mrr = r1 = r3 = r5 = ndcg = 0.0
        for qi in range(Q):
            cidx = cand_set(qi, use_meta, alias)
            if not cidx:
                continue
            ca = np.array(cidx)
            score = doc_emb[ca] @ q_emb[qi]
            if beta and d_ent is not None:
                score = score + beta * (d_ent[ca] @ q_ent[qi])
            order = np.argsort(-score)
            ranked = [cidx[o] for o in order[:5]]
            gpos = id_to_pos.get(gt[qi], -1)
            for rank, p in enumerate(ranked[:3], 1):
                if p == gpos:
                    mrr += 1.0 / rank; ndcg += 1.0 / math.log2(rank + 1); break
            if gpos in ranked[:1]: r1 += 1
            if gpos in ranked[:3]: r3 += 1
            if gpos in ranked[:5]: r5 += 1
        return {"MRR@3": mrr / Q, "Recall@1": r1 / Q, "Recall@3": r3 / Q,
                "Recall@5": r5 / Q, "NDCG@3": ndcg / Q}

    arms = {
        "dense": run_arm(0.0, None, None, False, False),
        "dense + hash-entity (rerank)": run_arm(args.beta, d_hash, q_hash, False, False),
        "dense + ontology-entity (rerank)": run_arm(args.beta, d_onto, q_onto, False, False),
        "FULL hash (exact filter)": run_arm(args.beta, d_hash, q_hash, True, False),
        "FULL ontology (alias filter, E1+E2)": run_arm(args.beta, d_onto, q_onto, True, True),
    }
    payload = {"dataset": cfgname, "split": split, "n_queries": Q, "n_corpus": N,
               "embed_model": args.embed_model, "beta": args.beta, "arms": arms}
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2))
    lines = [f"# Entity-channel ablation — {cfgname} (n={Q}, corpus={N})", "",
             "| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |", "|---|---|---|---|---|---|"]
    for k, m in arms.items():
        lines.append(f"| {k} | {m['MRR@3']:.3f} | {m['Recall@1']:.3f} | {m['Recall@3']:.3f} "
                     f"| {m['Recall@5']:.3f} | {m['NDCG@3']:.3f} |")
    (out_dir / "ablation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"[{args.dataset}] DONE in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
