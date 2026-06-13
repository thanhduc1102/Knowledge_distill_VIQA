#!/usr/bin/env python3
"""FULL-test-set retrieval evaluation + ablation on T²-RAGBench (all 3 datasets).

Efficient: corpus and ALL queries are embedded ONCE (batched) and cached; the four
ablation arms reuse those embeddings (no re-encoding). Saves, per dataset, under
``outputs/ledger_full/<dataset>/``:
  - ablation.json / ablation.md        (dense / +GSR-constraint / +CACL-entity / FULL)
  - retrieval_metrics.json             (FULL arm)
  - retrieval_top3.jsonl               (FULL arm, with table + evidence block for the generator)

Ablation arms (all share text embeddings):
  dense            : s = cos_text
  +GSR_constraint  : s = cos_text + γ·CS_rowmajor           (GSR constraint signal)
  +CACL_entity     : s = cos_text + β·cos(e_Q,e_D)          (trained metadata embedding)
  FULL             : metadata-aware candidates + cos_text + β·entity + γ·CS

Usage:
  python scripts/full_eval.py --dataset finqa --device cuda:0
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
    from datasets import load_dataset, DatasetDict
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from gsr_cacl.entity.train import train_entity_embedder
    from gsr_cacl.scoring.constraint_score import compute_row_major_constraint_score
    from gsr_cacl.ledger import extract_ledger, build_evidence_block

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--year-window", type=int, default=1)
    ap.add_argument("--fact-top-n", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfgname, split = CFG[args.dataset]
    out_dir = Path(args.out or f"outputs/ledger_full/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- load QA split + full corpus (carry table + gold) ----
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

    queries = [(f"{r.get('company_name')}: {r['question']}" if r.get("company_name") else str(r["question"]))
               for _, r in qa.iterrows()]
    qmetas = [meta(r) for _, r in qa.iterrows()]
    gt = qa["context_id"].astype(str).tolist()
    golds = [[r.get("program_answer"), r.get("original_answer")] for _, r in qa.iterrows()]
    Q = len(queries)
    print(f"[{args.dataset}] queries={Q} corpus={N}  ({time.time()-t0:.1f}s)")

    # ---- embed corpus + queries ONCE (batched) ----
    st = SentenceTransformer(args.embed_model, device=args.device)
    doc_emb = st.encode(ctx, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype("float32")
    q_emb = st.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False).astype("float32")
    print(f"[{args.dataset}] embedded ({time.time()-t0:.1f}s)")

    # ---- entity embedder (CACL metadata) ----
    ent = train_entity_embedder(cmetas, epochs=12, device=args.device)
    with torch.no_grad():
        doc_ent = ent.encode(cmetas).cpu().numpy().astype("float32")          # [N,de]
        q_ent = ent.encode(qmetas).cpu().numpy().astype("float32")            # [Q,de]

    # ---- per-doc row-major constraint score ----
    cs = np.array([compute_row_major_constraint_score(t).constraint_score for t in tables], dtype="float32")

    # ---- metadata index ----
    comp2idx: dict[str, list[int]] = {}
    dyears = []
    for i, m in enumerate(cmetas):
        comp2idx.setdefault(m["company_name"].lower().strip(), []).append(i)
        try:
            dyears.append(int(float(m["report_year"])))
        except (ValueError, TypeError):
            dyears.append(None)

    import faiss
    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(doc_emb)
    D, I = index.search(q_emb, args.cand)   # dense candidates for all queries at once

    id_to_pos = {cid: i for i, cid in enumerate(cids)}

    def cand_set(qi, use_meta):
        s = set(int(x) for x in I[qi] if x >= 0)
        if use_meta:
            m = qmetas[qi]; comp = m["company_name"].lower().strip()
            try:
                qy = int(float(m["report_year"]))
            except (ValueError, TypeError):
                qy = None
            for j in comp2idx.get(comp, []):
                if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= args.year_window:
                    s.add(j)
        return list(s)

    def run_arm(beta, gamma, use_meta, want_rank=False):
        mrr = r1 = r3 = r5 = ndcg = 0.0
        import math
        ranks_out = []
        for qi in range(Q):
            cidx = cand_set(qi, use_meta)
            if not cidx:
                ranks_out.append([]); continue
            cidx_arr = np.array(cidx)
            s_text = doc_emb[cidx_arr] @ q_emb[qi]
            score = s_text.copy()
            if beta:
                score = score + beta * (doc_ent[cidx_arr] @ q_ent[qi])
            if gamma:
                score = score + gamma * cs[cidx_arr]
            order = np.argsort(-score)
            ranked = [cidx[o] for o in order[:max(5, args.top_k)]]
            ranks_out.append(ranked)
            gpos = id_to_pos.get(gt[qi], -1)
            for rank, p in enumerate(ranked[:3], 1):
                if p == gpos:
                    mrr += 1.0 / rank; ndcg += 1.0 / math.log2(rank + 1); break
            top = ranked
            if gpos in top[:1]: r1 += 1
            if gpos in top[:3]: r3 += 1
            if gpos in top[:5]: r5 += 1
        m = {"MRR@3": mrr / Q, "Recall@1": r1 / Q, "Recall@3": r3 / Q, "Recall@5": r5 / Q, "NDCG@3": ndcg / Q}
        return (m, ranks_out) if want_rank else m

    # NOTE: the constraint score is a VERIFIER signal (generation), not a retrieval signal —
    # adding γ·CS to the retrieval score hurts MRR (shown as a labeled diagnostic arm below).
    # The recommended retrieval config (FULL) therefore uses entity + metadata-aware
    # candidates with γ=0.
    arms = {
        "dense": run_arm(0.0, 0.0, False),
        "dense+constraint (diagnostic; not used for retrieval)": run_arm(0.0, args.gamma, False),
        "dense+CACL_entity (rerank)": run_arm(args.beta, 0.0, False),
    }
    full_m, full_ranks = run_arm(args.beta, 0.0, True, want_rank=True)
    arms["FULL (entity+metadata-filter, recommended)"] = full_m
    print(f"[{args.dataset}] arms done ({time.time()-t0:.1f}s)")

    payload = {"dataset": cfgname, "split": split, "n_queries": Q, "n_corpus": N,
               "embed_model": args.embed_model, "arms": arms}
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(full_m, indent=2))
    lines = [f"# FULL retrieval ablation — {cfgname} (n={Q}, corpus={N})", "",
             "| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |", "|---|---|---|---|---|---|"]
    for k, m in arms.items():
        lines.append(f"| {k} | {m['MRR@3']:.3f} | {m['Recall@1']:.3f} | {m['Recall@3']:.3f} "
                     f"| {m['Recall@5']:.3f} | {m['NDCG@3']:.3f} |")
    (out_dir / "ablation.md").write_text("\n".join(lines))

    # ---- save FULL retrieval outputs for the generator ----
    with open(out_dir / "retrieval_top3.jsonl", "w") as f:
        for qi in range(Q):
            top = full_ranks[qi][: args.top_k]
            ledgers = [extract_ledger(table_md=tables[p], context=ctx[p], doc_id=cids[p], meta=cmetas[p])
                       for p in top]
            evidence = build_evidence_block(queries[qi], ledgers, top_n=args.fact_top_n)
            f.write(json.dumps({
                "query": queries[qi], "query_meta": qmetas[qi],
                "ground_truth_id": gt[qi], "gold": golds[qi],
                "retrieved": [{"id": cids[p], "meta": cmetas[p], "table": tables[p],
                               "page_content": ctx[p]} for p in top],
                "evidence_block": evidence,
            }) + "\n")

    print("\n".join(lines))
    print(f"[{args.dataset}] DONE in {time.time()-t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
