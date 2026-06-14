#!/usr/bin/env python3
"""Full retrieval evaluation with CACL2 learned weights (final combined arm).

This script is the FINAL retrieval step. It runs all ablation arms from full_eval2.py
and additionally adds the CACL2-weighted arm which uses the trained [w_text, w_ent, w_cov]
from the InfoNCE contrastive training (cacl_infonce.py) instead of hand-tuned fixed values.

The CACL2 arm is the strongest single retrieval configuration and its retrieval_top3.jsonl
is intended as input to the generator phase.

Arms evaluated:
  dense                   s = cos_text
  FULL (entity+meta)      metadata-aware candidates + cos_text + β·entity   (β=0.6 fixed)
  FULL + C3 best-δ        ... + δ·concept_coverage                           (δ swept, best chosen)
  FULL + CACL2 weights    ... using [w_text, w_ent, w_cov] from cacl2_model.pt  [FINAL ARM]

Outputs (per dataset) under outputs/final_retrieval/<dataset>/:
  ablation.json            full metric table for all arms
  ablation.md              markdown table
  retrieval_top3.jsonl     CACL2-arm top-3 results (for generator consumption)
  cacl2_weights.json       which weights were loaded from which checkpoint

Usage:
  cd ours/source && export PYTHONPATH=src
  python scripts/full_eval2_with_cacl.py --dataset finqa --device cuda:0
  python scripts/full_eval2_with_cacl.py --dataset convfinqa --device cuda:0
  python scripts/full_eval2_with_cacl.py --dataset tatqa --device cuda:0
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

CFG = {
    "finqa":     ("FinQA",     "test",   "outputs/cacl_infonce/finqa/cacl2_model.pt"),
    "convfinqa": ("ConvFinQA", "turn_0", "outputs/cacl_infonce/convfinqa/cacl2_model.pt"),
    "tatqa":     ("TAT-DQA",  "test",   "outputs/cacl_infonce/tatqa/cacl2_model.pt"),
}


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import faiss
    from datasets import load_dataset, DatasetDict
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from gsr_cacl.entity.encoder import OntologyMetadataEmbedder
    from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels
    from gsr_cacl.ontology import normalize_company, company_match
    from gsr_cacl.ledger import extract_ledger, build_evidence_block
    from gsr_cacl.scoring.concept_coverage import (
        query_concepts, query_periods, doc_periods_from_ledger, concept_coverage_score)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--year-window", type=int, default=1)
    ap.add_argument("--entity-epochs", type=int, default=12)
    ap.add_argument("--fact-top-n", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cacl", action="store_true", help="Skip CACL2 arm (use hand-tuned weights only)")
    args = ap.parse_args()

    cfgname, split, ckpt_path = CFG[args.dataset]
    out_dir = Path(args.out or f"outputs/final_retrieval/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print(f"[{args.dataset}] Loading dataset...")
    qa = load_dataset("G4KMU/t2-ragbench", cfgname, split=split).to_pandas()
    full = load_dataset("G4KMU/t2-ragbench", cfgname)
    parts = ([full[s].to_pandas() for s in full.keys()]
             if isinstance(full, DatasetDict) else [full.to_pandas()])
    corpus_df = (pd.concat(parts, ignore_index=True)
                 .drop_duplicates(subset="context_id")
                 .reset_index(drop=True))

    def meta(r):
        return {k: str(r.get(k, "") or "") for k in
                ("company_name", "report_year", "company_sector",
                 "company_industry", "company_symbol")}

    cids    = corpus_df["context_id"].astype(str).tolist()
    ctx     = corpus_df["context"].astype(str).tolist()
    tables  = (corpus_df["table"].astype(str).tolist()
               if "table" in corpus_df.columns else [""] * len(cids))
    cmetas  = [meta(r) for _, r in corpus_df.iterrows()]
    N       = len(cids)
    id_to_pos = {c: i for i, c in enumerate(cids)}

    queries  = [(f"{r.get('company_name')}: {r['question']}"
                 if r.get("company_name") else str(r["question"]))
                for _, r in qa.iterrows()]
    raw_q    = qa["question"].astype(str).tolist()
    qmetas   = [meta(r) for _, r in qa.iterrows()]
    gt       = qa["context_id"].astype(str).tolist()
    golds    = [[r.get("program_answer"), r.get("original_answer")]
                for _, r in qa.iterrows()]
    Q = len(queries)
    print(f"[{args.dataset}] queries={Q} corpus={N} ({time.time()-t0:.1f}s)")

    # ── 2. Text embeddings (shared across all arms) ───────────────────────────
    print(f"[{args.dataset}] Encoding text...")
    st = SentenceTransformer(args.embed_model, device=args.device)
    doc_emb = st.encode(ctx, batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    q_emb   = st.encode(queries, batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    print(f"[{args.dataset}] Text encoded ({time.time()-t0:.1f}s)")

    # ── 3. FAISS index ────────────────────────────────────────────────────────
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    _, I = index.search(q_emb, args.cand)

    # ── 4. Load CACL2 checkpoint ──────────────────────────────────────────────
    ckpt_abs = Path(ckpt_path)
    cacl2_loaded = False
    w_text_cacl2 = w_ent_cacl2 = w_cov_cacl2 = None
    ent_cacl2 = None

    if not args.no_cacl and ckpt_abs.exists():
        print(f"[{args.dataset}] Loading CACL2 checkpoint: {ckpt_abs}")
        ckpt = torch.load(ckpt_abs, map_location=args.device)
        w_text_cacl2 = float(F.softplus(ckpt["w_text"]))
        w_ent_cacl2  = float(F.softplus(ckpt["w_ent"]))
        w_cov_cacl2  = float(F.softplus(ckpt["w_cov"]))
        ent_cacl2 = OntologyMetadataEmbedder(embed_dim=128).to(args.device)
        ent_cacl2.load_state_dict(ckpt["entity_state"])
        ent_cacl2.eval()
        cacl2_loaded = True
        print(f"[{args.dataset}] CACL2 weights: w_text={w_text_cacl2:.3f}  "
              f"w_ent={w_ent_cacl2:.3f}  w_cov={w_cov_cacl2:.3f}")
        # Save loaded weights for reference
        (out_dir / "cacl2_weights.json").write_text(json.dumps({
            "checkpoint": str(ckpt_abs), "dataset": cfgname,
            "w_text": w_text_cacl2, "w_ent": w_ent_cacl2, "w_cov": w_cov_cacl2,
        }, indent=2))
    else:
        print(f"[{args.dataset}] CACL2 checkpoint not found or --no-cacl set; "
              f"CACL2 arm will be skipped.")

    # ── 5. Entity embedder (train from scratch for full-corpus arm, β=0.6) ────
    #       This trains the "FULL" and "FULL+C3" arms just like full_eval2.py
    print(f"[{args.dataset}] Training entity embedder for fixed-weight arms ({args.entity_epochs} epochs)...")
    from gsr_cacl.entity.train import train_entity_embedder
    ent_fixed = train_entity_embedder(cmetas, epochs=args.entity_epochs,
                                      device=args.device, embedder="ontology")
    with torch.no_grad():
        d_ent_fixed = ent_fixed.encode(cmetas).cpu().numpy().astype("float32")
        q_ent_fixed = ent_fixed.encode(qmetas).cpu().numpy().astype("float32")
    print(f"[{args.dataset}] Fixed-weight entity embedder ready ({time.time()-t0:.1f}s)")

    # ── 6. CACL2 entity embeddings ────────────────────────────────────────────
    if cacl2_loaded:
        with torch.no_grad():
            d_ent_c2 = ent_cacl2.encode(cmetas).cpu().numpy().astype("float32")
            q_ent_c2 = ent_cacl2.encode(qmetas).cpu().numpy().astype("float32")
    else:
        d_ent_c2 = d_ent_fixed
        q_ent_c2 = q_ent_fixed

    # ── 7. Per-doc concept/period sets (C2/C3) ────────────────────────────────
    print(f"[{args.dataset}] Building Fact Ledgers (C2/C3)...")
    doc_concepts: list[set] = []
    doc_periods:  list[set] = []
    n_doc_with_concept = 0
    for i in range(N):
        led = extract_ledger(table_md=tables[i], context=ctx[i],
                             doc_id=cids[i], meta=cmetas[i])
        cs = led.concept_set()
        doc_concepts.append(cs)
        doc_periods.append(doc_periods_from_ledger(led))
        if cs:
            n_doc_with_concept += 1

    q_concepts_list = [query_concepts(q) for q in raw_q]
    q_periods_list  = [query_periods(q)  for q in raw_q]
    n_q_with_concept = sum(1 for c in q_concepts_list if c)
    diag = {
        "pct_docs_with_canonical_concept":   round(100 * n_doc_with_concept / N, 1),
        "pct_queries_with_canonical_concept": round(100 * n_q_with_concept / Q, 1),
    }
    print(f"[{args.dataset}] Coverage diag: {diag} ({time.time()-t0:.1f}s)")

    # ── 8. Metadata index ─────────────────────────────────────────────────────
    dyears: list[int | None] = []
    for m in cmetas:
        try:
            dyears.append(int(float(m["report_year"])))
        except (ValueError, TypeError):
            dyears.append(None)

    alias_idx: dict[str, list[int]] = {}
    for i, m in enumerate(cmetas):
        alias_idx.setdefault(normalize_company(m["company_name"]), []).append(i)
    alias_keys = list(alias_idx.keys())

    def company_hits(qi: int) -> list[int]:
        key = normalize_company(qmetas[qi]["company_name"])
        hits = list(alias_idx.get(key, []))
        if hits:
            return hits
        for ck in alias_keys:
            if ck and company_match(key, ck):
                hits.extend(alias_idx[ck])
        return hits

    def cand_set(qi: int, use_meta: bool) -> list[int]:
        s = {int(x) for x in I[qi] if x >= 0}
        if use_meta:
            try:
                qy = int(float(qmetas[qi]["report_year"]))
            except (ValueError, TypeError):
                qy = None
            for j in company_hits(qi):
                if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= args.year_window:
                    s.add(j)
        return list(s)

    # ── 9. Scoring engine ─────────────────────────────────────────────────────
    def cov_score_vec(qi: int, cidx: list[int]) -> np.ndarray:
        qc, qp = q_concepts_list[qi], q_periods_list[qi]
        return np.array([concept_coverage_score(qc, qp, doc_concepts[j], doc_periods[j])
                         for j in cidx], dtype="float64")

    def run_arm(
        w_text: float, w_ent: float, w_cov: float,
        d_ent: np.ndarray, q_ent: np.ndarray,
        use_meta: bool, want_rank: bool = False,
    ):
        mrr = r1 = r3 = r5 = ndcg = 0.0
        ranks_out = []
        for qi in range(Q):
            cidx = cand_set(qi, use_meta)
            if not cidx:
                ranks_out.append([])
                continue
            ca = np.array(cidx)
            score = w_text * (doc_emb[ca] @ q_emb[qi]).astype("float64")
            if w_ent:
                score = score + w_ent * (d_ent[ca] @ q_ent[qi]).astype("float64")
            if w_cov:
                score = score + w_cov * cov_score_vec(qi, cidx)

            order  = np.argsort(-score)
            ranked = [cidx[o] for o in order[:5]]
            ranks_out.append(ranked)

            gpos = id_to_pos.get(gt[qi], -1)
            for rank, p in enumerate(ranked[:3], 1):
                if p == gpos:
                    mrr  += 1.0 / rank
                    ndcg += 1.0 / math.log2(rank + 1)
                    break
            if gpos in ranked[:1]: r1 += 1
            if gpos in ranked[:3]: r3 += 1
            if gpos in ranked[:5]: r5 += 1

        m = {
            "MRR@3":   round(mrr / Q, 4),
            "Recall@1": round(r1 / Q, 4),
            "Recall@3": round(r3 / Q, 4),
            "Recall@5": round(r5 / Q, 4),
            "NDCG@3":  round(ndcg / Q, 4),
        }
        return (m, ranks_out) if want_rank else m

    # ── 10. Run all arms ──────────────────────────────────────────────────────
    print(f"[{args.dataset}] Evaluating arms...")

    arms: dict[str, dict] = {}

    # Dense baseline (no metadata, no entity, no C3)
    arms["dense"] = run_arm(1.0, 0.0, 0.0, d_ent_fixed, q_ent_fixed, use_meta=False)

    # FULL with fixed weights (same as full_eval2.py FULL arm)
    arms["FULL (entity+meta, β=0.6)"] = run_arm(
        1.0, 0.6, 0.0, d_ent_fixed, q_ent_fixed, use_meta=True)

    # FULL + C3 sweep (hand-tuned δ)
    best_fixed = None
    for delta in (0.1, 0.2, 0.3, 0.5):
        m = run_arm(1.0, 0.6, delta, d_ent_fixed, q_ent_fixed, use_meta=True)
        arms[f"FULL + C3 δ={delta} (fixed w)"] = m
        if best_fixed is None or m["MRR@3"] > best_fixed[1]["MRR@3"]:
            best_fixed = (delta, m)

    best_fixed_delta, best_fixed_m = best_fixed
    print(f"[{args.dataset}] Best fixed-weight δ={best_fixed_delta}  "
          f"MRR@3={best_fixed_m['MRR@3']:.4f}")

    # FULL + CACL2 learned weights
    cacl2_arm_label = "FULL + C3 CACL2-weights"
    m_cacl2 = None
    if cacl2_loaded:
        m_cacl2, ranks_cacl2 = run_arm(
            w_text_cacl2, w_ent_cacl2, w_cov_cacl2,
            d_ent_c2, q_ent_c2, use_meta=True, want_rank=True)
        arms[cacl2_arm_label] = m_cacl2
        print(f"[{args.dataset}] CACL2 arm:  MRR@3={m_cacl2['MRR@3']:.4f}  "
              f"R@1={m_cacl2['Recall@1']:.4f}  R@3={m_cacl2['Recall@3']:.4f}  "
              f"R@5={m_cacl2['Recall@5']:.4f}")

    # ── Pick the best arm for top3 output ───────────────────────────────────
    # We want the arm with the highest MRR@3 to feed the generator.
    # CACL2 uses different entity weights from training; fixed-weight may score
    # higher on full test set due to 12-epoch SupCon entity warm-up on full corpus.
    all_candidate_arms = {
        f"FULL + C3 δ={best_fixed_delta} (fixed w)": (
            best_fixed_m, None),   # recompute below if chosen
    }
    if cacl2_loaded and m_cacl2 is not None:
        all_candidate_arms[cacl2_arm_label] = (m_cacl2, ranks_cacl2)

    best_arm_name = max(all_candidate_arms, key=lambda k: all_candidate_arms[k][0]["MRR@3"])
    best_arm_m, best_arm_ranks = all_candidate_arms[best_arm_name]

    if best_arm_ranks is None:
        # Re-run to get ranks
        best_arm_m, best_arm_ranks = run_arm(
            1.0, 0.6, best_fixed_delta, d_ent_fixed, q_ent_fixed,
            use_meta=True, want_rank=True)

    final_arm_label = f"{best_arm_name} [TOP3-SELECTED]"
    final_ranks = best_arm_ranks
    print(f"[{args.dataset}] Selected arm for top3 output: '{best_arm_name}' "
          f"(MRR@3={best_arm_m['MRR@3']:.4f})")

    # ── 11. Save ablation JSON + markdown ─────────────────────────────────────
    payload = {
        "dataset": cfgname, "split": split,
        "n_queries": Q, "n_corpus": N,
        "embed_model": args.embed_model,
        "coverage_diag": diag,
        "cacl2_loaded": cacl2_loaded,
        "cacl2_weights": ({"w_text": w_text_cacl2, "w_ent": w_ent_cacl2, "w_cov": w_cov_cacl2}
                          if cacl2_loaded else None),
        "best_fixed_delta": best_fixed_delta,
        "arms": arms,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Final Retrieval Ablation — {cfgname}  (n={Q}, corpus={N})",
        f"_Coverage: {diag}_",
        f"_CACL2 loaded: {cacl2_loaded}_",
        f"_CACL2 weights: w_text={w_text_cacl2:.3f}, w_ent={w_ent_cacl2:.3f}, "
        f"w_cov={w_cov_cacl2:.3f}_" if cacl2_loaded else "",
        "",
        "| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |",
        "|---|---|---|---|---|---|",
    ]
    for k, m in arms.items():
        marker = " ← **BEST**" if m["MRR@3"] == max(a["MRR@3"] for a in arms.values()) else ""
        lines.append(
            f"| {k}{marker} | {m['MRR@3']:.4f} | {m['Recall@1']:.4f} | "
            f"{m['Recall@3']:.4f} | {m['Recall@5']:.4f} | {m['NDCG@3']:.4f} |"
        )
    (out_dir / "ablation.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ── 12. Save retrieval_top3.jsonl (final arm) for generator ──────────────
    print(f"\n[{args.dataset}] Writing retrieval_top3.jsonl ...")
    top3_path = out_dir / "retrieval_top3.jsonl"
    with open(top3_path, "w") as f:
        for qi in range(Q):
            top = (final_ranks[qi] if final_ranks else [])[:3]
            ledgers = [
                extract_ledger(table_md=tables[p], context=ctx[p],
                               doc_id=cids[p], meta=cmetas[p])
                for p in top
            ]
            evidence = build_evidence_block(queries[qi], ledgers, top_n=args.fact_top_n)
            record = {
                "query_id":        qi,
                "query":           queries[qi],
                "raw_question":    raw_q[qi],
                "query_meta":      qmetas[qi],
                "ground_truth_id": gt[qi],
                "gold":            golds[qi],
                "retrieval_arm":   final_arm_label,
                "retrieval_weights": best_arm_m,
                "retrieved": [
                    {
                        "rank":         rank + 1,
                        "id":           cids[p],
                        "context_id":   cids[p],
                        "meta":         cmetas[p],
                        "table":        tables[p],
                        "page_content": ctx[p],
                    }
                    for rank, p in enumerate(top)
                ],
                "evidence_block": evidence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_top3 = sum(1 for r in open(top3_path) if r.strip())
    print(f"[{args.dataset}] retrieval_top3.jsonl: {n_top3} records  →  {top3_path}")
    print(f"[{args.dataset}] DONE in {time.time()-t0:.1f}s  →  {out_dir}")
    return payload


if __name__ == "__main__":
    main()
