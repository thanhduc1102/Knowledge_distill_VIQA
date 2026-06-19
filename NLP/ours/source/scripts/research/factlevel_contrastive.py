#!/usr/bin/env python3
"""C3 — Contrastive training of the disentangled fact encoder (decisive test vs loclex).

The C2 zero-shot result (docs/RESEARCH_AAAI27.md §10.2) showed that a frozen bge-small
concept stream does NOT beat pool-local IDF (loclex) for within-company chunk selection.
This script tests the C3 hypothesis: *a TRAINED concept encoder, supervised with the
within-company ranking objective, can beat loclex.*

Setup (honest, query-split):
  * Pool = company-complete set (recall 1.0; the within-(company) disambiguation problem).
  * Encoder = frozen bge-small  + a small trainable projection P (shared for query & facts).
  * Doc score s(q,d) = max_{f in L(d)} <P(q), P(c_f)>  (multiple-instance: doc-level labels).
  * Loss = listwise InfoNCE over the company pool: pull the gold doc up against the
    in-company OTHER chunks (the real hard negatives for within-filing disambiguation).
  * Eval on a held-out query split; compare trained factlevel vs loclex on the SAME pools.

Usage:  PYTHONPATH=src python scripts/research/factlevel_contrastive.py --dataset FinQA --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.datasets.gsr_document import extract_table
from gsr_cacl.experts.local_lexical import LocalLexicalExpert
from gsr_cacl.experts.meta_retriever import MetadataRetriever
from gsr_cacl.ledger.extract import extract_ledger_from_table

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}


def mrr3(ranks):
    return float(np.mean([1.0 / (r + 1) if r < 3 else 0.0 for r in ranks]))


def r_at(ranks, k):
    return float(np.mean([r < k for r in ranks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=list(SPLITS))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/research/factlevel_c3")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    split = SPLITS[args.dataset]
    data = load_t2ragbench_split(args.dataset, split=split, sample_size=(args.sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)
    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]

    # ---- per-doc fact labels -------------------------------------------------
    print("extracting fact labels ...", flush=True)
    label_strings, doc_fact_ids = [], []
    for d in corpus:
        ids = []
        tmd = extract_table(d.page_content)
        led = None
        if tmd:
            try:
                led = extract_ledger_from_table(tmd)
            except Exception:
                led = None
        if led:
            for f in led.facts:
                lab = (f.concept or "").strip()
                if lab:
                    ids.append(len(label_strings))
                    label_strings.append(lab)
        doc_fact_ids.append(ids)

    print(f"encoding {len(label_strings)} fact labels + {len(raw_q)} queries with {args.embed_model} ...", flush=True)
    enc = SentenceTransformer(args.embed_model, device=dev)
    fact_emb = torch.tensor(enc.encode(label_strings, normalize_embeddings=True,
                                       convert_to_numpy=True, batch_size=512,
                                       show_progress_bar=False), dtype=torch.float32, device=dev)
    q_emb = torch.tensor(enc.encode(raw_q, normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=512, show_progress_bar=False),
                         dtype=torch.float32, device=dev)
    D = fact_emb.shape[1]

    # ---- company pools -------------------------------------------------------
    meta = MetadataRetriever(company_pool=True, max_add=10000)
    meta.prepare(corpus, doc_metas)
    meta.set_queries(raw_q, metas)
    pools, gpos, valid = [], [], []
    for qi in range(len(raw_q)):
        gi = gold_idx[qi]
        pool = meta.get_candidates(qi, 10000)
        if gi >= 0 and gi in pool and len(pool) >= 2:
            pools.append(pool); gpos.append(pool.index(gi)); valid.append(qi)
        else:
            pools.append(pool); gpos.append(-1); valid.append(None)
    usable = [qi for qi in range(len(raw_q)) if valid[qi] is not None]
    rng.shuffle(usable)
    cut = len(usable) // 2
    train_q, test_q = usable[:cut], usable[cut:]
    print(f"usable queries: {len(usable)} (train {len(train_q)} / test {len(test_q)})", flush=True)

    # Precompute padded fact-embedding tensor per doc for fast max-pool.
    # doc_facts_emb[d] = [n_f, D]; cache as python list of tensors.
    doc_fe = [fact_emb[torch.tensor(ids, device=dev)] if ids else None for ids in doc_fact_ids]

    # ---- model: shared projection -------------------------------------------
    proj = nn.Sequential(nn.Linear(D, 256), nn.Tanh(), nn.Linear(256, 256)).to(dev)
    log_tau = torch.tensor(np.log(0.07), device=dev, requires_grad=True)
    opt = torch.optim.Adam(list(proj.parameters()) + [log_tau], lr=1e-3)

    def doc_scores(qi, pool, pq):
        """max-over-facts <P(q),P(f)> for each doc in pool. pq = P(query) [256]."""
        s = []
        for d in pool:
            fe = doc_fe[d]
            if fe is None:
                s.append(torch.tensor(-1.0, device=dev)); continue
            pf = nn.functional.normalize(proj(fe), dim=1)   # [n_f,256]
            s.append((pf @ pq).max())
        return torch.stack(s)

    # ---- train ---------------------------------------------------------------
    for ep in range(args.epochs):
        proj.train()
        order = list(train_q); rng.shuffle(order)
        tot = 0.0
        opt.zero_grad()
        for bi, qi in enumerate(order):
            pool, gp = pools[qi], gpos[qi]
            pq = nn.functional.normalize(proj(q_emb[qi]), dim=0)
            s = doc_scores(qi, pool, pq) / log_tau.exp()
            loss = nn.functional.cross_entropy(s.unsqueeze(0), torch.tensor([gp], device=dev))
            loss.backward(); tot += float(loss)
            if (bi + 1) % 16 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:2d}  loss={tot/max(len(order),1):.4f}  tau={log_tau.exp().item():.3f}", flush=True)

    # ---- eval: trained factlevel vs loclex on test queries -------------------
    proj.eval()
    loc = LocalLexicalExpert(abbr_expand=True)
    loc.prepare(corpus, doc_metas); loc.set_queries(raw_q, metas)

    def rank_gold(scores_np, pool, gp):
        order = np.argsort(-scores_np)
        hit = np.where(order == gp)[0]
        return int(hit[0]) if hit.size else 10**9

    ft_ranks, lx_ranks, fuse_ranks = [], [], []
    with torch.no_grad():
        for qi in test_q:
            pool, gp = pools[qi], gpos[qi]
            pq = nn.functional.normalize(proj(q_emb[qi]), dim=0)
            s_ft = doc_scores(qi, pool, pq).cpu().numpy()
            s_lx = loc.score_pool(qi, pool)
            ft_ranks.append(rank_gold(s_ft, pool, gp))
            lx_ranks.append(rank_gold(s_lx, pool, gp))
            # simple sum of min-maxed scores (no training) as a quick fusion probe
            def mm(x):
                lo, hi = x.min(), x.max()
                return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(x, 0.5)
            fuse_ranks.append(rank_gold(mm(s_ft) + mm(s_lx), pool, gp))

    res = {
        "dataset": args.dataset, "n_test": len(test_q),
        "factlevel_trained": {"MRR@3": round(mrr3(ft_ranks), 4), "R@1": round(r_at(ft_ranks, 1), 4), "R@3": round(r_at(ft_ranks, 3), 4)},
        "loclex": {"MRR@3": round(mrr3(lx_ranks), 4), "R@1": round(r_at(lx_ranks, 1), 4), "R@3": round(r_at(lx_ranks, 3), 4)},
        "fusion_ft+loclex(untrained)": {"MRR@3": round(mrr3(fuse_ranks), 4), "R@1": round(r_at(fuse_ranks, 1), 4), "R@3": round(r_at(fuse_ranks, 3), 4)},
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset.lower()}.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== C3 trained fact encoder vs loclex — {args.dataset} (test={len(test_q)}) ===")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k:<28} MRR@3={v['MRR@3']}  R@1={v['R@1']}  R@3={v['R@3']}")
    print(f"Saved → {Path(args.out)/(args.dataset.lower()+'.json')}")


if __name__ == "__main__":
    main()
