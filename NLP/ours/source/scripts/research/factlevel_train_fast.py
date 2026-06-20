#!/usr/bin/env python3
"""C3 (fast) — train a fact-level encoder with the within-company ranking objective.

The last untested retrieval lever for the TAT-DQA residual (loclex 0.681). Freeze bge-small,
train a small projection P (shared for query & fact); doc score = max_f <P(q),P(c_f)>; listwise
InfoNCE over the company-complete pool (pull the gold doc above the in-company OTHER chunks).
Vectorised: P is applied to ALL facts once per minibatch (the only heavy op), so it is fast.

Usage:  PYTHONPATH=src python scripts/research/factlevel_train_fast.py --dataset TAT-DQA --device cuda:0
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


def mrr3(ranks): return float(np.mean([1.0/(r+1) if r < 3 else 0.0 for r in ranks]))
def rat(ranks, k): return float(np.mean([r < k for r in ranks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="TAT-DQA", choices=list(SPLITS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="outputs/research/factlevel_c3")
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = args.device if torch.cuda.is_available() else "cpu"

    data = load_t2ragbench_split(args.dataset, split=SPLITS[args.dataset], sample_size=None)
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str((m or {}).get("company_name", "")).strip()
        raw_q.append(q[len(c)+1:].lstrip() if c and q.startswith(c+":") else q)
    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]

    # per-doc fact labels → global fact index ranges
    labels, doc_fact = [], []
    for d in corpus:
        ids = []
        tmd = extract_table(d.page_content)
        if tmd:
            try:
                for f in extract_ledger_from_table(tmd).facts:
                    lab = (f.concept or "").strip()
                    if lab:
                        ids.append(len(labels)); labels.append(lab)
            except Exception:
                pass
        doc_fact.append(ids)
    print(f"facts={len(labels)} docs={len(corpus)} queries={len(raw_q)}", flush=True)

    enc = SentenceTransformer(args.embed_model, device=dev)
    fact_emb = torch.tensor(enc.encode(labels, normalize_embeddings=True, convert_to_numpy=True,
                                       batch_size=512, show_progress_bar=False),
                            dtype=torch.float32, device=dev)
    q_emb = torch.tensor(enc.encode(raw_q, normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=512, show_progress_bar=False),
                         dtype=torch.float32, device=dev)

    meta = MetadataRetriever(company_pool=True, max_add=10000)
    meta.prepare(corpus, doc_metas); meta.set_queries(raw_q, metas)
    pools, gpos, usable = [], [], []
    for qi in range(len(raw_q)):
        gi = gold_idx[qi]; pool = meta.get_candidates(qi, 10000)
        pools.append(pool)
        if gi >= 0 and gi in pool and len(pool) >= 2:
            gpos.append(pool.index(gi)); usable.append(qi)
        else:
            gpos.append(-1)
    rng.shuffle(usable); cut = len(usable)//2
    train_q, test_q = usable[:cut], usable[cut:]
    print(f"usable={len(usable)} train={len(train_q)} test={len(test_q)}", flush=True)

    # pad per-doc fact indices to a tensor for fast segment-max
    maxf = max((len(f) for f in doc_fact), default=1) or 1
    fidx = torch.full((len(corpus), maxf), -1, dtype=torch.long, device=dev)
    fmask = torch.zeros((len(corpus), maxf), dtype=torch.bool, device=dev)
    for di, ids in enumerate(doc_fact):
        if ids:
            fidx[di, :len(ids)] = torch.tensor(ids, device=dev)
            fmask[di, :len(ids)] = True

    proj = nn.Sequential(nn.Linear(384, 256), nn.Tanh(), nn.Linear(256, 256)).to(dev)
    log_tau = torch.tensor(np.log(0.07), device=dev, requires_grad=True)
    opt = torch.optim.Adam(list(proj.parameters()) + [log_tau], lr=1e-3)

    def doc_scores_for(pool, pq, Pf):
        idx = fidx[pool]; msk = fmask[pool]                     # [P,maxf]
        pf = Pf[idx]                                            # [P,maxf,256]
        sims = (pf * pq).sum(-1)                                # [P,maxf]
        sims = sims.masked_fill(~msk, -1e4)
        return sims.max(dim=1).values                          # [P]

    for ep in range(args.epochs):
        proj.train(); order = list(train_q); rng.shuffle(order); tot = 0.0
        for b in range(0, len(order), 64):
            batch = order[b:b+64]
            Pf = F.normalize(proj(fact_emb), dim=1)
            Pq = F.normalize(proj(q_emb[batch]), dim=1)
            loss = 0.0
            for j, qi in enumerate(batch):
                s = doc_scores_for(pools[qi], Pq[j], Pf) / log_tau.exp()
                loss = loss + F.cross_entropy(s.unsqueeze(0), torch.tensor([gpos[qi]], device=dev))
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if ep % 5 == 0 or ep == args.epochs-1:
            print(f"  ep{ep:2d} loss={tot/max(1,len(order)//64+1):.4f}", flush=True)

    # eval vs loclex
    proj.eval()
    loc = LocalLexicalExpert(abbr_expand=True); loc.prepare(corpus, doc_metas); loc.set_queries(raw_q, metas)
    ft_r, lx_r, fu_r = [], [], []
    with torch.no_grad():
        Pf = F.normalize(proj(fact_emb), dim=1)
        Pq = F.normalize(proj(q_emb), dim=1)
        for qi in test_q:
            pool = pools[qi]; gp = gpos[qi]
            s_ft = doc_scores_for(pool, Pq[qi], Pf).cpu().numpy()
            s_lx = loc.score_pool(qi, pool)
            def rk(s):
                o = np.argsort(-s); h = np.where(o == gp)[0]; return int(h[0]) if h.size else 10**9
            ft_r.append(rk(s_ft)); lx_r.append(rk(s_lx))
            def mm(x):
                lo, hi = x.min(), x.max(); return (x-lo)/(hi-lo) if hi-lo > 1e-9 else np.full_like(x, .5)
            fu_r.append(rk(mm(s_ft)+mm(s_lx)))
    res = {"dataset": args.dataset, "n_test": len(test_q),
           "factlevel_trained_MRR@3": round(mrr3(ft_r), 4),
           "loclex_MRR@3": round(mrr3(lx_r), 4),
           "fusion_MRR@3": round(mrr3(fu_r), 4),
           "factlevel_R@1": round(rat(ft_r, 1), 4), "loclex_R@1": round(rat(lx_r, 1), 4)}
    print(f"\n=== C3 trained fact encoder vs loclex — {args.dataset} (test={len(test_q)}) ===")
    print(f"   loclex            MRR@3={res['loclex_MRR@3']}")
    print(f"   factlevel-trained MRR@3={res['factlevel_trained_MRR@3']}")
    print(f"   fusion(ft+loclex) MRR@3={res['fusion_MRR@3']}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out)/f"{args.dataset.lower()}.json").write_text(json.dumps(res, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset.lower()+'.json')}")


if __name__ == "__main__":
    main()
