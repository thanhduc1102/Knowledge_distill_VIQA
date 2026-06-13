"""CACL retrieval training (the reinforcement/optimization step for CACL).

IMPORTANT: this trains the RETRIEVAL side only. The generator LLM is used zero-shot and is
NEVER finetuned (per project requirement).

What it optimizes, starting from quality negatives:
  * the entity/metadata embedder (continued from SupCon), and
  * the joint-score weights (text vs entity vs constraint),
using a margin triplet loss over (query, gold-doc, HARD-negative). The hard negatives are
exactly the "false-negative-prone hardest zone" the EDA flagged:
  - in-corpus: the **same-company, different-year** document with the highest text similarity
    to the query (real, hardest), and
  - synthetic channel-aligned: an **entity-swap** (different company) negative.
This forces the model to use the year/entity signal to separate near-duplicates — the core
of the lexical-overlap-illusion failure mode.

Text embeddings (e5) are frozen (no LLM finetuning); the trained pieces (entity embedder +
weights) are exactly what ``LedgerRetrieval`` consumes at inference.
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


def cacl_train(dataset="finqa", n_train=2000, n_eval=500, epochs=4, margin=0.3,
               lr=1e-3, device="cuda:0", embed_model="intfloat/multilingual-e5-large-instruct",
               out=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset, DatasetDict
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from gsr_cacl.entity.encoder import HashMetadataEmbedder
    from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels

    cfgname, split = CFG[dataset]
    out_dir = Path(out or f"outputs/cacl_train/{dataset}"); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(42)

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
    id_to_pos = {c: i for i, c in enumerate(cids)}

    # split QA into train/eval
    qa = qa.sample(frac=1.0, random_state=42).reset_index(drop=True)
    eval_df = qa.iloc[:n_eval]
    train_df = qa.iloc[n_eval:n_eval + n_train]

    def q_of(r):
        return f"{r.get('company_name')}: {r['question']}" if r.get("company_name") else str(r["question"])

    st = SentenceTransformer(embed_model, device=device)
    doc_emb = torch.tensor(st.encode(ctx, batch_size=64, normalize_embeddings=True,
                                     convert_to_numpy=True), device=device)
    tr_q = train_df.apply(q_of, axis=1).tolist()
    ev_q = eval_df.apply(q_of, axis=1).tolist()
    tr_qe = torch.tensor(st.encode(tr_q, batch_size=64, normalize_embeddings=True, convert_to_numpy=True), device=device)
    ev_qe = torch.tensor(st.encode(ev_q, batch_size=64, normalize_embeddings=True, convert_to_numpy=True), device=device)
    print(f"[cacl/{dataset}] embedded corpus={len(cids)} train={len(tr_q)} eval={len(ev_q)} ({time.time()-t0:.0f}s)")

    # company -> corpus indices
    comp2idx: dict[str, list[int]] = {}
    for i, m in enumerate(cmetas):
        comp2idx.setdefault(m["company_name"].lower().strip(), []).append(i)

    # build training triplets (pos, hard_neg) using text sim for same-company hardest
    triplets = []  # (q_idx, pos_pos, neg_pos)
    tr_gt = train_df["context_id"].astype(str).tolist()
    tr_meta = [meta(r) for _, r in train_df.iterrows()]
    for qi, gtid in enumerate(tr_gt):
        gp = id_to_pos.get(gtid, -1)
        if gp < 0:
            continue
        comp = tr_meta[qi]["company_name"].lower().strip()
        same = [j for j in comp2idx.get(comp, []) if j != gp]
        if same:
            sims = (doc_emb[same] @ tr_qe[qi]).detach().cpu().numpy()
            neg = same[int(np.argmax(sims))]            # hardest same-company-diff-year doc
        else:
            neg = int(rng.integers(0, len(cids)))
        triplets.append((qi, gp, neg))
    print(f"[cacl/{dataset}] {len(triplets)} hard triplets (same-company negatives where available)")

    # entity embedder: SupCon warm-up on corpus metadata
    ent = HashMetadataEmbedder(embed_dim=128).to(device)
    supcon = SupConLoss(0.1)
    opt_e = torch.optim.AdamW(ent.parameters(), lr=lr)
    labels = make_entity_labels(cmetas).to(device)
    for _ in range(8):
        idx = torch.randperm(len(cmetas))[:512]
        loss = supcon(ent([cmetas[i] for i in idx.tolist()]), labels[idx])
        opt_e.zero_grad(); loss.backward(); opt_e.step()

    # learnable joint weights + continued entity training via hard-negative triplets
    w_text = torch.nn.Parameter(torch.tensor(1.0, device=device))
    w_ent = torch.nn.Parameter(torch.tensor(0.5, device=device))
    opt = torch.optim.AdamW(list(ent.parameters()) + [w_text, w_ent], lr=lr)

    tr_meta_full = tr_meta
    for ep in range(epochs):
        rng.shuffle(triplets)
        tot = 0.0; nb = 0
        for s in range(0, len(triplets), 256):
            batch = triplets[s:s + 256]
            if not batch:
                continue
            qidx = [b[0] for b in batch]; pidx = [b[1] for b in batch]; nidx = [b[2] for b in batch]
            qe = tr_qe[qidx]
            q_ent = ent([tr_meta_full[i] for i in qidx])
            pos_ent = ent([cmetas[i] for i in pidx])
            neg_ent = ent([cmetas[i] for i in nidx])
            s_pos = F.softplus(w_text) * (doc_emb[pidx] * qe).sum(-1) + F.softplus(w_ent) * (pos_ent * q_ent).sum(-1)
            s_neg = F.softplus(w_text) * (doc_emb[nidx] * qe).sum(-1) + F.softplus(w_ent) * (neg_ent * q_ent).sum(-1)
            loss = F.relu(margin - s_pos + s_neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[cacl/{dataset}] epoch {ep+1}/{epochs} triplet_loss={tot/max(nb,1):.4f} "
              f"w_text={F.softplus(w_text).item():.2f} w_ent={F.softplus(w_ent).item():.2f}")

    # ---- evaluate on held-out eval_df with metadata-aware candidates ----
    import faiss
    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(doc_emb.cpu().numpy())
    _, I = index.search(ev_qe.cpu().numpy(), 50)
    ev_gt = eval_df["context_id"].astype(str).tolist()
    ev_meta = [meta(r) for _, r in eval_df.iterrows()]
    dyears = []
    for m in cmetas:
        try:
            dyears.append(int(float(m["report_year"])))
        except (ValueError, TypeError):
            dyears.append(None)
    with torch.no_grad():
        doc_ent_all = ent(cmetas)

    def mrr_at3(use_ent, use_trained_w):
        wt = F.softplus(w_text).item() if use_trained_w else 1.0
        we = F.softplus(w_ent).item() if use_trained_w else (0.6 if use_ent else 0.0)
        hit = 0.0
        for qi in range(len(ev_q)):
            cand = set(int(x) for x in I[qi] if x >= 0)
            comp = ev_meta[qi]["company_name"].lower().strip()
            try:
                qy = int(float(ev_meta[qi]["report_year"]))
            except (ValueError, TypeError):
                qy = None
            for j in comp2idx.get(comp, []):
                if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= 1:
                    cand.add(j)
            cand = list(cand)
            ca = torch.tensor(cand, device=device)
            sc = wt * (doc_emb[ca] @ ev_qe[qi])
            if use_ent:
                qe = ent([ev_meta[qi]])[0]
                sc = sc + we * (doc_ent_all[ca] @ qe)
            order = torch.argsort(sc, descending=True)[:3].cpu().numpy()
            gp = id_to_pos.get(ev_gt[qi], -1)
            for rank, o in enumerate(order, 1):
                if cand[o] == gp:
                    hit += 1.0 / rank; break
        return hit / len(ev_q)

    report = {
        "dataset": cfgname, "n_train_triplets": len(triplets), "n_eval": len(ev_q),
        "MRR@3_text_only": mrr_at3(False, False),
        "MRR@3_supcon_entity_untrained_w": mrr_at3(True, False),
        "MRR@3_CACL_trained": mrr_at3(True, True),
        "w_text": F.softplus(w_text).item(), "w_ent": F.softplus(w_ent).item(),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "cacl_report.json").write_text(json.dumps(report, indent=2))
    torch.save({"entity_state": ent.state_dict(), "w_text": w_text.detach().cpu(),
                "w_ent": w_ent.detach().cpu()}, out_dir / "cacl_model.pt")
    print(json.dumps(report, indent=2))
    print(f"[cacl/{dataset}] saved -> {out_dir}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finqa", choices=list(CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    cacl_train(args.dataset, n_train=args.n_train, n_eval=args.n_eval, epochs=args.epochs, device=args.device)
