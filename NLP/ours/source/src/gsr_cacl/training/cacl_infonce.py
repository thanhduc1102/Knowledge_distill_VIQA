"""CACL v2 — InfoNCE retrieval training over realistic hard negatives (D1–D4).

Upgrades the original ``cacl_train.py`` (single-triplet margin, entity-only):

  * **D2 — InfoNCE** with many negatives + temperature, instead of a single margin triplet.
  * **D4 — false-negative guard**: negatives are drawn from the *actual confusable pool*
    (same company ±year — the EDA's hardest zone) but the gold ``context_id`` is excluded,
    and same-(company,year) duplicates are optionally dropped to avoid training against
    documents that may also be correct.
  * **D1 — train the signals that retrieval actually uses**: the loss optimises the joint
    weights over **text + entity + concept-coverage** (the C3 signal) *and* continues the
    entity embedder. Previously the elaborate negatives never touched a used signal; now the
    concept-coverage channel is learned jointly, so a same-company doc with the *wrong
    concept/period* is pushed below the gold.
  * **Channel-aligned negatives (D3 link)**: the value-perturbation negatives from
    ``ChannelAlignedSampler`` break the *constraint/verifier* channel (used by generation),
    so they are surfaced as a diagnostic here rather than mixed into the retrieval InfoNCE,
    keeping the retrieval objective clean.

Text embeddings (e5) stay frozen (no LLM finetuning). Trained artefacts = entity embedder +
joint weights [w_text, w_ent, w_cov], which ``LedgerRetrieval`` / ``full_eval2`` consume.
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


def cacl_infonce_train(dataset="finqa", n_train=2000, n_eval=500, epochs=6, tau=0.05,
                       n_neg=8, year_window=1, drop_same_cy=True, lr=1e-3, device="cuda:0",
                       embed_model="intfloat/multilingual-e5-large-instruct", out=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset, DatasetDict
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from gsr_cacl.entity.encoder import OntologyMetadataEmbedder
    from gsr_cacl.entity.supcon import SupConLoss, make_entity_labels
    from gsr_cacl.ontology import normalize_company
    from gsr_cacl.ledger import extract_ledger
    from gsr_cacl.scoring.concept_coverage import (
        query_concepts, query_periods, doc_periods_from_ledger, concept_coverage_score)

    cfgname, split = CFG[dataset]
    out_dir = Path(out or f"outputs/cacl_infonce/{dataset}"); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rng = np.random.default_rng(42)

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
    N = len(cids); id_to_pos = {c: i for i, c in enumerate(cids)}

    qa = qa.sample(frac=1.0, random_state=42).reset_index(drop=True)
    eval_df = qa.iloc[:n_eval]; train_df = qa.iloc[n_eval:n_eval + n_train]

    def q_of(r):
        return f"{r.get('company_name')}: {r['question']}" if r.get("company_name") else str(r["question"])

    st = SentenceTransformer(embed_model, device=device)
    doc_emb = torch.tensor(st.encode(ctx, batch_size=64, normalize_embeddings=True, convert_to_numpy=True), device=device)
    tr_q = train_df.apply(q_of, axis=1).tolist(); ev_q = eval_df.apply(q_of, axis=1).tolist()
    tr_qe = torch.tensor(st.encode(tr_q, batch_size=64, normalize_embeddings=True, convert_to_numpy=True), device=device)
    ev_qe = torch.tensor(st.encode(ev_q, batch_size=64, normalize_embeddings=True, convert_to_numpy=True), device=device)

    # company -> corpus indices (alias-canonical), doc years
    comp2idx: dict[str, list[int]] = {}
    dyears = []
    for i, m in enumerate(cmetas):
        comp2idx.setdefault(normalize_company(m["company_name"]), []).append(i)
        try:
            dyears.append(int(float(m["report_year"])))
        except (ValueError, TypeError):
            dyears.append(None)

    # per-doc concept/period sets (C3)
    doc_concepts, doc_periods = [], []
    for i in range(N):
        led = extract_ledger(table_md=tables[i], context=ctx[i], doc_id=cids[i], meta=cmetas[i])
        doc_concepts.append(led.concept_set()); doc_periods.append(doc_periods_from_ledger(led))

    def cov_vec(q_text, idxs):
        qc, qp = query_concepts(q_text), query_periods(q_text)
        return np.array([concept_coverage_score(qc, qp, doc_concepts[j], doc_periods[j]) for j in idxs],
                        dtype="float32")

    def metapool(qmeta, gpos):
        comp = normalize_company(qmeta.get("company_name", ""))
        try:
            qy = int(float(qmeta.get("report_year", "")))
        except (ValueError, TypeError):
            qy = None
        pool = []
        for j in comp2idx.get(comp, []):
            if j == gpos:
                continue
            if drop_same_cy and qy is not None and dyears[j] == qy:
                continue                     # D4: skip same-(company,year) potential false negs
            if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= year_window:
                pool.append(j)
        return pool

    # build training examples: (q_idx, gold_pos, [neg_pos...])
    examples = []
    tr_gt = train_df["context_id"].astype(str).tolist()
    tr_meta = [meta(r) for _, r in train_df.iterrows()]
    raw_tr_q = train_df["question"].astype(str).tolist()
    for qi, gid in enumerate(tr_gt):
        gp = id_to_pos.get(gid, -1)
        if gp < 0:
            continue
        pool = metapool(tr_meta[qi], gp)
        if pool:
            sims = (doc_emb[pool] @ tr_qe[qi]).detach().cpu().numpy()
            order = np.argsort(-sims)[:n_neg]            # hardest (text-similar) negatives
            negs = [pool[o] for o in order]
        else:
            negs = [int(rng.integers(0, N)) for _ in range(min(n_neg, 4))]
        examples.append((qi, gp, negs))
    print(f"[cacl2/{dataset}] {len(examples)} train examples, embedded ({time.time()-t0:.0f}s)")

    # entity embedder warm-up (SupCon over full corpus metadata) + joint weights
    ent = OntologyMetadataEmbedder(embed_dim=128).to(device)
    supcon = SupConLoss(0.1); labels = make_entity_labels(cmetas).to(device)
    opt_e = torch.optim.AdamW(ent.parameters(), lr=lr)
    for _ in range(12):                              # epochs over the whole corpus
        perm_e = torch.randperm(N)
        for b in range(0, N, 256):
            idx = perm_e[b:b + 256]
            loss = supcon(ent([cmetas[i] for i in idx.tolist()]), labels[idx])
            opt_e.zero_grad(); loss.backward(); opt_e.step()

    w_text = torch.nn.Parameter(torch.tensor(1.0, device=device))
    w_ent = torch.nn.Parameter(torch.tensor(0.6, device=device))
    w_cov = torch.nn.Parameter(torch.tensor(0.1, device=device))
    opt = torch.optim.AdamW(list(ent.parameters()) + [w_text, w_ent, w_cov], lr=lr)

    # precompute coverage rows per example (depends only on q + doc sets, not on params)
    cov_rows = []
    for (qi, gp, negs) in examples:
        idxs = [gp] + negs
        cov_rows.append(torch.tensor(cov_vec(raw_tr_q[qi], idxs), device=device))

    for ep in range(epochs):
        perm = list(range(len(examples))); rng.shuffle(perm)
        tot = 0.0; nb = 0
        for s in range(0, len(perm), 128):
            batch = perm[s:s + 128]
            if not batch:
                continue
            loss = 0.0
            q_metas_b = [tr_meta[examples[k][0]] for k in batch]
            q_ent_b = ent(q_metas_b)                               # [B, de]
            for bi, k in enumerate(batch):
                qi, gp, negs = examples[k]
                idxs = [gp] + negs
                de = ent([cmetas[j] for j in idxs])                # [1+n, de]
                s_text = doc_emb[idxs] @ tr_qe[qi]                 # [1+n]
                s_ent = de @ q_ent_b[bi]
                s_cov = cov_rows[k]
                score = F.softplus(w_text) * s_text + F.softplus(w_ent) * s_ent + F.softplus(w_cov) * s_cov
                loss = loss + F.cross_entropy(score.unsqueeze(0) / tau,
                                              torch.zeros(1, dtype=torch.long, device=device))
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[cacl2/{dataset}] epoch {ep+1}/{epochs} infonce={tot/max(nb,1):.4f} "
              f"w_text={F.softplus(w_text).item():.2f} w_ent={F.softplus(w_ent).item():.2f} "
              f"w_cov={F.softplus(w_cov).item():.2f}")

    # ---- eval on held-out queries (metadata-aware candidates) ----
    import faiss
    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(doc_emb.cpu().numpy())
    _, I = index.search(ev_qe.cpu().numpy(), 50)
    ev_gt = eval_df["context_id"].astype(str).tolist()
    ev_meta = [meta(r) for _, r in eval_df.iterrows()]
    raw_ev_q = eval_df["question"].astype(str).tolist()
    with torch.no_grad():
        doc_ent_all = ent(cmetas)

    def cand(qi):
        s = set(int(x) for x in I[qi] if x >= 0)
        comp = normalize_company(ev_meta[qi]["company_name"])
        try:
            qy = int(float(ev_meta[qi]["report_year"]))
        except (ValueError, TypeError):
            qy = None
        for j in comp2idx.get(comp, []):
            if qy is None or dyears[j] is None or abs(dyears[j] - qy) <= year_window:
                s.add(j)
        return list(s)

    def mrr(use_trained_w, use_cov):
        wt = F.softplus(w_text).item() if use_trained_w else 1.0
        we = F.softplus(w_ent).item() if use_trained_w else 0.6
        wc = (F.softplus(w_cov).item() if use_trained_w else 0.1) if use_cov else 0.0
        hit = 0.0
        for qi in range(len(ev_q)):
            ci = cand(qi)
            if not ci:
                continue
            ca = torch.tensor(ci, device=device)
            sc = wt * (doc_emb[ca] @ ev_qe[qi]) + we * (doc_ent_all[ca] @ ent([ev_meta[qi]])[0])
            if wc:
                sc = sc + wc * torch.tensor(cov_vec(raw_ev_q[qi], ci), device=device)
            order = torch.argsort(sc, descending=True)[:3].cpu().numpy()
            gp = id_to_pos.get(ev_gt[qi], -1)
            for rank, o in enumerate(order, 1):
                if ci[o] == gp:
                    hit += 1.0 / rank; break
        return hit / len(ev_q)

    report = {"dataset": cfgname, "n_train": len(examples), "n_eval": len(ev_q),
              "tau": tau, "n_neg": n_neg, "drop_same_cy": drop_same_cy,
              "MRR@3_text+entity_default_w": mrr(False, False),
              "MRR@3_CACL2_trained_no_cov": mrr(True, False),
              "MRR@3_CACL2_trained_full": mrr(True, True),
              "w_text": F.softplus(w_text).item(), "w_ent": F.softplus(w_ent).item(),
              "w_cov": F.softplus(w_cov).item(), "seconds": round(time.time() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"entity_state": ent.state_dict(),
                "w_text": w_text.detach().cpu(), "w_ent": w_ent.detach().cpu(),
                "w_cov": w_cov.detach().cpu()}, out_dir / "cacl2_model.pt")
    print(json.dumps(report, indent=2)); print(f"[cacl2/{dataset}] saved -> {out_dir}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finqa", choices=list(CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n-neg", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.05)
    args = ap.parse_args()
    cacl_infonce_train(args.dataset, n_train=args.n_train, n_eval=args.n_eval,
                       epochs=args.epochs, n_neg=args.n_neg, tau=args.tau, device=args.device)
