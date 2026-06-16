#!/usr/bin/env python3
"""Fine-tune the LateInteraction fact encoder (Phase: lateint-FT).

Goal: teach the fact encoder the financial query→fact matching so fact-level MaxSim both
ranks better AND pulls more gold documents into the candidate pool (the recall lever that
caps TAT-DQA). Training is contrastive (InfoNCE via MultipleNegativesRankingLoss):

  anchor   = query (with the bge retrieval instruction, matching inference)
  positive = the gold document's fact most similar to the query (mined hard positive)
  hard neg = a fact from a SAME-COMPANY ±1-year document (the EDA's hardest zone),
             chosen as the most query-similar negative; + in-batch negatives.

Honest split usage: trained ONLY on FinQA.train + TAT-DQA.train (ConvFinQA has no train split,
so it stays a pure zero-shot transfer test). Eval queries (the test splits) are never seen.

Usage:  PYTHONPATH=src python scripts/finetune_lateint.py --device cuda --out outputs/lateint_ft
"""
from __future__ import annotations

import argparse, os, random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.experts.late_interaction import _doc_fact_strings, _BGE_QUERY_INSTR
from gsr_cacl.ontology.aliases import normalize_company
from gsr_cacl.ledger.numeric import extract_years

TRAIN = [("FinQA", "train"), ("TAT-DQA", "train")]


def mine_dataset(cfg, split, model, max_q, max_facts=80, seed=0):
    rng = random.Random(seed)
    data = load_t2ragbench_split(cfg, split=split, sample_size=None)
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)
    id2idx = {str(d.id): i for i, d in enumerate(corpus)}

    # per-doc facts + flat encode
    doc_facts = [_doc_fact_strings(d.page_content)[:max_facts] for d in corpus]
    flat, fdoc = [], []
    for di, fs in enumerate(doc_facts):
        for s in fs:
            flat.append(s); fdoc.append(di)
    if not flat:
        return []
    print(f"  [{cfg}] encoding {len(flat)} facts for mining ...", flush=True)
    fact_emb = np.asarray(model.encode(flat, normalize_embeddings=True, convert_to_numpy=True,
                                       batch_size=256, show_progress_bar=False), np.float32)
    fdoc = np.asarray(fdoc)
    # per-doc fact index ranges (flat is grouped by doc)
    doc_fstart = np.searchsorted(fdoc, np.arange(len(corpus)))

    # company index for hard negatives
    comp2idx: dict[str, list[int]] = {}
    dyear = []
    for i, m in enumerate(metas if False else [d.meta_data for d in corpus]):
        comp2idx.setdefault(normalize_company(m.get("company_name", "")), []).append(i)
        ys = extract_years(str(m.get("report_year", "")))
        dyear.append(ys[0] if ys else None)

    # sample queries
    qidx = list(range(len(raw_q)))
    rng.shuffle(qidx)
    qidx = qidx[:max_q]
    q_emb = np.asarray(model.encode([_BGE_QUERY_INSTR + raw_q[i] for i in qidx],
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=256, show_progress_bar=False), np.float32)

    def doc_fact_slice(di):
        a = doc_fstart[di]
        b = doc_fstart[di + 1] if di + 1 < len(corpus) else len(flat)
        return a, b

    triples = []
    for k, qi in enumerate(qidx):
        g = id2idx.get(str(gts[qi]), -1)
        if g < 0:
            continue
        ga, gb = doc_fact_slice(g)
        if gb <= ga:
            continue
        qv = q_emb[k]
        # hard positive = gold fact most similar to query
        pos_local = int(np.argmax(fact_emb[ga:gb] @ qv))
        pos_fact = flat[ga + pos_local]
        # hard negative = same-company ±1yr doc's most-similar fact
        m = corpus[g].meta_data
        comp = normalize_company(m.get("company_name", "")); gy = dyear[g]
        neg_docs = [j for j in comp2idx.get(comp, [])
                    if j != g and (gy is None or dyear[j] is None or abs(dyear[j] - gy) <= 1)]
        neg_fact = None
        best = -2.0
        for j in neg_docs:
            a, b = doc_fact_slice(j)
            if b <= a:
                continue
            loc = int(np.argmax(fact_emb[a:b] @ qv)); sc = float(fact_emb[a + loc] @ qv)
            if sc > best:
                best, neg_fact = sc, flat[a + loc]
        if neg_fact is None:  # fallback: a random other-doc fact
            for _ in range(5):
                r = rng.randrange(len(flat))
                if fdoc[r] != g:
                    neg_fact = flat[r]; break
        if neg_fact is None:
            continue
        triples.append((_BGE_QUERY_INSTR + raw_q[qi], pos_fact, neg_fact))
    print(f"  [{cfg}] mined {len(triples)} triples", flush=True)
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="outputs/lateint_ft")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-q", type=int, default=4000, help="train queries per dataset")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader

    model = SentenceTransformer(args.base, device=args.device)

    triples = []
    for cfg, split in TRAIN:
        triples += mine_dataset(cfg, split, model, args.max_q)
    print(f"TOTAL mined triples: {len(triples)}", flush=True)

    examples = [InputExample(texts=[a, p, n]) for a, p, n in triples]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warm = int(0.1 * len(loader) * args.epochs)
    print(f"training: {len(examples)} ex | {len(loader)} steps/epoch | {args.epochs} epochs", flush=True)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, warmup_steps=warm,
              optimizer_params={"lr": args.lr}, show_progress_bar=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nSaved fine-tuned fact encoder to {args.out}")


if __name__ == "__main__":
    main()
