#!/usr/bin/env python3
"""Retrieval ablation: isolate the contribution of each signal on T²-RAGBench.

Saves a JSON + markdown table under outputs/ledger_eval/<dataset>_ablation/.
All arms SHARE the same document embeddings so differences are purely from the
entity-embedding signal and the metadata-aware candidate construction.

Usage:
  python scripts/retrieval_ablation.py --dataset finqa --sample 200
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    import torch
    from datasets import load_dataset
    from gsr_cacl.datasets.wrappers import load_t2ragbench_split
    from gsr_cacl.benchmark_gsr import (SentenceTransformerEmbeddingFunction,
                                        compute_mrr, compute_recall, compute_ndcg)
    from gsr_cacl.methods.ledger_retrieval import LedgerRetrieval
    from gsr_cacl.entity.train import train_entity_embedder
    from gsr_cacl.core import RetrievalResult

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finqa", choices=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = {"finqa": ("FinQA", "test"), "convfinqa": ("ConvFinQA", "turn_0"),
           "tatqa": ("TAT-DQA", "test")}[args.dataset]
    split = args.split or cfg[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_t2ragbench_split(cfg[0], split=split, sample_size=args.sample)
    metas = [d.meta_data for d in data.corpus]
    ent = train_entity_embedder(metas, epochs=12, device=device)
    emb = SentenceTransformerEmbeddingFunction(args.embed_model, device=device)
    ret = LedgerRetrieval(data.corpus, emb, entity_embedder=ent, top_k=5, device=device)

    def evaluate(alpha, beta, gamma, filt):
        ret.alpha, ret.beta, ret.gamma, ret.use_metadata_filter = alpha, beta, gamma, filt
        res = []
        for q, m, gt in zip(data.queries, data.meta_data, data.ground_truth_ids):
            docs = [d for d, _ in ret._rank_candidates(q, m)]
            res.append(RetrievalResult(query=q, retrieved_docs=docs, ground_truth_id=gt, meta_data=m))
        return {"MRR@3": compute_mrr(res, 3), "Recall@1": compute_recall(res, 1),
                "Recall@3": compute_recall(res, 3), "Recall@5": compute_recall(res, 5),
                "NDCG@3": compute_ndcg(res, 3)}

    arms = {
        "dense_only": (1.0, 0.0, 0.0, False),
        "dense+equationCS": (1.0, 0.0, 0.2, False),
        "dense+entity_rerank": (1.0, 0.6, 0.0, False),
        "FULL (entity+metadata_filter)": (1.0, 0.6, 0.2, True),
    }
    results = {name: evaluate(*p) for name, p in arms.items()}

    out_dir = Path(args.out or f"outputs/ledger_eval/{args.dataset}_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": cfg[0], "split": split, "n_queries": len(data.queries),
               "n_corpus": len(data.corpus), "embed_model": args.embed_model, "arms": results}
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2))

    lines = [f"# Retrieval ablation — {cfg[0]} (n={len(data.queries)}, corpus={len(data.corpus)})", "",
             "| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |", "|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['MRR@3']:.3f} | {m['Recall@1']:.3f} | {m['Recall@3']:.3f} "
                     f"| {m['Recall@5']:.3f} | {m['NDCG@3']:.3f} |")
    (out_dir / "ablation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
