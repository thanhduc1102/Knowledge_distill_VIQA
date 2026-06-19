"""End-to-end LEDGER-RAG evaluation harness.

Stages (run any subset):
  retrieval  — metadata-aware LedgerRetrieval → MRR@3 / Recall@k / NDCG@3
  generation — Fact-Ledger fact selection → generator → Number-Match + verifier rewards

Everything is saved under ``outputs/<run_name>/`` for full reproducibility:
  config.json, retrieval_metrics.json, retrieval_topk.jsonl,
  generation_metrics.json, predictions.jsonl, summary.json

Usage::
  python -m gsr_cacl.eval.pipeline --dataset finqa --split test --sample 200 \
      --stage all --generator extractive
  python -m gsr_cacl.eval.pipeline --dataset finqa --sample 200 \
      --stage generation --generator hf --gen-model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset, DatasetDict

from gsr_cacl.core import Document, RetrievalResult
from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg
from gsr_cacl.generation import build_generator, verify, compute_number_match
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ledger_eval")

DATASET_CONFIGS = {
    "finqa": {"config_name": "FinQA", "eval_split": "test"},
    "convfinqa": {"config_name": "ConvFinQA", "eval_split": "turn_0"},
    "tatqa": {"config_name": "TAT-DQA", "eval_split": "test"},
}
HF_NAME = "G4KMU/t2-ragbench"


# ---------------------------------------------------------------------------
# Data loading (carries table + gold answers, which DatasetSplit drops)
# ---------------------------------------------------------------------------

def _meta(row) -> dict:
    return {k: str(row.get(k, "")) for k in
            ("company_name", "report_year", "company_sector", "company_industry", "company_symbol")}


def load_eval_data(dataset: str, split: str | None, sample_size: int | None):
    cfg = DATASET_CONFIGS[dataset]
    split = split or cfg["eval_split"]
    qa_df = load_dataset(HF_NAME, cfg["config_name"], split=split).to_pandas()
    if sample_size and sample_size < len(qa_df):
        qa_df = qa_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    full = load_dataset(HF_NAME, cfg["config_name"])
    parts = [full[s].to_pandas() for s in full.keys()] if isinstance(full, DatasetDict) else [full.to_pandas()]
    corpus_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="context_id")

    corpus, tables = [], []
    ctx_to_table = {}
    for _, r in corpus_df.iterrows():
        cid = str(r.get("context_id", r.get("id", "")))
        tbl = str(r.get("table", "") or "")
        corpus.append(Document(page_content=str(r.get("context", "") or ""), meta_data=_meta(r), id=cid))
        tables.append(tbl)
        ctx_to_table[cid] = tbl

    queries, gt_ids, metas, golds = [], [], [], []
    for _, r in qa_df.iterrows():
        q = str(r.get("question", ""))
        if r.get("company_name"):
            q = f"{r.get('company_name')}: {q}"
        queries.append(q)
        gt_ids.append(str(r.get("context_id", r.get("id", ""))))
        metas.append(_meta(r))
        golds.append([r.get("program_answer"), r.get("original_answer")])
    return dict(corpus=corpus, tables=tables, ctx_to_table=ctx_to_table,
                queries=queries, gt_ids=gt_ids, metas=metas, golds=golds, name=cfg["config_name"])


# ---------------------------------------------------------------------------
# Retrieval stage
# ---------------------------------------------------------------------------

def stage_retrieval(data, args, out_dir: Path) -> list[list]:
    import torch
    from gsr_cacl.benchmark_gsr import SentenceTransformerEmbeddingFunction
    from gsr_cacl.methods.ledger_retrieval import LedgerRetrieval
    from gsr_cacl.entity.train import train_entity_embedder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ent = None
    if args.use_entity:
        logger.info("Training entity embedder on corpus metadata ...")
        ent = train_entity_embedder([d.meta_data for d in data["corpus"]],
                                    epochs=args.entity_epochs, device=device)
    emb = SentenceTransformerEmbeddingFunction(args.embed_model, device=device)
    ret = LedgerRetrieval(
        data["corpus"], emb, entity_embedder=ent, tables=data["tables"], top_k=max(args.top_k, 5),
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        use_metadata_filter=args.metadata_filter, device=device,
    )

    results, scored_all = [], []
    for q, m, gt in zip(data["queries"], data["metas"], data["gt_ids"]):
        scored = ret._rank_candidates(q, m)
        scored_all.append(scored)
        results.append(RetrievalResult(query=q, retrieved_docs=[d for d, _ in scored],
                                       ground_truth_id=gt, meta_data=m))

    metrics = {"MRR@3": compute_mrr(results, 3), "Recall@1": compute_recall(results, 1),
               "Recall@3": compute_recall(results, 3), "Recall@5": compute_recall(results, 5),
               "NDCG@3": compute_ndcg(results, 3), "n_queries": len(results)}
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Retrieval metrics: %s", metrics)

    # persist top-k + generator-ready evidence
    with open(out_dir / "retrieval_topk.jsonl", "w") as f:
        for scored, q, m, gt, gold in zip(scored_all, data["queries"], data["metas"],
                                          data["gt_ids"], data["golds"]):
            top = scored[: args.top_k]
            retrieved = [{"id": str(d.id), "score": s, "meta": d.meta_data,
                          "table": data["ctx_to_table"].get(str(d.id), ""),
                          "page_content": d.page_content} for d, s in top]
            pack = build_evidence_pack(q, retrieved, top_n_facts=args.fact_top_n)
            evidence = render_prompt_context(pack)
            f.write(json.dumps({
                "query": q, "query_meta": m, "ground_truth_id": gt, "gold": gold,
                "retrieved": retrieved,
                "evidence_block": evidence,
                "evidence_pack": pack.to_dict(),
            }) + "\n")
    return metrics, str(out_dir / "retrieval_topk.jsonl")


# ---------------------------------------------------------------------------
# Generation stage
# ---------------------------------------------------------------------------

def stage_generation(retrieval_jsonl: str, args, out_dir: Path) -> dict:
    import torch

    records = [json.loads(l) for l in open(retrieval_jsonl)]
    if args.generator == "hf":
        gen_device = args.gen_device or ("auto" if torch.cuda.is_available() else "cpu")
        gen_dtype = args.gen_dtype or ("float16" if torch.cuda.is_available() else "float32")
        gen_kwargs = {
            "model_name": args.gen_model,
            "device": gen_device,
            "dtype": gen_dtype,
            "max_new_tokens": args.gen_max_new_tokens,
            "temperature": args.gen_temperature,
        }
    else:
        gen_kwargs = {}
    generator = build_generator(args.generator, **gen_kwargs)
    logger.info("Generator: %s", generator.name)

    def _get_retrieved(rec: dict) -> list[dict]:
        """Support both current and legacy retrieval JSONL schemas."""
        if "retrieved" in rec:
            return rec["retrieved"]
        if "retrieved_docs" in rec:
            return [
                {
                    "id": d.get("id", ""),
                    "score": d.get("score", 0.0),
                    "meta": d.get("metadata", d.get("meta", {})),
                    "table": d.get("table", ""),
                    "page_content": d.get("page_content", ""),
                }
                for d in rec["retrieved_docs"]
            ]
        raise KeyError("retrieved")

    preds, golds, rewards, grounded, dump = [], [], [], [], []
    t0 = time.time()
    batch_size = max(1, getattr(args, "gen_batch_size", 1))

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        retrieved_batch = [_get_retrieved(rec) for rec in batch]
        packs = [
            build_evidence_pack(rec["query"], retrieved, top_n_facts=args.fact_top_n)
            for rec, retrieved in zip(batch, retrieved_batch)
        ]
        ledgers_batch = [[d.ledger for d in pack.ranked] for pack in packs]
        facts_batch = [pack.selected_facts for pack in packs]
        evidence_batch = [render_prompt_context(pack) for pack in packs]
        meta_batch = [rec.get("query_meta", rec.get("meta", {})) for rec in batch]
        query_batch = [rec["query"] for rec in batch]
        pred_batch = generator.generate_batch(query_batch, evidence_batch, meta_batch, facts_batch)

        for rec, ledgers, evidence, pred, pack in zip(batch, ledgers_batch, evidence_batch, pred_batch, packs):
            union = ledgers[0] if ledgers else None
            if union is not None:
                for lg in ledgers[1:]:
                    union.facts.extend(lg.facts)
            vr = verify(pred, union, rec["query"], gold=rec.get("gold")) if union else None
            preds.append(pred)
            golds.append(rec.get("gold"))
            rewards.append(vr.reward if vr else 0.0)
            grounded.append(vr.grounded if vr else False)
            dump.append({
                "query": rec["query"],
                "gold": rec.get("gold"),
                "prediction": pred,
                "pred_value": vr.pred_value if vr else None,
                "reward": vr.reward if vr else 0.0,
                "grounded": vr.grounded if vr else False,
                "explanation": vr.explanation if vr else "",
                "evidence_block": evidence,
                "evidence_pack": pack.to_dict(),
            })

        if len(preds) % 25 == 0:
            logger.info("  generated %d/%d", len(preds), len(records))

    nm = compute_number_match(preds, golds)
    metrics = {
        **nm,
        "mean_verifier_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "grounded_rate": sum(grounded) / len(grounded) if grounded else 0.0,
        "generator": generator.name,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "generation_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "predictions.jsonl", "w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    logger.info("Generation metrics: %s", metrics)
    return metrics

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="LEDGER-RAG end-to-end evaluation")
    p.add_argument("--dataset", default="finqa", choices=list(DATASET_CONFIGS))
    p.add_argument("--split", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--stage", default="all", choices=["retrieval", "generation", "all"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--fact-top-n", type=int, default=6)
    # retrieval
    p.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--use-entity", action="store_true", default=True)
    p.add_argument("--no-entity", dest="use_entity", action="store_false")
    p.add_argument("--entity-epochs", type=int, default=12)
    p.add_argument("--metadata-filter", action="store_true", default=True)
    p.add_argument("--no-metadata-filter", dest="metadata_filter", action="store_false")
    # generation
    p.add_argument("--generator", default="extractive", choices=["extractive", "hf"])
    p.add_argument("--gen-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--gen-device", default=None, help="HF generator device override (cpu, auto, cuda)")
    p.add_argument("--gen-dtype", default=None, help="HF generator dtype override (float16, bfloat16, float32)")
    p.add_argument("--gen-max-new-tokens", type=int, default=16)
    p.add_argument("--gen-temperature", type=float, default=0.0)
    p.add_argument("--gen-batch-size", type=int, default=4)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--retrieval-jsonl", default=None, help="reuse an existing retrieval file")
    args = p.parse_args()

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run = args.out_dir or f"outputs/ledger_eval/{args.dataset}_{args.stage}_{ts}"
    out_dir = Path(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    summary = {"dataset": args.dataset, "stage": args.stage}
    retrieval_jsonl = args.retrieval_jsonl

    if args.stage in ("retrieval", "all"):
        data = load_eval_data(args.dataset, args.split, args.sample)
        rmetrics, retrieval_jsonl = stage_retrieval(data, args, out_dir)
        summary["retrieval"] = rmetrics

    if args.stage in ("generation", "all"):
        if retrieval_jsonl is None:
            raise SystemExit("generation stage needs --retrieval-jsonl (or run --stage all)")
        summary["generation"] = stage_generation(retrieval_jsonl, args, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done. Outputs in %s", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
