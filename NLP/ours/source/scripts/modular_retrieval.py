#!/usr/bin/env python3
"""Modular Multi-Expert Retrieval (MMER) — independent experts + learned fusion.

Pipeline (honest: company prefix stripped, years/company/concepts from the QUESTION):
  1. Build experts independently; each prepares its own doc representation.
  2. Candidate pool = Lexical(BM25) top-`pool` (∪ Dense top-`pool` if --with-dense).
  3. Per query: each expert scores the pool → min-max per column → matrix F[pool×k].
  4. STANDALONE eval: rank the pool by each single expert (each method measured on its own).
  5. Train three fusion heads (linear / mlp / gate) listwise-InfoNCE on a TRAIN query split;
     report all of them + standalone experts on the held-out TEST split.

Usage:  PYTHONPATH=src python scripts/modular_retrieval.py --dataset FinQA [--with-dense]
"""
from __future__ import annotations

import argparse, json, os, re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.experts.base import minmax
from gsr_cacl.experts.lexical import LexicalExpert, _toks
from gsr_cacl.experts.local_lexical import LocalLexicalExpert
from gsr_cacl.experts.concept import ConceptExpert
from gsr_cacl.experts.cell import CellExpert
from gsr_cacl.experts.entity import EntityExpert
from gsr_cacl.experts.graph import GraphExpert
from gsr_cacl.experts.factgate import FactGateExpert
from gsr_cacl.experts.meta_retriever import MetadataRetriever
from gsr_cacl.experts.fusion import FusionData, train_fusion, rank_scores
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.ontology.concepts import concepts_in_text

SPLITS = {"FinQA": "test", "ConvFinQA": "turn_0", "TAT-DQA": "test"}


def make_experts(spec: str, device: str, lateint_model: str | None = None,
                 company_pool: bool = False, meta_max_add: int = 15,
                 meta_provided: bool = False):
    """Build the requested experts. ``spec`` is a comma list of expert names."""
    want = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for nm in want:
        if nm == "lexical":
            out.append(LexicalExpert(abbr_expand=True))
        elif nm == "loclex":
            out.append(LocalLexicalExpert(abbr_expand=True))
        elif nm == "factlevel":
            from gsr_cacl.experts.fact_level import FactLevelExpert
            out.append(FactLevelExpert(device=device))
        elif nm == "entity":
            out.append(EntityExpert(device=device))
        elif nm == "concept":
            out.append(ConceptExpert())
        elif nm == "cell":
            out.append(CellExpert())
        elif nm == "graph":
            out.append(GraphExpert())
        elif nm == "factgate":
            out.append(FactGateExpert())
        elif nm == "meta":
            out.append(MetadataRetriever(company_pool=company_pool, max_add=meta_max_add,
                                         use_query_meta_company=(True if meta_provided else None)))
        elif nm == "dense":
            from gsr_cacl.experts.dense import DenseExpert
            out.append(DenseExpert(device=device))
        elif nm == "lateint":
            from gsr_cacl.experts.late_interaction import LateInteractionExpert
            kw = {"model_name": lateint_model} if lateint_model else {}
            out.append(LateInteractionExpert(device=device, **kw))
        else:
            raise ValueError(f"unknown expert: {nm}")
    return out


def rank_of_gold(order_idx: np.ndarray, gold_pos: int) -> int:
    """Position of gold in a ranked order of pool positions (∞ if gold absent)."""
    if gold_pos < 0:
        return 10**9
    hit = np.where(order_idx == gold_pos)[0]
    return int(hit[0]) if hit.size else 10**9


def metrics_from_ranks(ranks: list[int]) -> dict:
    mrr3 = np.mean([1.0 / (r + 1) if r < 3 else 0.0 for r in ranks])
    return {
        "MRR@3": round(float(mrr3), 4),
        "R@1": round(float(np.mean([r < 1 for r in ranks])), 4),
        "R@3": round(float(np.mean([r < 3 for r in ranks])), 4),
        "R@5": round(float(np.mean([r < 5 for r in ranks])), 4),
        "NDCG@3": round(float(np.mean([1.0 / np.log2(r + 2) if r < 3 else 0.0 for r in ranks])), 4),
    }


def build(dataset: str, sample: int, pool_size: int, expert_spec: str, device: str,
          lateint_model=None, company_pool: bool = False, meta_max_add: int = 15,
          meta_provided: bool = False):
    split = SPLITS[dataset]
    data = load_t2ragbench_split(dataset, split=split, sample_size=(sample or None))
    corpus, gts, metas = data.corpus, data.ground_truth_ids, data.meta_data
    raw_q = []
    for q, m in zip(data.queries, metas):
        c = str(m.get("company_name", "")).strip()
        raw_q.append(q[len(c) + 1:].lstrip() if c and q.startswith(c + ":") else q)
    id2idx = {str(d.id): i for i, d in enumerate(corpus)}
    gold_idx = [id2idx.get(str(g), -1) for g in gts]
    doc_metas = [dict(d.meta_data) if isinstance(d.meta_data, dict) else {} for d in corpus]

    # ---- experts (independent) -------------------------------------------------
    experts = make_experts(expert_spec, device, lateint_model,
                           company_pool=company_pool, meta_max_add=meta_max_add,
                           meta_provided=meta_provided)
    for ex in experts:
        print(f"  prepare {ex.name} ...", flush=True)
        ex.prepare(corpus, doc_metas)
        ex.set_queries(raw_q, metas)
    names = [ex.name for ex in experts]
    retrievers = [ex for ex in experts if ex.is_retriever]
    # In company-pool mode the pool is the company-complete set: only MetadataRetriever
    # seeds it; every other expert (incl. lexical/loclex) re-scores within the company.
    if company_pool:
        seeders = [ex for ex in experts if ex.name == "meta"]
        if not seeders:
            raise ValueError("--company-pool requires the 'meta' expert in --experts")
    else:
        seeders = retrievers

    # ---- candidate pool per query (HONEST: never inject gold) ------------------
    # Pool = seeder candidates only. Gold may be absent → that query is a genuine miss
    # (the recall ceiling). Training later uses only queries whose gold is in-pool;
    # test eval counts an out-of-pool gold as rank ∞.
    pools = []
    for qi in range(len(raw_q)):
        cand = set()
        for ex in seeders:
            if hasattr(ex, "get_candidates"):
                # Expert controls its own candidate selection (e.g. MetadataRetriever
                # adds only genuinely-matching docs, ignoring pool_size to avoid noise).
                cand.update(ex.get_candidates(qi, pool_size))
            else:
                cand.update(int(j) for j in np.argsort(-ex.full_scores(qi))[:pool_size])
        pools.append(sorted(cand))

    # ---- expert score matrices (min-max per column) ---------------------------
    feats, gold_pos = [], []
    for qi in range(len(raw_q)):
        pool = pools[qi]
        cols = [minmax(ex.score_pool(qi, pool)) for ex in experts]
        feats.append(np.stack(cols, axis=1))         # [pool, k]
        gp = pool.index(gold_idx[qi]) if gold_idx[qi] in pool else -1
        gold_pos.append(gp)

    # ---- query discriminativeness features φ(Q) for the gate ------------------
    # Independent of expert selection: doc periods extracted here; company via a CompanyIndex.
    from gsr_cacl.retrieval.self_query import CompanyIndex
    from gsr_cacl.experts.concept import _doc_concepts
    comp_index = CompanyIndex(doc_metas)
    doc_periods = [p for _, p in (_doc_concepts(d.page_content) for d in corpus)]
    qfeats = []
    for qi in range(len(raw_q)):
        pool = pools[qi]
        qy = set(extract_years(raw_q[qi]))
        has_company = bool(comp_index.detect(raw_q[qi]))
        n_con = len(concepts_in_text(raw_q[qi]))
        qlen = len(_toks(raw_q[qi]))
        yr_match = np.mean([1.0 if (doc_periods[i] & qy) else 0.0 for i in pool]) if qy else 0.0
        con_col = names.index("concept") if "concept" in names else None
        con_hit = float(np.mean(feats[qi][:, con_col] >= 0.6)) if con_col is not None else 0.0
        qfeats.append([1.0 if qy else 0.0, 1.0 if has_company else 0.0,
                       min(n_con / 3.0, 1.0), min(qlen / 30.0, 1.0),
                       float(yr_match), con_hit, 1.0])
    qfeats = np.asarray(qfeats, dtype=np.float64)

    return raw_q, names, feats, gold_pos, qfeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FinQA", choices=list(SPLITS))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--experts", default="lexical,entity,concept,cell,graph,factgate")
    ap.add_argument("--cv", type=int, default=5, help="k-fold CV (0/1 = single 50/50 split)")
    ap.add_argument("--lateint-model", default=None, help="path/name for the lateint fact encoder")
    ap.add_argument("--company-pool", action="store_true",
                    help="Seed pool from MetadataRetriever company-complete set (recall→1.0); "
                         "all other experts re-score within the company.")
    ap.add_argument("--meta-max-add", type=int, default=15,
                    help="Cap on docs MetadataRetriever adds (use ~200 with --company-pool).")
    ap.add_argument("--meta-provided", action="store_true",
                    help="meta expert uses PROVIDED query metadata (company_name/report_year) "
                         "instead of detecting from the question — the leaderboard 'metadata-aware' setting.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/modular")
    args = ap.parse_args()

    print(f"### {args.dataset}  (experts={args.experts}) "
          f"{'[company-pool]' if args.company_pool else ''} ###", flush=True)
    raw_q, names, feats, gold_pos, qfeats = build(
        args.dataset, args.sample, args.pool, args.experts, args.device, args.lateint_model,
        company_pool=args.company_pool, meta_max_add=args.meta_max_add,
        meta_provided=args.meta_provided)
    Q = len(raw_q)
    data = FusionData(feats=feats, gold_pos=gold_pos, qfeats=qfeats, expert_names=names)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(Q)
    nfold = max(args.cv, 1)
    folds = [perm[f::nfold] for f in range(nfold)] if nfold > 1 else None

    def eval_order_all(order_fn):
        ranks = [rank_of_gold(order_fn(qi), gold_pos[qi]) for qi in range(Q)]
        return metrics_from_ranks(ranks)

    # standalone experts — evaluated on ALL queries (no training needed)
    results = {"n_queries": Q, "experts": {}, "fusion": {}, "cv": nfold}
    for j, nm in enumerate(names):
        results["experts"][nm] = eval_order_all(lambda qi, j=j: np.argsort(-feats[qi][:, j]))

    # learned fusion — k-fold CV: each query ranked by a model NOT trained on its fold.
    weight_acc = {nm: [] for nm in names}
    for kind in ("linear", "mlp", "gate"):
        order_cache: dict[int, np.ndarray] = {}
        if folds is None:
            tr = perm[: Q // 2]
            model = train_fusion(kind, data, tr, device=args.device, seed=args.seed)
            eval_idx = perm[Q // 2:]
            for qi in eval_idx:
                order_cache[qi] = np.argsort(-rank_scores(model, data, qi, args.device))
            ranks = [rank_of_gold(order_cache[qi], gold_pos[qi]) for qi in eval_idx]
            results["fusion"][kind] = metrics_from_ranks(ranks)
        else:
            for f in range(nfold):
                te = folds[f]
                tr = np.concatenate([folds[g] for g in range(nfold) if g != f])
                model = train_fusion(kind, data, tr, device=args.device, seed=args.seed)
                if kind == "linear":
                    for i, w in enumerate(model.weights()):
                        weight_acc[names[i]].append(float(w))
                for qi in te:
                    order_cache[qi] = np.argsort(-rank_scores(model, data, qi, args.device))
            ranks = [rank_of_gold(order_cache[qi], gold_pos[qi]) for qi in range(Q)]
            results["fusion"][kind] = metrics_from_ranks(ranks)
    results["linear_weights"] = {nm: round(float(np.mean(v)), 3) for nm, v in weight_acc.items() if v}

    pool_recall = round(float(np.mean([gp >= 0 for gp in gold_pos])), 4)
    avg_pool = round(float(np.mean([f.shape[0] for f in feats])), 1)
    results["pool_recall"] = pool_recall
    results["avg_pool_size"] = avg_pool

    # ---- print ----------------------------------------------------------------
    print(f"\npool recall = {pool_recall} | avg pool = {avg_pool} | CV={nfold}-fold | experts={names}")
    print(f"\n{'method':<26}{'MRR@3':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'NDCG@3':>8}")
    print("-- standalone experts (all queries) --")
    for nm in names:
        m = results["experts"][nm]
        print(f"{nm:<26}{m['MRR@3']:>8}{m['R@1']:>8}{m['R@3']:>8}{m['R@5']:>8}{m['NDCG@3']:>8}")
    print(f"-- learned fusion ({nfold}-fold CV, full set) --")
    for kind in ("linear", "mlp", "gate"):
        m = results["fusion"][kind]
        print(f"fusion:{kind:<19}{m['MRR@3']:>8}{m['R@1']:>8}{m['R@3']:>8}{m['R@5']:>8}{m['NDCG@3']:>8}")
    print(f"\nlinear weights (CV-avg): {results['linear_weights']}")

    outdir = Path(args.out, args.dataset.lower())
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "modular.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {outdir}/modular.json")


if __name__ == "__main__":
    main()
