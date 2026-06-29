#!/usr/bin/env python3
"""Learned operand attribution (the #1 research lever).

Problem (measured): CPR's *role* component is the workhorse, but it rests on the heuristic
``calculation_plan`` whose operand selection agrees with a strong oracle only ~F1 0.5, and
whose standalone answer precision is 11-24%. This caps CPR and the auditable ceiling.

Idea: learn which candidate facts are the OPERANDS of a query's computation, with DISTANT
SUPERVISION from the gold answer (no manual labels): for each query, find an operand set
that reconstructs the gold answer (``derivation.derivation_depth`` returns the operands);
those facts are positive operands. Train a scorer s(fact | query) from text-embedding +
structural features; at inference, restrict the derivation search to the top-scored facts.

Eval (5-fold CV on gold-doc ledgers, distant-supervised; the model never trains on the
query it scores):
  * operand-F1   : predicted top-k operands vs the gold-reconstructing operand set
  * derivation hit-rate (learned ceiling): can top-m learned facts reconstruct the gold?
  * vs heuristic : calculation_plan operands, and full unrestricted search (the cap).

Uses bge-small (GPU) for embeddings; logistic head. Self-contained, reuses retrieval_top3
(gold doc only, isolating attribution from retrieval).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.ledger.extract import extract_ledger_from_table, extract_ledger
from gsr_cacl.ledger.numeric import numbers_close, parse_financial_number, extract_numbers, extract_years
from gsr_cacl.research.derivation import derivation_depth
from gsr_cacl.ledger.select import calculation_plan, select_facts, infer_task_type, _tokens
from gsr_cacl.ontology.concepts import concepts_in_text

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}


def _gold_value(gold):
    if gold is None:
        return None
    if isinstance(gold, list):
        for g in gold:
            v = parse_financial_number(str(g))
            if v is not None:
                return v
        return None
    return parse_financial_number(str(gold))


def _gold_doc_ledger(rec):
    gid = str(rec.get("ground_truth_id") or rec.get("context_id") or "")
    for d in rec.get("retrieved", []):
        if str(d.get("context_id") or d.get("id")) == gid:
            tbl = d.get("table"); ctx = d.get("page_content") or d.get("context") or ""
            meta = d.get("meta") or d.get("metadata") or {}
            if tbl:
                return extract_ledger_from_table(tbl, doc_id=gid, meta=meta, caption=ctx[:200])
            return extract_ledger(context=ctx, doc_id=gid, meta=meta)
    return None


def _concept_consistency(fact, q_concepts, q_tokens):
    if fact.concept_canonical and q_concepts and fact.concept_canonical in q_concepts:
        return 1.0
    ftoks = _tokens(fact.concept) | _tokens(fact.column_header or "")
    if not ftoks:
        return 0.0
    ov = len(q_tokens & ftoks)
    return min(1.0, ov / 2.0)


def _period_int(fact):
    try:
        return int(str(fact.period)[:4]) if fact.period else None
    except (ValueError, TypeError):
        return None


def featurize(query, facts, embedder):
    """Return X[n_facts, d] and the list of fact values."""
    q_tokens = _tokens(query)
    q_concepts = concepts_in_text(query)
    q_years = set(extract_years(query))
    fstrs = [f"{f.concept} {f.period or ''}".strip() for f in facts]
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    f_emb = embedder.encode(fstrs, normalize_embeddings=True) if fstrs else np.zeros((0, q_emb.shape[0]))
    vals = [abs(f.value) for f in facts if f.value is not None]
    maxv = max(vals) if vals else 1.0
    rows = []
    for i, f in enumerate(facts):
        cos = float(np.dot(q_emb, f_emb[i])) if len(f_emb) else 0.0
        cc = _concept_consistency(f, q_concepts, q_tokens)
        p = _period_int(f)
        pm = 1.0 if (p is not None and p in q_years) else (0.5 if p is None else 0.0)
        v = abs(f.value) if f.value is not None else 0.0
        vlog = np.log1p(v) / np.log1p(maxv) if maxv > 0 else 0.0
        is_total = 1.0 if (getattr(f, "is_total", False) or "total" in (f.concept or "").lower()) else 0.0
        rows.append([cos, cc, pm, vlog, is_total, 1.0])
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))


def build_examples(ds, embedder, limit=None):
    recs = [json.loads(l) for l in open(RETR[ds]) if l.strip()]
    if limit:
        recs = recs[:limit]
    examples = []  # each: dict(X, y, vals, gold, query, heur_operands, oracle_operands)
    for rec in recs:
        query = rec.get("raw_question") or rec.get("query")
        gold = _gold_value(rec.get("gold"))
        if gold is None:
            continue
        led = _gold_doc_ledger(rec)
        if led is None:
            continue
        facts = [f for f in led.numeric_facts() if f.value is not None]
        if len(facts) < 2:
            continue
        vals = [f.value for f in facts]
        depth, operands = derivation_depth(gold, vals, rel_tol=1e-2, max_ops=3, return_operands=True)
        # distant labels: facts whose value is in the gold-reconstructing operand set
        if depth in (None,):
            continue
        if depth == "grounded":
            op_vals = [v for v in vals if numbers_close(gold, v, 1e-2)]
        else:
            op_vals = list(operands)
        y = np.array([1 if any(numbers_close(f.value, ov, 1e-2) for ov in op_vals) else 0
                      for f in facts], dtype=int)
        if y.sum() == 0:
            continue
        X = featurize(query, facts, embedder)
        # heuristic operands for comparison
        plan = calculation_plan(query, select_facts(query, [led], top_n=10))
        heur = [op.get("value") for op in (plan.get("operands") or []) if op.get("value") is not None]
        examples.append({"X": X, "y": y, "vals": vals, "gold": gold, "facts_n": len(facts),
                         "op_vals": op_vals, "heur": heur, "depth": depth})
    return examples


def _f1(pred_vals, gold_vals, tol=1e-2):
    if not pred_vals and not gold_vals:
        return 1.0
    if not pred_vals or not gold_vals:
        return 0.0
    used = [False] * len(gold_vals); tp = 0
    for p in pred_vals:
        for i, g in enumerate(gold_vals):
            if not used[i] and numbers_close(p, g, tol):
                used[i] = True; tp += 1; break
    prec = tp / len(pred_vals); rec = tp / len(gold_vals)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def run(ds, embedder, folds=5, limit=None, seed=0):
    import random
    from sklearn.linear_model import LogisticRegression
    ex = build_examples(ds, embedder, limit=limit)
    n = len(ex)
    if n < 20:
        return {"dataset": ds, "n": n, "error": "too few supervised examples"}
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    fold = {idx[i]: i % folds for i in range(n)}

    learned_f1, heur_f1 = [], []
    learned_hit, full_hit, heur_hit = 0, 0, 0
    for f in range(folds):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] != f and False] or [i for i in range(n) if fold[i] == f]
        Xtr = np.vstack([ex[i]["X"] for i in tr])
        ytr = np.concatenate([ex[i]["y"] for i in tr])
        if len(set(ytr)) < 2:
            continue
        clf = LogisticRegression(max_iter=600, C=1.0, class_weight="balanced").fit(Xtr, ytr)
        for i in te:
            e = ex[i]
            scores = clf.predict_proba(e["X"])[:, 1]
            k = max(1, len(e["op_vals"]))
            top = np.argsort(-scores)[:k]
            pred_vals = [e["vals"][j] for j in top]
            learned_f1.append(_f1(pred_vals, e["op_vals"]))
            heur_f1.append(_f1(e["heur"], e["op_vals"]))
            # derivation hit-rate: can top-m learned facts reconstruct gold?
            topm = np.argsort(-scores)[:6]
            vm = [e["vals"][j] for j in topm]
            d_learned, _ = derivation_depth(e["gold"], vm, max_ops=3, return_operands=True)
            d_full, _ = derivation_depth(e["gold"], e["vals"], max_ops=3, return_operands=True)
            d_heur, _ = derivation_depth(e["gold"], e["heur"] or [0], max_ops=3, return_operands=True)
            learned_hit += int(d_learned is not None)
            full_hit += int(d_full is not None)
            heur_hit += int(d_heur is not None)
    m = len(learned_f1)
    return {"dataset": ds, "n_supervised": n, "n_eval": m,
            "operand_F1_learned": round(float(np.mean(learned_f1)), 4),
            "operand_F1_heuristic": round(float(np.mean(heur_f1)), 4),
            "deriv_hit_learned_top6": round(learned_hit / max(m, 1), 4),
            "deriv_hit_heuristic": round(heur_hit / max(m, 1), 4),
            "deriv_hit_full_search(ceiling)": round(full_hit / max(m, 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/learned_operand/report.json")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.model, device=args.device)
    allr = {}
    for ds in args.datasets:
        t0 = time.time()
        r = run(ds, embedder, limit=args.limit)
        allr[ds] = r
        print(f"[{ds}] {json.dumps(r)}  ({time.time()-t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
