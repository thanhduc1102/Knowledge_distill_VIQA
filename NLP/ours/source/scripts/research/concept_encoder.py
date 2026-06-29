#!/usr/bin/env python3
"""Learned concept encoder (annotation-free) — generalize concept typing beyond the
curated ontology (which exact-matches only ~22-31% of line-item labels).

Idea: embed line-item strings with a sentence encoder; each canonical concept is an
ANCHOR = mean embedding of its alias phrases (supervision is the ontology itself, no manual
labels). A line-item maps to the nearest anchor; cos >= tau => typed. This replaces brittle
token-overlap concept-consistency with semantic matching that handles unseen phrasings.

Evals:
  1. INTRINSIC (does the encoder generalize?): 5-fold over the alias set — hold out aliases,
     predict their concept from anchors built on the rest; top-1 accuracy.
  2. SEMANTIC COVERAGE: of corpus line-items NOT exact-matched by the ontology, what fraction
     get a confident (cos>=tau) concept assignment -> coverage extension.
Optional contrastive fine-tune of a projection head on alias<->concept pairs.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from gsr_cacl.ontology.concepts import CONCEPT_ALIASES, canonical_concept
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.ledger.fact import FactLedger

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}


def _union(pack):
    if not pack.ranked:
        return None
    m = FactLedger(doc_id="u", facts=list(pack.ranked[0].ledger.facts), meta={})
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def intrinsic(embedder, folds=5, seed=0):
    """5-fold over aliases: predict held-out alias's concept from anchors on the rest."""
    pairs = [(a, c) for c, als in CONCEPT_ALIASES.items() for a in als]
    rng = random.Random(seed); rng.shuffle(pairs)
    concepts = list(CONCEPT_ALIASES.keys())
    aliases = [a for a, _ in pairs]; labels = [c for _, c in pairs]
    emb = embedder.encode(aliases, normalize_embeddings=True)
    fold = {i: i % folds for i in range(len(pairs))}
    correct = 0; tot = 0
    for f in range(folds):
        tr = [i for i in range(len(pairs)) if fold[i] != f]
        te = [i for i in range(len(pairs)) if fold[i] == f]
        # anchors from training aliases
        anc = {}
        for c in concepts:
            idx = [i for i in tr if labels[i] == c]
            if idx:
                anc[c] = np.mean([emb[i] for i in idx], axis=0)
        if not anc:
            continue
        ac = list(anc.keys()); am = np.stack([anc[c] for c in ac])
        am = am / (np.linalg.norm(am, axis=1, keepdims=True) + 1e-9)
        for i in te:
            sims = am @ emb[i]
            pred = ac[int(np.argmax(sims))]
            correct += int(pred == labels[i]); tot += 1
    return round(correct / max(tot, 1), 4), tot


def semantic_coverage(embedder, ds, tau=0.55, limit=200):
    recs = [json.loads(l) for l in open(RETR[ds]) if l.strip()][:limit]
    # anchors from full ontology
    concepts = list(CONCEPT_ALIASES.keys())
    anc = np.stack([np.mean(embedder.encode(CONCEPT_ALIASES[c], normalize_embeddings=True), axis=0)
                    for c in concepts])
    anc = anc / (np.linalg.norm(anc, axis=1, keepdims=True) + 1e-9)
    labels = set()
    for r in recs:
        pack = build_evidence_pack(r.get("query", ""), r.get("retrieved", []), top_n_facts=8)
        led = _union(pack)
        if not led:
            continue
        for f in led.numeric_facts():
            lab = (f.concept or "").strip()
            if lab:
                labels.add(lab)
    labels = list(labels)
    exact = [l for l in labels if canonical_concept(l)]
    unmatched = [l for l in labels if not canonical_concept(l)]
    if unmatched:
        em = embedder.encode(unmatched, normalize_embeddings=True)
        sims = em @ anc.T
        semantic_hit = int(np.sum(sims.max(axis=1) >= tau))
    else:
        semantic_hit = 0
    total = len(labels)
    return {"dataset": ds, "n_labels": total,
            "exact_coverage": round(len(exact) / max(total, 1), 4),
            "semantic_extra": round(semantic_hit / max(total, 1), 4),
            "combined_coverage": round((len(exact) + semantic_hit) / max(total, 1), 4),
            "tau": tau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--out", default="outputs/research/concept_encoder/report.json")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.model, device=args.device)
    acc, n = intrinsic(emb)
    print(f"INTRINSIC alias->concept top-1 (5-fold, n={n}): {acc}")
    res = {"intrinsic_alias_concept_top1": acc, "intrinsic_n": n, "coverage": {}}
    for ds in args.datasets:
        c = semantic_coverage(emb, ds, tau=args.tau)
        res["coverage"][ds] = c
        print(f"{ds}: exact={c['exact_coverage']} +semantic={c['semantic_extra']} "
              f"= combined={c['combined_coverage']} (tau={args.tau})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
