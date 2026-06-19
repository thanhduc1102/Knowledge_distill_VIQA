#!/usr/bin/env python3
"""Capstone — KG-for-generator evaluation (training-free), isolating the 3 KG goals.

Consumes a ``retrieval_top3.jsonl`` (each record: question, gold answer, top-3 retrieved
docs with table+meta) and measures what the KG layer contributes WITHOUT training any
generator:

  G1 — Focus the right doc: among the top-3, does the KG evidence score rank the GOLD doc
       first? (doc-selection accuracy, and accuracy by confidence tier.)
  G2 — Offload the LLM: the symbolic ExtractiveGenerator answers straight from the KG-
       selected facts (NO LLM). Number-Match vs gold = how far the KG alone gets us.
  G3 — Transparency: every answer carries provenance (reported as a sample trace).

We restrict G2/NM to records whose gold IS in the retrieved top-3, to isolate the
generation/KG contribution from retrieval misses (which are a separate, measured axis).

Usage:  PYTHONPATH=src python scripts/research/kg_generation_eval.py \
            --top3 outputs/final_retrieval/finqa/retrieval_top3.jsonl
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, "src")

import numpy as np

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.generator import ExtractiveGenerator
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.ledger.select import select_facts
from gsr_cacl.ledger.extract import extract_ledger_from_table


def _gold_value(rec):
    g = rec.get("gold")
    if isinstance(g, (list, tuple)) and g:
        return g[0]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top3", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.top3)]
    if args.limit:
        recs = recs[: args.limit]

    gen = ExtractiveGenerator()
    n = 0
    g1_hit = 0; g1_total = 0
    by_tier = {}
    nm_kg = []; nm_naive = []
    sample_trace = None

    for rec in recs:
        q = rec.get("raw_question") or rec.get("query") or ""
        gold_id = str(rec.get("ground_truth_id") or "")
        gold_val = _gold_value(rec)
        retrieved = rec.get("retrieved") or []
        if not retrieved or gold_val is None:
            continue
        ret_ids = [str(r.get("context_id") or r.get("id") or "") for r in retrieved]
        gold_in_top3 = gold_id in ret_ids
        n += 1

        pack = build_evidence_pack(q, retrieved)

        # G1: did the KG rank the gold doc #1 among the retrieved set?
        if gold_in_top3:
            g1_total += 1
            picked = pack.ranked[0].doc_id if pack.ranked else ""
            hit = (picked == gold_id)
            g1_hit += int(hit)
            t = by_tier.setdefault(pack.tier, {"n": 0, "g1": 0, "nm": []})
            t["n"] += 1; t["g1"] += int(hit)

            # G2: extractive (no-LLM) answer from KG facts on the gold doc's ledger
            gold_rec = retrieved[ret_ids.index(gold_id)]
            led = extract_ledger_from_table(gold_rec.get("table") or gold_rec.get("page_content", ""),
                                            doc_id=gold_id, meta=gold_rec.get("meta") or {})
            facts = select_facts(q, [led], top_n=6)
            ans = gen.generate(q, "", facts=facts)
            pred = ans.replace("Answer:", "").strip()
            ok = number_match(pred, gold_val, rel_tol=1e-2)
            nm_kg.append(int(ok)); t["nm"].append(int(ok))

            # naive baseline: strongest single fact's value, no task logic
            naive = str(facts[0].value) if facts else ""
            nm_naive.append(int(number_match(naive, gold_val, rel_tol=1e-2)))

            if sample_trace is None and pack.provenance:
                sample_trace = {"q": q, "tier": pack.tier, "pred": pred, "gold": gold_val,
                                "prov": pack.provenance[:3], "verify": pack.verification}

    print(f"\n=== KG-for-generator eval — {Path(args.top3).parent.name} (N={n}) ===")
    print(f"gold-in-top3 (retrieval hit): {g1_total}/{n} = {g1_total/max(n,1):.1%}")
    print(f"\nG1  KG doc-selection acc (pick gold #1 among top-3): {g1_hit}/{g1_total} = {g1_hit/max(g1_total,1):.1%}")
    print("    by tier:")
    for tier in ("TRUST_TOP1", "PREFER_TOP1", "REVIEW"):
        t = by_tier.get(tier)
        if t and t["n"]:
            nm = np.mean(t["nm"]) if t["nm"] else 0.0
            print(f"      {tier:<12} n={t['n']:4}  g1={t['g1']/t['n']:.1%}  extractiveNM={nm:.1%}")
    print(f"\nG2  Extractive (NO-LLM) Number-Match on KG facts: {np.mean(nm_kg):.1%}  "
          f"(naive strongest-fact baseline: {np.mean(nm_naive):.1%})")
    if sample_trace:
        print(f"\nG3  sample provenance trace:")
        print(f"    Q: {sample_trace['q'][:110]}")
        print(f"    tier={sample_trace['tier']}  pred={sample_trace['pred']}  gold={sample_trace['gold']}")
        for p in sample_trace["prov"]:
            print(f"      {p['concept']} [{p['period']}] = {p['raw_text']}  @ {p['cell']}")
        v = sample_trace["verify"]
        print(f"      accounting check: {v.get('n_edges',0)} identities, {v.get('n_violated',0)} violated")


if __name__ == "__main__":
    main()
