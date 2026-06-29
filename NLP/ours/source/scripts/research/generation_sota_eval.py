#!/usr/bin/env python3
"""Generator-SOTA: does structure (CPR/ledger) IMPROVE Number-Match (not just verify)?

We already cached, per query, the temp-0 answer + k=5 sampled answers (gemini_generate.py).
Candidate-selection arms (ZERO new API calls):
  * raw           : the temp-0 answer (baseline NM).
  * sc_majority   : self-consistency — modal numeric answer among {raw}∪samples.
  * cpr_select    : pick the candidate with the highest CPR structure-grounding confidence.
  * cpr_then_freq : CPR confidence, tie-broken by self-consistency frequency (structure⊕SC).
  * oracle_bestof : any candidate correct (upper bound = ceiling of candidate selection).

If cpr_select / cpr_then_freq beat raw and sc_majority on Number-Match, structure grounding
is not only a reliability signal — it actively raises end-to-end accuracy by choosing the
grounded answer. Reuses the union Fact-Ledger from retrieval_top3 (same as the pipeline).
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, numbers_close
from gsr_cacl.research.cpr_verifier import verify_cpr

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}


def _union(pack):
    if not pack.ranked:
        return None
    m = FactLedger(doc_id="u", facts=list(pack.ranked[0].ledger.facts), meta=dict(pack.ranked[0].ledger.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _modal(cands, tol=1e-2):
    vals = [extract_final_number(c) for c in cands]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    best, bc = None, 0
    for v in vals:
        c = sum(1 for u in vals if numbers_close(u, v, tol))
        if c > bc:
            best, bc = v, c
    return best


def run(ds, pred_path):
    retr = {str(r.get("query_id")): r for r in (json.loads(l) for l in open(RETR[ds]) if l.strip())}
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    arms = {k: 0 for k in ("raw", "sc_majority", "cpr_select", "cpr_then_freq", "oracle_bestof")}
    n = 0
    for p in preds:
        rec = retr.get(str(p.get("query_id")))
        if not rec or not rec.get("retrieved"):
            continue
        q = p.get("query") or p.get("question"); gold = p.get("gold")
        raw = p.get("raw_pred") or ""
        cands = [raw] + list(p.get("sc_samples") or [])
        cand_vals = [extract_final_number(c) for c in cands]
        # need numeric candidates
        idx_num = [i for i, v in enumerate(cand_vals) if v is not None]
        if not idx_num:
            n += 1
            continue
        n += 1
        pack = build_evidence_pack(q, rec["retrieved"], top_n_facts=8)
        led = _union(pack)
        # arm: raw
        arms["raw"] += int(number_match(raw, gold))
        # arm: sc majority
        mv = _modal(cands)
        arms["sc_majority"] += int(mv is not None and number_match(mv, gold))
        # frequency map (for tie-break)
        freq = Counter()
        for v in cand_vals:
            if v is not None:
                for u in list(freq):
                    if numbers_close(u, v, 1e-2):
                        freq[u] += 1; break
                else:
                    freq[v] += 1
        # CPR confidence per unique candidate
        scored = []
        seen = []
        for i in idx_num:
            v = cand_vals[i]
            if any(numbers_close(v, s, 1e-2) for s in seen):
                continue
            seen.append(v)
            conf = verify_cpr(cands[i], led, q, selected_facts=pack.selected_facts).confidence if led else 0.0
            fr = max((c for u, c in freq.items() if numbers_close(u, v, 1e-2)), default=1)
            scored.append((v, conf, fr))
        if scored:
            best_cpr = max(scored, key=lambda x: x[1])[0]
            arms["cpr_select"] += int(number_match(best_cpr, gold))
            best_cf = max(scored, key=lambda x: (x[1], x[2]))[0]
            arms["cpr_then_freq"] += int(number_match(best_cf, gold))
        arms["oracle_bestof"] += int(any(number_match(c, gold) for c in cands))
    return {"dataset": ds, "n": n, **{k: round(v / max(n, 1), 4) for k, v in arms.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/gemini_gen/{ds}_predictions.jsonl")
    ap.add_argument("--out", default="outputs/research/generation_sota/report.json")
    args = ap.parse_args()
    allr = {}
    for ds in args.datasets:
        pp = args.pred.format(ds=ds)
        if not Path(pp).exists():
            continue
        r = run(ds, pp); allr[ds] = r
        print(f"{ds}: raw={r['raw']} sc_maj={r['sc_majority']} cpr_select={r['cpr_select']} "
              f"cpr+freq={r['cpr_then_freq']} | oracle={r['oracle_bestof']} (n={r['n']})")
    # weighted avg
    if allr:
        W = {"finqa": 1147, "convfinqa": 3458, "tatqa": 1144}
        for arm in ("raw", "sc_majority", "cpr_select", "cpr_then_freq", "oracle_bestof"):
            num = sum(allr[d][arm] * W[d] for d in allr); den = sum(W[d] for d in allr)
            print(f"  W.Avg {arm:14s} = {num/den:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
