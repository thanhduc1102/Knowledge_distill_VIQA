#!/usr/bin/env python3
"""Does structure-grounded verify-then-reask improve end-to-end Number-Match with a
STRONG generator? (Accuracy axis, complementary to the reliability study.)

Policy: keep the raw (table-context) answer when CPR says it is grounded; otherwise
re-ask Gemini with the structure/KG evidence (selected facts, provenance, structure
paths) — with the deterministic symbolic answer stripped, so we measure whether
*structure helps the generator reason*, not whether a symbolic answer replaces it.

Reports raw NM, verify-then-reask NM, and NM specifically on the re-asked subset.
Reuses the cached raw answers; only the low-CPR subset incurs new Gemini calls.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.research.cpr_verifier import verify_cpr
from gsr_cacl.utils.gemini_client import GeminiClient

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}
_STRIP = ("KG_SYMBOLIC_ANSWER:", "KG_SYMBOLIC_CONFIDENCE:", "KG_CALCULATION_TRACE:")
REASK_SYSTEM = ("You are a precise financial QA model. Use the provided selected facts, cell "
                "provenance, and structure evidence paths to compute the answer. Do not invent "
                "values. Return one line only: Answer: <number>.")


def _union(pack):
    if not pack.ranked:
        return None
    m = FactLedger(doc_id="u", facts=list(pack.ranked[0].ledger.facts), meta=dict(pack.ranked[0].ledger.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _kg_block(pack):
    out, skip = [], False
    for line in render_prompt_context(pack).splitlines():
        s = line.strip()
        if s.startswith(_STRIP):
            skip = s.startswith("KG_CALCULATION_TRACE:"); continue
        if skip and s.startswith(("-", "[", "{")):
            continue
        skip = False; out.append(line)
    return "\n".join(out)


def run(ds, pred, retr, client, tau=0.5, limit=None):
    preds = [json.loads(l) for l in open(pred) if l.strip()]
    by_id = {str(r.get("query_id")): r for r in (json.loads(l) for l in open(retr) if l.strip())}
    if limit:
        preds = preds[:limit]
    n = raw_ok = vtr_ok = reasked = reask_ok = reask_was_wrong = 0
    for p in preds:
        rec = by_id.get(str(p.get("query_id")))
        if not rec or not rec.get("retrieved"):
            continue
        q = p.get("query"); gold = p.get("gold"); raw = p.get("raw_pred") or ""
        pack = build_evidence_pack(q, rec["retrieved"], top_n_facts=8); led = _union(pack)
        if led is None:
            continue
        n += 1
        raw_correct = bool(number_match(raw, gold)) if gold is not None else False
        raw_ok += int(raw_correct)
        cpr = verify_cpr(raw, led, q, gold=gold, selected_facts=pack.selected_facts)
        if cpr.confidence >= tau:
            vtr_ok += int(raw_correct)
            continue
        # ungrounded -> re-ask with structure evidence
        reasked += 1
        block = _kg_block(pack)
        prompt = f"{block}\n\nQuestion: {q}\nReturn one line only: Answer: <number>"
        ans = client.generate(prompt, system=REASK_SYSTEM, temperature=0.0, max_tokens=48, idx=101)
        new_correct = bool(number_match(ans, gold)) if gold is not None else False
        vtr_ok += int(new_correct)
        reask_ok += int(new_correct)
        reask_was_wrong += int(not raw_correct)
        # track rescues/regressions
        run._rescued = getattr(run, "_rescued", 0) + int((not raw_correct) and new_correct)
        run._broke = getattr(run, "_broke", 0) + int(raw_correct and not new_correct)
    res = {"dataset": ds, "n": n, "tau": tau,
           "raw_NM": round(raw_ok / max(n, 1), 4),
           "verify_then_reask_NM": round(vtr_ok / max(n, 1), 4),
           "n_reasked": reasked,
           "reask_subset_NM": round(reask_ok / max(reasked, 1), 4),
           "rescued": getattr(run, "_rescued", 0),
           "broke": getattr(run, "_broke", 0)}
    run._rescued = 0; run._broke = 0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--pred", default="outputs/research/gemini_gen/{ds}_predictions.jsonl")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/research/gemini_verify_reask/report.json")
    args = ap.parse_args()
    client = GeminiClient()
    allr = {}
    for ds in args.datasets:
        pp = args.pred.format(ds=ds)
        if not Path(pp).exists():
            continue
        r = run(ds, pp, RETR[ds], client, tau=args.tau, limit=args.limit); allr[ds] = r
        print(f"{ds}: raw_NM={r['raw_NM']} -> verify_then_reask_NM={r['verify_then_reask_NM']} "
              f"(reasked {r['n_reasked']}, subset_NM={r['reask_subset_NM']}, rescued={r['rescued']}, broke={r['broke']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out, "|", client.stats())


if __name__ == "__main__":
    main()
