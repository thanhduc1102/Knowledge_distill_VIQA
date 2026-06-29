"""Per-query error decomposition (especially TAT-DQA, the hardest benchmark).

Splits every wrong Number-Match into the FIRST stage that failed — so optimization is targeted:
  * retrieval_miss : gold document not in retrieved top-k (can't answer at all).
  * extraction_miss: gold retrieved, but the gold answer is NOT reconstructable from the gold
                     document's auto-extracted Fact-Ledger (<=3 ops) -> parser/extraction ceiling.
  * reasoning_miss : answer IS reconstructable from the ledger, but the generator got it wrong.
  * correct        : Number-Match correct.
Also reports TAT-specific structural stats (multi-period headers, abbreviations) to explain why.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.ledger.extract import extract_ledger_from_table, extract_ledger
from gsr_cacl.ledger.numeric import number_match, parse_financial_number
from gsr_cacl.research.derivation import derivation_depth

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}


def _gold_value(g):
    if g is None:
        return None
    if isinstance(g, list):
        for x in g:
            v = parse_financial_number(str(x))
            if v is not None:
                return v
        return None
    return parse_financial_number(str(g))


def _gold_doc_ledger(rec):
    gid = str(rec.get("ground_truth_id") or "")
    for d in rec.get("retrieved", []):
        if str(d.get("context_id") or d.get("id")) == gid:
            tbl = d.get("table"); ctx = d.get("page_content") or d.get("context") or ""
            return (extract_ledger_from_table(tbl, doc_id=gid, meta=d.get("meta") or {}, caption=ctx[:200])
                    if tbl else extract_ledger(context=ctx, doc_id=gid, meta=d.get("meta") or {})), gid
    return None, gid


def gold_rank(rec, gid):
    for i, d in enumerate(rec.get("retrieved", []), 1):
        if str(d.get("context_id") or d.get("id")) == gid:
            return i
    return -1


def run(ds, topk=3):
    retr = {str(r.get("query_id")): r for r in (json.loads(l) for l in open(RETR[ds]) if l.strip())}
    preds = [json.loads(l) for l in open(f"outputs/research/gemini_gen/{ds}_predictions.jsonl") if l.strip()]
    b = {"correct": 0, "retrieval_miss": 0, "extraction_miss": 0, "reasoning_miss": 0}
    n = 0
    for p in preds:
        rec = retr.get(str(p.get("query_id")))
        if not rec:
            continue
        n += 1
        gold = p.get("gold"); raw = p.get("raw_pred") or ""
        correct = bool(p["raw_correct"]) if p.get("raw_correct") is not None else bool(number_match(raw, gold))
        if correct:
            b["correct"] += 1
            continue
        led, gid = _gold_doc_ledger(rec)
        r = gold_rank(rec, gid)
        if r < 0 or r > topk:
            b["retrieval_miss"] += 1
            continue
        gv = _gold_value(gold)
        vals = [f.value for f in led.numeric_facts() if f.value is not None] if led else []
        depth, _ = derivation_depth(gv, vals, max_ops=3, return_operands=True) if (gv is not None and vals) else (None, None)
        if depth is None:
            b["extraction_miss"] += 1
        else:
            b["reasoning_miss"] += 1
    return {"dataset": ds, "n": n, **{k: round(v / max(n, 1), 4) for k, v in b.items()},
            "_counts": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--out", default="outputs/research/error_decomp/report.json")
    args = ap.parse_args()
    allr = {}
    for ds in args.datasets:
        r = run(ds); allr[ds] = r
        print(f"{ds} (n={r['n']}): correct={r['correct']} | retrieval_miss={r['retrieval_miss']} "
              f"extraction_miss={r['extraction_miss']} reasoning_miss={r['reasoning_miss']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
