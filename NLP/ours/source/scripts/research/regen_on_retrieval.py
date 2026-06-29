"""Regenerate answers on a (stronger) retrieval_top3 and measure Number-Match.

Used to convert a retrieval gain into an end-to-end NM gain (esp. TAT-DQA, retrieval-bound).
Gold answers are joined by question text from a reference predictions file (gemini_gen), so we
evaluate exactly the same queries as the baseline → apples-to-apples NM comparison.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.utils.gemini_client import GeminiClient

SYS = ("You are a precise financial QA model. Use only the retrieved financial tables and text. "
       "Return exactly one line in the form 'Answer: <number>'. No explanation.")


def _trunc(t, n):
    t = str(t or "")
    return t if len(t) <= n else t[:n] + "\n[TRUNCATED]"


def raw_block(retrieved, k=3, mx=1400):
    out = []
    for i, d in enumerate(retrieved[:k], 1):
        m = d.get("meta") or d.get("metadata") or {}
        out.append(f"[Doc {i}] company={m.get('company_name','')} year={m.get('report_year','')}\n"
                   f"{_trunc(d.get('table') or d.get('page_content') or '', mx)}")
    return "\n\n".join(out)


def norm_q(q):
    return " ".join(str(q or "").lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retr", required=True)
    ap.add_argument("--ref", required=True, help="reference predictions (carries gold + defines query set)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # gold + baseline by question
    gold_by_q, base_by_q = {}, {}
    for l in open(args.ref):
        if l.strip():
            r = json.loads(l)
            key = norm_q(r.get("question") or r.get("query"))
            gold_by_q[key] = r.get("gold"); base_by_q[key] = r.get("raw_correct")
    recs = [json.loads(l) for l in open(args.retr) if l.strip()]
    client = GeminiClient()
    n = base_ok = new_ok = 0
    t0 = time.time()
    for rec in recs:
        key = norm_q(rec.get("raw_question") or rec.get("query"))
        if key not in gold_by_q:
            continue
        gold = gold_by_q[key]
        n += 1
        base_ok += int(bool(base_by_q.get(key)))
        ev = raw_block(rec.get("retrieved", []))
        q = rec.get("raw_question") or rec.get("query")
        prompt = f"{ev}\n\nQuestion: {q}\nReturn one line only: Answer: <number>"
        ans = client.generate(prompt, system=SYS, temperature=0.0, max_tokens=48, idx=0)
        new_ok += int(number_match(ans, gold))
    res = {"retr": args.retr, "n": n,
           "baseline_NM": round(base_ok / max(n, 1), 4),
           "new_retrieval_NM": round(new_ok / max(n, 1), 4),
           "delta": round((new_ok - base_ok) / max(n, 1), 4),
           "seconds": round(time.time() - t0, 1), "client": client.stats()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
