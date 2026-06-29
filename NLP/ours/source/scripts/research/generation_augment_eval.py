"""Does structure AUGMENTATION of the prompt improve Number-Match? (generator-SOTA)

Candidate-selection gave ~0 gain (samples barely diverge); the bottleneck is the INPUT.
Here structure is ADDED to the raw table (never replacing it — replacing hurt, see §5):
  * raw          : raw top-k tables (baseline, cached from gemini_generate).
  * +ledger      : raw + the clean extracted Fact-Ledger facts (concept [period] = value).
  * +kgpath      : raw + structure evidence paths + operand provenance (symbolic answer stripped).
Hypothesis: clean structure reduces table-parsing burden, helping most on TAT-DQA (multi-level
headers). Reports Number-Match per arm. New Gemini calls for the augmented arms (cached).
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, "src")
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context
from gsr_cacl.ledger.numeric import number_match
from gsr_cacl.utils.gemini_client import GeminiClient

RETR = {"finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
        "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
        "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl"}
SYS = ("You are a precise financial QA model. Use the retrieved tables AND the extracted "
       "structured facts to answer. Return exactly one line: Answer: <number>.")
_STRIP = ("KG_SYMBOLIC_ANSWER:", "KG_SYMBOLIC_CONFIDENCE:", "KG_CALCULATION_TRACE:")


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


def ledger_block(pack, k=12):
    lines = []
    for f in (pack.selected_facts or [])[:k]:
        if f.value is None:
            continue
        u = f"{f.unit or ''} {f.scale or ''}".strip()
        lines.append(f"- {f.concept} [{f.period or '?'}] = {f.value}" + (f" ({u})" if u else ""))
    return "Extracted facts:\n" + "\n".join(lines) if lines else ""


def kgpath_block(pack):
    out, skip = [], False
    for line in render_prompt_context(pack).splitlines():
        s = line.strip()
        if s.startswith(_STRIP):
            skip = s.startswith("KG_CALCULATION_TRACE:"); continue
        if skip and s.startswith(("-", "[", "{")):
            continue
        skip = False; out.append(line)
    return "\n".join(out)


def run(ds, client, sample, seed=20260628):
    import random
    recs = [json.loads(l) for l in open(RETR[ds]) if l.strip()]
    rng = random.Random(seed); rng.shuffle(recs)
    recs = recs[:sample]
    # cached raw answers by query_id (reuse gemini_generate output)
    rawmap = {}
    gp = Path(f"outputs/research/gemini_gen/{ds}_predictions.jsonl")
    if gp.exists():
        for l in gp.read_text().splitlines():
            if l.strip():
                r = json.loads(l); rawmap[str(r["query_id"])] = r
    arms = {"raw": 0, "ledger": 0, "kgpath": 0}; n = 0
    for rec in recs:
        qid = str(rec.get("query_id"))
        cached = rawmap.get(qid)
        if cached is None:
            continue
        q = cached.get("query") or rec.get("raw_question"); gold = cached.get("gold")
        n += 1
        raw_ans = cached.get("raw_pred") or ""
        arms["raw"] += int(number_match(raw_ans, gold))
        pack = build_evidence_pack(q, rec["retrieved"], top_n_facts=12)
        rb = raw_block(rec["retrieved"])
        # +ledger
        lp = f"{rb}\n\n{ledger_block(pack)}\n\nQuestion: {q}\nReturn one line: Answer: <number>"
        a_l = client.generate(lp, system=SYS, temperature=0.0, max_tokens=48, idx=0)
        arms["ledger"] += int(number_match(a_l, gold))
        # +kgpath
        kp = f"{rb}\n\n{kgpath_block(pack)}\n\nQuestion: {q}\nReturn one line: Answer: <number>"
        a_k = client.generate(kp, system=SYS, temperature=0.0, max_tokens=48, idx=0)
        arms["kgpath"] += int(number_match(a_k, gold))
    return {"dataset": ds, "n": n, **{k: round(v / max(n, 1), 4) for k, v in arms.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--out", default="outputs/research/generation_augment/report.json")
    args = ap.parse_args()
    client = GeminiClient()
    allr = {}
    for ds in args.datasets:
        t0 = time.time()
        r = run(ds, client, args.sample); allr[ds] = r
        print(f"{ds}: raw={r['raw']} +ledger={r['ledger']} +kgpath={r['kgpath']} "
              f"(n={r['n']}, {time.time()-t0:.0f}s, {client.stats()})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(allr, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
