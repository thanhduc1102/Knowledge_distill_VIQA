#!/usr/bin/env python3
"""Self-consistency baseline (k sampled answers) vs CPR reliability.

The strongest annotation-free confidence baseline for LLM QA is *self-consistency*: sample k
answers and use the agreement of the majority as confidence. We compare its AUROC (predicting
correctness of the majority answer) against CPR on the same answers. If CPR matches or beats a
k=5 self-consistency signal, the structure-grounded signal is a genuinely strong reliability
estimator (and it is k× cheaper — one greedy decode).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack
from gsr_cacl.generation.verifier import extract_final_number
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match, numbers_close
from gsr_cacl.research.cpr_verifier import verify_cpr

RAW_SYSTEM = (
    "You are a precise financial QA model. Use only the retrieved financial tables and text. "
    "Return exactly one line in the form Answer: <number>."
)


def _auroc(scores, labels):
    pos = sum(1 for l in labels if l); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sp = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sp - pos * (pos + 1) / 2.0) / (pos * neg)


def _union(pack):
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    m = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for d in pack.ranked[1:]:
        m.facts.extend(d.ledger.facts)
    return m


def _raw_block(retrieved, max_chars=1200):
    blocks = []
    for i, doc in enumerate(retrieved[:3], 1):
        meta = doc.get("meta") or doc.get("metadata") or {}
        body = str(doc.get("table") or doc.get("page_content") or "")[:max_chars]
        blocks.append(f"[Doc {i}] company={meta.get('company_name','')} year={meta.get('report_year','')}\n{body}")
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["finqa", "convfinqa", "tatqa"])
    ap.add_argument("--retr", default="outputs/final_retrieval/{ds}/retrieval_top3.jsonl")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-input-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/research/self_consistency")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 trust_remote_code=True, low_cpu_mem_usage=True,
                                                 device_map={"": args.device})
    model.eval()
    in_dev = next(model.parameters()).device
    torch.manual_seed(args.seed)

    def gen(prompts, sample):
        outs = []
        for s in range(0, len(prompts), args.batch_size):
            batch = prompts[s:s + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_input_tokens).to(in_dev)
            with torch.no_grad():
                kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id)
                if sample:
                    kw.update(do_sample=True, temperature=args.temperature, top_p=0.95)
                else:
                    kw.update(do_sample=False)
                g = model.generate(**enc, **kw)
            plen = enc["input_ids"].shape[1]
            for r in g:
                txt = tok.decode(r[plen:], skip_special_tokens=True)
                m = re.search(r"answer:\s*(-?[\d,]*\.?\d+%?)", txt, re.I)
                outs.append("Answer: " + (m.group(1).replace(",", "") if m else (txt.strip().splitlines()[0] if txt.strip() else "")))
        return outs

    allres = {}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        recs = [json.loads(l) for l in open(args.retr.format(ds=ds)) if l.strip()][:args.sample]
        prompts, packs = [], []
        for rec in recs:
            retrieved = rec.get("retrieved") or rec.get("retrieved_docs") or []
            user = _raw_block(retrieved) + f"\n\nQuestion: {rec['query']}\nReturn one line only: Answer: <number>"
            msg = [{"role": "system", "content": RAW_SYSTEM}, {"role": "user", "content": user}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            packs.append(build_evidence_pack(rec["query"], retrieved, top_n_facts=8))
        t0 = time.time()
        samples = [gen(prompts, sample=True) for _ in range(args.k)]  # k sampled decodes
        rows = []
        for i, rec in enumerate(recs):
            answers = [extract_final_number(samples[j][i]) for j in range(args.k)]
            answers = [a for a in answers if a is not None]
            if not answers:
                continue
            # majority numeric answer (bucket by closeness)
            buckets = []
            for a in answers:
                placed = False
                for b in buckets:
                    if numbers_close(a, b[0]):
                        b.append(a); placed = True; break
                if not placed:
                    buckets.append([a])
            best = max(buckets, key=len)
            maj = best[0]
            sc_conf = len(best) / args.k
            gold = rec.get("gold")
            correct = bool(number_match(maj, gold))
            ledger = _union(packs[i])
            cpr_conf = verify_cpr(f"Answer: {maj}", ledger, rec["query"], gold=gold,
                                  selected_facts=packs[i].selected_facts).confidence if ledger else 0.0
            rows.append({"correct": correct, "sc_conf": sc_conf, "cpr_conf": cpr_conf})
        elapsed = time.time() - t0
        n = len(rows)
        res = {
            "dataset": ds, "n": n, "k": args.k,
            "majority_accuracy": round(sum(r["correct"] for r in rows) / max(n, 1), 4),
            "self_consistency_auroc": round(_auroc([r["sc_conf"] for r in rows], [r["correct"] for r in rows]), 4),
            "cpr_auroc": round(_auroc([r["cpr_conf"] for r in rows], [r["correct"] for r in rows]), 4),
            "seconds": round(elapsed, 1),
        }
        allres[ds] = res
        (out / f"{ds}.json").write_text(json.dumps(res, indent=2))
        print(f"{ds}: n={n} maj_acc={res['majority_accuracy']} | self-consistency AUROC={res['self_consistency_auroc']} "
              f"vs CPR AUROC={res['cpr_auroc']}  ({res['seconds']}s)", flush=True)
    (out / "summary.json").write_text(json.dumps(allres, indent=2))
    print(f"Saved -> {out}/")


if __name__ == "__main__":
    main()
