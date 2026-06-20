#!/usr/bin/env python3
"""End-to-end generation: raw top-3 vs KG-evidence vs agreement-hybrid (frozen Qwen).

Conditions (no generator training):
  RAW    — prompt the LLM with the raw top-3 markdown tables.
  KG     — prompt the LLM with the KG evidence pack (selected facts + provenance + symbolic
           hint when the heuristic is confident); the LLM copies/checks.
  HYBRID — if the 3-path ensemble agrees (votes>=2), answer deterministically with the
           symbolic value (NO LLM call); otherwise fall back to the KG condition.

Reports Number-Match per condition + the fraction of HYBRID answers served by the KG alone.

Usage:  PYTHONPATH=src python scripts/research/generation_e2e.py --dataset convfinqa --sample 150 \
            --model Qwen/Qwen2.5-3B-Instruct --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, re, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context
from gsr_cacl.generation.prompts import build_chat_messages
from gsr_cacl.ledger.extract import extract_ledger_from_table
from gsr_cacl.ledger.multipath import multipath_answer
from gsr_cacl.ledger.select import infer_task_type
from gsr_cacl.ledger.numeric import number_match

DEFAULTS = {
    "finqa": "outputs/strong_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/strong_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/strong_retrieval/tat-dqa/retrieval_top3.jsonl",
}
_ANS = re.compile(r"answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+%?)", re.I)


def parse_answer(text: str):
    m = list(_ANS.finditer(text or ""))
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text or "")
        return nums[-1].replace(",", "") if nums else ""
    return m[-1].group(1).replace(",", "")


def raw_block(retrieved):
    out = []
    for i, d in enumerate(retrieved[:3], 1):
        tbl = d.get("table") or d.get("page_content", "")
        out.append(f"[Doc {i}] {(d.get('meta') or {}).get('company_name','')}\n{tbl[:1200]}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="convfinqa")
    ap.add_argument("--input", default=None)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default=None, help="set 'auto' to shard a big model over 2 GPUs")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--out", default="outputs/research/generation_e2e")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.input or DEFAULTS[args.dataset]) if l.strip()]
    recs = recs[: args.sample] if args.sample else recs

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dmap = args.device_map or args.device
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 device_map=dmap)
    model.eval()
    gen_device = args.device if args.device_map is None else next(model.parameters()).device

    def gen(evidence_block, query, meta):
        msgs = build_chat_messages(query, evidence_block, meta)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=3072).to(gen_device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    nm = {"raw": 0, "kg": 0, "hybrid2": 0, "hybrid3": 0, "kg+verify": 0, "selective": 0}
    fired = {"hybrid2": 0, "hybrid3": 0, "selective": 0}
    hallucination_caught = 0
    # faithfulness: does the KG grounding-flag predict raw-answer correctness?
    faith = {"grounded_n": 0, "grounded_nm": 0, "ungrounded_n": 0, "ungrounded_nm": 0}
    n = 0
    for rec in recs:
        q = rec["query"]; gold = rec.get("gold"); meta = (rec.get("retrieved") or [{}])[0].get("meta") or {}
        retrieved = rec.get("retrieved") or []
        if not retrieved:
            continue
        n += 1
        pack = build_evidence_pack(q, retrieved)

        raw_ans = parse_answer(gen(raw_block(retrieved), q, meta))
        raw_ok = int(number_match(raw_ans, gold))
        nm["raw"] += raw_ok
        kg_ans = parse_answer(gen(render_prompt_context(pack), q, meta))
        kg_ok = int(number_match(kg_ans, gold))
        nm["kg"] += kg_ok

        # multipath on the bridge-selected (arbitration-winner) doc
        top = pack.ranked[0].doc_id if pack.ranked else ""
        src = next((d for d in retrieved if str(d.get("context_id") or d.get("id")) == top), retrieved[0])
        led = extract_ledger_from_table(src.get("table") or src.get("page_content", ""),
                                        doc_id=top, meta=src.get("meta") or {})
        mp = multipath_answer(q, led, infer_task_type(q))

        # hybrid@k: symbolic override only at votes>=k, else KG-LLM answer
        for k, key in ((2, "hybrid2"), (3, "hybrid3")):
            if mp.answer is not None and mp.votes >= k:
                fired[key] += 1
                nm[key] += int(number_match(mp.answer, gold))
            else:
                nm[key] += kg_ok

        # kg+verify: keep the LLM answer, but if it is ungrounded (not near any ledger value)
        # AND a >=2-vote symbolic exists, replace with the symbolic value (catch hallucination).
        vals = [f.value for f in led.numeric_facts() if f.value is not None]
        kg_num = parse_answer(kg_ans)
        try:
            kgv = float(str(kg_num).replace('%', ''))
            grounded = any(abs(kgv - v) <= 1e-2 * max(abs(v), 1.0) for v in vals)
        except (ValueError, TypeError):
            grounded = True
        if (not grounded) and mp.answer is not None and mp.votes >= 2:
            hallucination_caught += 1
            nm["kg+verify"] += int(number_match(mp.answer, gold))
        else:
            nm["kg+verify"] += kg_ok

        # selective: TRUST raw (best baseline) unless it is ungrounded — then KG steps in.
        # grounded = raw answer matches a ledger CELL value OR any task-aware multipath
        # candidate (covers COMPUTED answers: difference/ratio/%change/sum), not just single cells.
        cand_vals = list(vals) + [c["answer"] for c in (mp.candidates or [])]
        try:
            rv = float(str(parse_answer(raw_ans)).replace('%', ''))
            raw_grounded = any(abs(rv - v) <= 1e-2 * max(abs(v), 1.0) for v in cand_vals)
        except (ValueError, TypeError):
            raw_grounded = False
        if raw_grounded:
            nm["selective"] += raw_ok
            faith["grounded_n"] += 1; faith["grounded_nm"] += raw_ok
        else:
            fired["selective"] += 1
            faith["ungrounded_n"] += 1; faith["ungrounded_nm"] += raw_ok
            if mp.answer is not None and mp.votes >= 2:
                nm["selective"] += int(number_match(mp.answer, gold))
            else:
                nm["selective"] += kg_ok

    print(f"\n=== End-to-end generation — {args.dataset} (n={n}, model={args.model}) ===")
    for k in ("raw", "kg", "hybrid2", "hybrid3", "kg+verify", "selective"):
        print(f"   {k:<11} NM = {nm[k]/max(n,1):.3f}")
    print(f"   override fired: votes>=2 {fired['hybrid2']}/{n} ({fired['hybrid2']/max(n,1):.0%}), "
          f"votes>=3 {fired['hybrid3']}/{n} ({fired['hybrid3']/max(n,1):.0%})")
    print(f"   hallucinations caught (ungrounded LLM → symbolic): {hallucination_caught}/{n}")
    gn, un = faith["grounded_n"], faith["ungrounded_n"]
    print(f"   FAITHFULNESS — KG grounding-flag vs raw correctness:")
    print(f"      grounded   : n={gn:4} ({gn/max(n,1):.0%})  NM={faith['grounded_nm']/max(gn,1):.3f}")
    print(f"      ungrounded : n={un:4} ({un/max(n,1):.0%})  NM={faith['ungrounded_nm']/max(un,1):.3f}")
    out = {"dataset": args.dataset, "n": n, "model": args.model,
           "NM": {k: round(nm[k]/max(n,1), 4) for k in nm},
           "fired": fired, "hallucination_caught": hallucination_caught,
           "faithfulness": {"grounded_n": gn, "grounded_nm": round(faith['grounded_nm']/max(gn,1), 4),
                            "ungrounded_n": un, "ungrounded_nm": round(faith['ungrounded_nm']/max(un,1), 4)}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {Path(args.out)/(args.dataset+'.json')}")


if __name__ == "__main__":
    main()
