#!/usr/bin/env python3
"""Verify-then-reask pipeline.

Policy under the revised paper narrative:
  raw table/context -> frozen LLM answer -> Ledger verifier
  if ungrounded: re-ask with KG evidence/provenance, otherwise keep raw answer.

For reproducibility this script can reuse an existing raw `predictions.jsonl` and run a
deterministic extractive re-ask.  With `--generator hf`, it performs the same policy with
a HuggingFace model for the re-ask only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from gsr_cacl.generation.generator import ExtractiveGenerator
from gsr_cacl.generation.prompts import build_chat_messages
from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context
from gsr_cacl.ledger.numeric import number_match

DEFAULT_TOP3 = {
    "finqa": "outputs/strong_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/strong_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/strong_retrieval/tat-dqa/retrieval_top3.jsonl",
}

_ANS = re.compile(r"answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+%?)", re.I)


def parse_answer(text: str):
    hits = list(_ANS.finditer(text or ""))
    if hits:
        return hits[-1].group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]*\.?\d+", text or "")
    return nums[-1].replace(",", "") if nums else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULT_TOP3), default="finqa")
    ap.add_argument("--raw-predictions", required=True)
    ap.add_argument("--top3", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--generator", choices=["extractive", "hf"], default="extractive")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/research/verify_reask")
    args = ap.parse_args()

    raw_rows = [json.loads(l) for l in open(args.raw_predictions) if l.strip()]
    top_rows = [json.loads(l) for l in open(args.top3 or DEFAULT_TOP3[args.dataset]) if l.strip()]
    if args.sample:
        raw_rows = raw_rows[:args.sample]
        top_rows = top_rows[:args.sample]

    hf = None
    tok = None
    gen_device = args.device
    if args.generator == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        hf = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=args.device
        )
        hf.eval()
        gen_device = next(hf.parameters()).device

    ext = ExtractiveGenerator()
    kept, reasked, raw_correct, final_correct = 0, 0, 0, 0
    examples = []
    preds = []

    for raw, top in zip(raw_rows, top_rows):
        q = raw.get("query") or top.get("query") or ""
        gold = raw.get("gold") if raw.get("gold") is not None else top.get("gold")
        raw_ans = raw.get("pred_value")
        raw_ok = bool(number_match(raw_ans, gold)) if gold is not None and raw_ans is not None else False
        raw_correct += int(raw_ok)
        if raw.get("grounded", False):
            kept += 1
            final_ans = raw_ans
            final_text = raw.get("prediction", "")
            policy = "keep_raw_grounded"
        else:
            reasked += 1
            pack = build_evidence_pack(q, top.get("retrieved") or [])
            context = render_prompt_context(pack)
            if args.generator == "extractive":
                final_text = ext.generate(q, context, facts=pack.selected_facts)
            else:
                msgs = build_chat_messages(q, context, (top.get("retrieved") or [{}])[0].get("meta") or {})
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(gen_device)
                import torch
                with torch.no_grad():
                    out = hf.generate(**inp, max_new_tokens=48, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
                final_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            final_ans = parse_answer(final_text)
            policy = "reask_kg_evidence"
            if len(examples) < 8:
                examples.append({
                    "query": q,
                    "gold": gold,
                    "raw_prediction": raw.get("prediction", ""),
                    "reask_prediction": final_text,
                    "kg_tier": pack.tier,
                    "kg_selected_doc": pack.ranked[0].doc_id if pack.ranked else "",
                })
        final_ok = bool(number_match(final_ans, gold)) if gold is not None else False
        final_correct += int(final_ok)
        preds.append({
            "query": q,
            "gold": gold,
            "raw_answer": raw_ans,
            "final_answer": final_ans,
            "policy": policy,
            "raw_correct": raw_ok,
            "final_correct": final_ok,
        })

    n = len(preds)
    out = {
        "dataset": args.dataset,
        "generator": args.generator,
        "n": n,
        "kept_raw_grounded": kept,
        "reasked": reasked,
        "raw_nm": round(raw_correct / max(n, 1), 4),
        "verify_reask_nm": round(final_correct / max(n, 1), 4),
        "examples": examples,
    }
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.dataset}.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / f"{args.dataset}_predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"\n=== Verify-then-reask — {args.dataset} (n={n}) ===")
    print(f"raw NM={out['raw_nm']:.4f} -> verify_reask NM={out['verify_reask_nm']:.4f} | "
          f"reasked={reasked}/{n}")
    print(f"Saved -> {out_dir / f'{args.dataset}.json'}")


if __name__ == "__main__":
    main()
