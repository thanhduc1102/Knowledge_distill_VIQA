#!/usr/bin/env python3
"""GPU-backed generation audit for the structure/KG RAG pipeline.

This script is intentionally stricter than ``generation_e2e.py``:

* RAW: the LLM answers from the raw top-k retrieved tables/text.
* KG: the same LLM answers from structure/KG-selected facts and provenance, with the
  deterministic KG final answer stripped out. This measures whether structure helps the
  generator, not whether a symbolic answer can replace it.
* VERIFY_THEN_REASK: RAW answer is verified against the fact ledger. If it is not
  grounded or derivable from retrieved cells, the LLM is re-asked with explicit
  provenance and structure paths.

The output is prediction-level JSONL plus aggregate research diagnostics: grounded vs
ungrounded correctness, hallucination catch rate, provenance/support precision, and a
small risk-coverage curve based on annotation-free verifier confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "src")

from gsr_cacl.generation.retrieval_bridge import build_evidence_pack, render_prompt_context
from gsr_cacl.generation.verifier import VerificationResult, extract_final_number, verify
from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ledger.numeric import number_match


DEFAULTS = {
    "finqa": "outputs/final_retrieval/finqa/retrieval_top3.jsonl",
    "convfinqa": "outputs/final_retrieval/convfinqa/retrieval_top3.jsonl",
    "tatqa": "outputs/final_retrieval/tatqa/retrieval_top3.jsonl",
}

RAW_SYSTEM = (
    "You are a precise financial QA model. Use only the retrieved financial tables "
    "and text. Return exactly one line in the form Answer: <number>. Do not provide "
    "explanation, bullets, or extra text. For percentage questions, prefer the decimal "
    "fraction when the table evidence supports it."
)

KG_SYSTEM = (
    "You are a precise financial QA model. Use only the provided selected facts, "
    "cell provenance, operand provenance, and structure evidence paths. Do not invent "
    "values. If arithmetic is needed, compute it from the provided operands/cells. "
    "Return exactly one line in the form Answer: <number>."
)


def _hardware_info() -> dict[str, Any]:
    try:
        import torch

        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_mib": int(props.total_memory // 1024**2),
            })
        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": devices,
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": str(exc)}


def _get_retrieved(rec: dict) -> list[dict]:
    if "retrieved" in rec:
        return rec["retrieved"]
    if "retrieved_docs" in rec:
        return [
            {
                "id": d.get("id", ""),
                "context_id": d.get("context_id", d.get("id", "")),
                "score": d.get("score", 0.0),
                "meta": d.get("metadata", d.get("meta", {})),
                "table": d.get("table", ""),
                "page_content": d.get("page_content", ""),
            }
            for d in rec["retrieved_docs"]
        ]
    raise KeyError("retrieved")


def _gold(rec: dict):
    return rec.get("gold") or rec.get("answers") or rec.get("answer")


def _query_meta(rec: dict) -> dict:
    return rec.get("query_meta") or rec.get("meta") or {}


def _doc_id(doc: dict) -> str:
    return str(doc.get("context_id") or doc.get("id") or "")


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars] + "\n[TRUNCATED]"


def raw_evidence_block(retrieved: list[dict], max_doc_chars: int = 1200) -> str:
    blocks = []
    for i, doc in enumerate(retrieved[:3], 1):
        meta = doc.get("meta") or doc.get("metadata") or {}
        title = (
            f"[Doc {i}] id={_doc_id(doc)} company={meta.get('company_name', '')} "
            f"year={meta.get('report_year', '')}"
        )
        body = doc.get("table") or doc.get("page_content") or ""
        blocks.append(f"{title}\n{_truncate(body, max_doc_chars)}")
    return "\n\n".join(blocks)


_STRIP_PREFIXES = (
    "KG_SYMBOLIC_ANSWER:",
    "KG_SYMBOLIC_CONFIDENCE:",
    "KG_CALCULATION_TRACE:",
)


def audit_kg_block(pack) -> str:
    """Render KG/structure context without a final symbolic answer replacement line."""
    lines = []
    skip_trace_continuation = False
    for line in render_prompt_context(pack).splitlines():
        stripped = line.strip()
        if stripped.startswith(_STRIP_PREFIXES):
            skip_trace_continuation = stripped.startswith("KG_CALCULATION_TRACE:")
            continue
        if skip_trace_continuation and stripped.startswith(("-", "[", "{")):
            continue
        skip_trace_continuation = False
        lines.append(line)
    return "\n".join(lines)


def build_messages(system: str, query: str, evidence: str, meta: dict | None = None,
                   extra: str = "") -> list[dict]:
    meta = meta or {}
    head = []
    if meta.get("company_name"):
        line = f"Company: {meta.get('company_name')}"
        if meta.get("report_year"):
            line += f" | Reporting context year: {meta.get('report_year')}"
        head.append(line)
    if extra:
        head.append(extra.strip())
    if evidence.strip():
        head.append(evidence.strip())
    head.append(f"Question: {query}")
    head.append("Return one line only: Answer: <number>")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(head)},
    ]


def reask_messages(query: str, first_answer: str, kg_block: str, meta: dict | None = None) -> list[dict]:
    extra = (
        "A previous answer could not be grounded or derived from the retrieved ledger. "
        f"Previous answer: {first_answer}. Re-answer using the cell/provenance evidence "
        "below. If two documents conflict, prefer the facts whose concept and period "
        "match the question."
    )
    return build_messages(KG_SYSTEM, query, kg_block, meta, extra=extra)


class BatchedHF:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        max_input_tokens: int,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None

        aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = aliases.get(dtype.lower(), torch.float16 if torch.cuda.is_available() else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        def _load(**kw):
            try:
                return AutoModelForCausalLM.from_pretrained(model_name, **kw)
            except (ValueError, KeyError) as exc:
                if AutoModelForImageTextToText is None:
                    raise
                print(f"  (CausalLM load failed: {exc}; falling back to ImageTextToText)", flush=True)
                return AutoModelForImageTextToText.from_pretrained(model_name, **kw)

        if device == "cpu":
            self.model = _load(**load_kwargs).to("cpu")
        else:
            load_kwargs["device_map"] = "auto" if device == "auto" else {"": device}
            self.model = _load(**load_kwargs)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    def _prompt_texts(self, messages: list[list[dict]]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msg in messages
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return "Answer: "
        lower = text.lower()
        for marker in ("answer:", "final answer:", "the answer is"):
            idx = lower.rfind(marker)
            if idx >= 0:
                return "Answer: " + text[idx + len(marker):].strip().splitlines()[0]
        nums = re.findall(r"-?[\d,]*\.?\d+%?", text)
        if nums:
            return "Answer: " + nums[-1].replace(",", "")
        return text.splitlines()[0].strip()

    def generate_messages(self, messages: list[list[dict]], batch_size: int) -> list[str]:
        import torch

        outs: list[str] = []
        prompts = self._prompt_texts(messages)
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.input_device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            prompt_len = inputs["input_ids"].shape[1]
            for row in generated:
                text = self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                outs.append(self._normalize(text))
        return outs


def union_ledger(pack) -> FactLedger | None:
    if not pack.ranked:
        return None
    base = pack.ranked[0].ledger
    merged = FactLedger(doc_id="union", facts=list(base.facts), meta=dict(base.meta))
    for doc in pack.ranked[1:]:
        merged.facts.extend(doc.ledger.facts)
    return merged


def support_confidence(vr: VerificationResult | None) -> float:
    if vr is None or vr.pred_value is None:
        return 0.0
    scaled = getattr(vr, "normalization", "identity") != "identity"
    if vr.grounded:
        return 0.75 if scaled else 1.0
    if vr.derivable:
        return 0.55 if scaled else 0.85
    return max(0.05, min(0.6, 0.4 * vr.grounding_fraction + 0.2 * vr.arithmetic_fraction))


def _safe_nm(prediction: str, gold) -> bool:
    value = extract_final_number(prediction)
    return bool(value is not None and number_match(value, gold))


def risk_coverage(rows: list[dict], pred_key: str, confidence_key: str) -> list[dict]:
    scored = sorted(rows, key=lambda r: r[confidence_key], reverse=True)
    curve = []
    n = len(scored)
    for coverage in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(round(n * coverage)))
        kept = scored[:k]
        acc = sum(1 for r in kept if r[pred_key]) / max(len(kept), 1)
        curve.append({"coverage": coverage, "risk": round(1.0 - acc, 4), "accuracy": round(acc, 4), "n": len(kept)})
    return curve


def summarize(rows: list[dict], args, elapsed: float) -> dict:
    n = len(rows)
    def acc(key: str) -> float:
        return sum(1 for r in rows if r[key]) / max(n, 1)

    triggered = [r for r in rows if r["reask_triggered"]]
    raw_unsupported = [r for r in rows if not r["raw_supported"]]
    raw_supported = [r for r in rows if r["raw_supported"]]
    hallucination_pool = [r for r in raw_unsupported if not r["raw_correct"]]
    rescued = [r for r in triggered if (not r["raw_correct"]) and r["reask_correct"]]
    harmed = [r for r in triggered if r["raw_correct"] and not r["reask_correct"]]

    return {
        "dataset": args.dataset,
        "input": args.input or DEFAULTS[args.dataset],
        "n": n,
        "model": args.model,
        "sample": args.sample,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "seconds": round(elapsed, 1),
        "hardware": _hardware_info(),
        "number_match": {
            "raw": round(acc("raw_correct"), 4),
            "kg_audit_prompt": round(acc("kg_correct"), 4),
            "verify_then_reask": round(acc("final_correct"), 4),
        },
        "grounded_vs_ungrounded_raw": {
            "supported_n": len(raw_supported),
            "supported_accuracy": round(sum(r["raw_correct"] for r in raw_supported) / max(len(raw_supported), 1), 4),
            "unsupported_n": len(raw_unsupported),
            "unsupported_accuracy": round(sum(r["raw_correct"] for r in raw_unsupported) / max(len(raw_unsupported), 1), 4),
        },
        "verifier": {
            "raw_supported_rate": round(len(raw_supported) / max(n, 1), 4),
            "raw_cell_grounded_rate": round(sum(r["raw_grounded"] for r in rows) / max(n, 1), 4),
            "raw_derivable_rate": round(sum(r["raw_derivable"] for r in rows) / max(n, 1), 4),
            "support_precision_proxy": round(
                sum(r["raw_correct"] for r in raw_supported) / max(len(raw_supported), 1), 4
            ),
            "support_recall_on_correct": round(
                sum(1 for r in rows if r["raw_correct"] and r["raw_supported"]) /
                max(sum(1 for r in rows if r["raw_correct"]), 1), 4
            ),
        },
        "reask": {
            "triggered": len(triggered),
            "trigger_rate": round(len(triggered) / max(n, 1), 4),
            "rescued_raw_errors": len(rescued),
            "harmed_raw_correct": len(harmed),
            "hallucination_catch_rate_proxy": round(len(rescued) / max(len(hallucination_pool), 1), 4),
        },
        "risk_coverage": {
            "raw_verifier_confidence": risk_coverage(rows, "raw_correct", "raw_confidence"),
            "final_verifier_confidence": risk_coverage(rows, "final_correct", "final_confidence"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--sample", type=int, default=200, help="0 means full input")
    ap.add_argument("--sample-strategy", choices=["head", "random"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--top-n-facts", type=int, default=8)
    ap.add_argument("--raw-doc-chars", type=int, default=1200)
    ap.add_argument("--out", default="outputs/research/generation_system_eval")
    args = ap.parse_args()

    path = args.input or DEFAULTS[args.dataset]
    records = [json.loads(line) for line in open(path) if line.strip()]
    if args.sample and args.sample < len(records):
        if args.sample_strategy == "random":
            rng = random.Random(args.seed)
            indices = sorted(rng.sample(range(len(records)), args.sample))
            records = [records[i] for i in indices]
        else:
            records = records[:args.sample]

    prepared = []
    raw_messages = []
    kg_messages = []
    t0 = time.time()
    for rec in records:
        retrieved = _get_retrieved(rec)
        if not retrieved:
            continue
        query = rec["query"]
        meta = _query_meta(rec)
        pack = build_evidence_pack(query, retrieved, top_n_facts=args.top_n_facts)
        kg_block = audit_kg_block(pack)
        prepared.append({
            "rec": rec,
            "retrieved": retrieved,
            "pack": pack,
            "kg_block": kg_block,
            "ledger": union_ledger(pack),
        })
        raw_messages.append(build_messages(
            RAW_SYSTEM, query, raw_evidence_block(retrieved, args.raw_doc_chars), meta
        ))
        kg_messages.append(build_messages(KG_SYSTEM, query, kg_block, meta))

    print(f"Loading {args.model} on device={args.device} dtype={args.dtype} ...", flush=True)
    model = BatchedHF(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    print(f"Generating RAW for {len(prepared)} examples ...", flush=True)
    raw_preds = model.generate_messages(raw_messages, args.batch_size)
    print(f"Generating KG audit prompt for {len(prepared)} examples ...", flush=True)
    kg_preds = model.generate_messages(kg_messages, args.batch_size)

    rows = []
    reask_indices = []
    reask_prompts = []
    for i, item in enumerate(prepared):
        rec = item["rec"]
        ledger = item["ledger"]
        gold = _gold(rec)
        raw_vr = verify(raw_preds[i], ledger, rec["query"], gold=gold) if ledger else None
        kg_vr = verify(kg_preds[i], ledger, rec["query"], gold=gold) if ledger else None
        raw_supported = bool(raw_vr and (raw_vr.grounded or raw_vr.derivable))
        if not raw_supported:
            reask_indices.append(i)
            reask_prompts.append(reask_messages(rec["query"], raw_preds[i], item["kg_block"], _query_meta(rec)))
        rows.append({
            "query_id": rec.get("query_id"),
            "query": rec["query"],
            "gold": gold,
            "ground_truth_id": rec.get("ground_truth_id"),
            "retrieved_ids": [_doc_id(d) for d in item["retrieved"][:3]],
            "kg_selected_doc": item["pack"].ranked[0].doc_id if item["pack"].ranked else "",
            "kg_tier": item["pack"].tier,
            "kg_margin": item["pack"].margin,
            "kg_structure_score": item["pack"].ranked[0].structure_score if item["pack"].ranked else 0.0,
            "raw_prediction": raw_preds[i],
            "kg_prediction": kg_preds[i],
            "raw_pred_value": raw_vr.pred_value if raw_vr else None,
            "kg_pred_value": kg_vr.pred_value if kg_vr else None,
            "raw_correct": bool(raw_vr.answer_match if raw_vr else _safe_nm(raw_preds[i], gold)),
            "kg_correct": bool(kg_vr.answer_match if kg_vr else _safe_nm(kg_preds[i], gold)),
            "raw_grounded": bool(raw_vr.grounded if raw_vr else False),
            "raw_derivable": bool(raw_vr.derivable if raw_vr else False),
            "raw_supported": raw_supported,
            "kg_grounded": bool(kg_vr.grounded if kg_vr else False),
            "kg_derivable": bool(kg_vr.derivable if kg_vr else False),
            "raw_confidence": support_confidence(raw_vr),
            "kg_confidence": support_confidence(kg_vr),
            "raw_explanation": raw_vr.explanation if raw_vr else "",
            "kg_explanation": kg_vr.explanation if kg_vr else "",
            "reask_triggered": not raw_supported,
            "reask_prediction": "",
            "reask_pred_value": None,
            "reask_correct": False,
            "final_prediction": raw_preds[i],
            "final_correct": bool(raw_vr.answer_match if raw_vr else _safe_nm(raw_preds[i], gold)),
            "final_confidence": support_confidence(raw_vr),
            "provenance": item["pack"].provenance[:8],
            "structure_paths": item["pack"].ranked[0].structure_paths[:4] if item["pack"].ranked else [],
        })

    if reask_prompts:
        print(f"Generating verifier-triggered REASK for {len(reask_prompts)} examples ...", flush=True)
        reask_preds = model.generate_messages(reask_prompts, args.batch_size)
        for idx, pred in zip(reask_indices, reask_preds):
            item = prepared[idx]
            rec = item["rec"]
            gold = _gold(rec)
            ledger = item["ledger"]
            vr = verify(pred, ledger, rec["query"], gold=gold) if ledger else None
            rows[idx]["reask_prediction"] = pred
            rows[idx]["reask_pred_value"] = vr.pred_value if vr else None
            rows[idx]["reask_correct"] = bool(vr.answer_match if vr else _safe_nm(pred, gold))
            rows[idx]["reask_grounded"] = bool(vr.grounded if vr else False)
            rows[idx]["reask_derivable"] = bool(vr.derivable if vr else False)
            rows[idx]["reask_confidence"] = support_confidence(vr)
            rows[idx]["reask_explanation"] = vr.explanation if vr else ""
            rows[idx]["final_prediction"] = pred
            rows[idx]["final_correct"] = rows[idx]["reask_correct"]
            rows[idx]["final_confidence"] = rows[idx]["reask_confidence"]

    elapsed = time.time() - t0
    summary = summarize(rows, args, elapsed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.dataset}.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / f"{args.dataset}_predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved -> {out_dir / (args.dataset + '.json')}", flush=True)
    print(f"Saved -> {out_dir / (args.dataset + '_predictions.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
