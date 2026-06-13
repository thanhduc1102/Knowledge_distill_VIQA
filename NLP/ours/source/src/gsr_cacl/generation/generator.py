"""Generators for the RAG answer phase.

Two implementations:
  * ``ExtractiveGenerator`` — deterministic, GPU-free baseline that computes the answer
    directly from the selected ledger facts (intent detection: change/sum/ratio/lookup).
    Lets the full pipeline run + be evaluated without any LLM, and acts as a sanity floor.
  * ``HFGenerator`` — HuggingFace causal-LM generator (default a small Qwen that fits a
    T4), prompted with the fact-grounded evidence block.

Both share the interface ``generate(query, evidence_block, meta, facts) -> str``.
"""

from __future__ import annotations

import re
from typing import Optional

from gsr_cacl.ledger.fact import Fact
from gsr_cacl.ledger.numeric import extract_years
from gsr_cacl.generation.prompts import build_chat_messages


class BaseGenerator:
    name = "base"

    def generate(self, query: str, evidence_block: str, meta: dict | None = None,
                 facts: Optional[list[Fact]] = None) -> str:
        raise NotImplementedError

    def generate_batch(self, queries, evidence_blocks, metas=None, facts_list=None) -> list[str]:
        metas = metas or [None] * len(queries)
        facts_list = facts_list or [None] * len(queries)
        return [self.generate(q, e, m, f)
                for q, e, m, f in zip(queries, evidence_blocks, metas, facts_list)]


# ----------------------------------------------------------------------
# Deterministic extractive baseline
# ----------------------------------------------------------------------

_CHANGE = re.compile(r"\b(change|difference|increase|decrease|growth|grew|grow|decline|net change|how much (?:more|less))\b", re.I)
_SUM = re.compile(r"\b(sum|total|combined|together|aggregate)\b", re.I)
_RATIO = re.compile(r"\b(ratio|percent|percentage|proportion|margin|share of|fraction|portion)\b", re.I)
_AVG = re.compile(r"\b(average|mean)\b", re.I)


class ExtractiveGenerator(BaseGenerator):
    name = "extractive"

    def generate(self, query, evidence_block, meta=None, facts=None) -> str:
        facts = [f for f in (facts or []) if f.value is not None]
        if not facts:
            return "Answer: "
        q_years = extract_years(query)

        # If the query references specific years, prefer facts for those periods.
        def by_year(yr):
            for f in facts:
                if f.period and str(yr) in str(f.period):
                    return f
            return None

        # ordered pair (old, new): prefer the two referenced years, else top-2 facts
        def ordered_pair():
            if len(q_years) >= 2:
                fo, fn = by_year(min(q_years)), by_year(max(q_years))
                if fo and fn:
                    return fo.value, fn.value
            if len(facts) >= 2:
                return facts[1].value, facts[0].value  # facts[0] is the strongest match -> treat as 'new'
            return None

        is_pct_change = _RATIO.search(query) and _CHANGE.search(query)
        if is_pct_change:
            pair = ordered_pair()
            if pair and pair[0]:
                old, new = pair
                return f"Answer: {round((new - old) / abs(old), 6)}"

        if _RATIO.search(query) and len(facts) >= 2:
            a, b = facts[0].value, facts[1].value
            val = (a / b) if b else a
            return f"Answer: {round(val, 4)}"

        if _CHANGE.search(query):
            pair = ordered_pair()
            if pair:
                return f"Answer: {round(pair[1] - pair[0], 4)}"

        if _SUM.search(query) and len(facts) >= 2:
            return f"Answer: {round(sum(f.value for f in facts[:4]), 4)}"

        if _AVG.search(query) and len(facts) >= 2:
            vals = [f.value for f in facts[:4]]
            return f"Answer: {round(sum(vals) / len(vals), 4)}"

        # lookup: best-matching fact, preferring a year match
        if q_years:
            f = by_year(q_years[0])
            if f:
                return f"Answer: {f.value}"
        return f"Answer: {facts[0].value}"


# ----------------------------------------------------------------------
# HuggingFace causal-LM generator (Qwen by default)
# ----------------------------------------------------------------------

class HFGenerator(BaseGenerator):
    name = "hf"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 64,
        dtype: str = "bfloat16",
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, query, evidence_block, meta=None, facts=None) -> str:
        import torch

        messages = build_chat_messages(query, evidence_block, meta)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True)


def build_generator(kind: str = "extractive", **kwargs) -> BaseGenerator:
    kind = (kind or "extractive").lower()
    if kind in ("extractive", "baseline", "rule"):
        return ExtractiveGenerator()
    if kind in ("hf", "qwen", "llm"):
        return HFGenerator(**kwargs)
    raise ValueError(f"Unknown generator kind: {kind}")
