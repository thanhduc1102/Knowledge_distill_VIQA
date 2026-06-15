"""Generators for the RAG answer phase.

Two implementations:
  * ``ExtractiveGenerator`` — deterministic, GPU-free baseline that computes the answer
    directly from the selected ledger facts (intent detection: change/sum/ratio/lookup).
    Lets the full pipeline run + be evaluated without any LLM, and acts as a sanity floor.
  * ``HFGenerator`` — HuggingFace generator with a multimodal-first loader that can run
    Qwen/Qwen3.5-4B via ``AutoProcessor`` + ``AutoModelForMultimodalLM`` when available,
    while still falling back to a standard causal LM for text-only checkpoints.

Both share the interface ``generate(query, evidence_block, meta, facts) -> str``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from gsr_cacl.generation.prompts import build_chat_messages
from gsr_cacl.ledger.fact import Fact
from gsr_cacl.ledger.numeric import extract_years

logger = logging.getLogger(__name__)


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
# HuggingFace generator (multimodal-first, text fallback)
# ----------------------------------------------------------------------


def _resolve_torch_dtype(dtype):
    import torch

    if dtype is None or str(dtype).lower() == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "double": torch.float64,
    }
    return aliases.get(key, getattr(torch, key, torch.float32))



class HFGenerator(BaseGenerator):
    name = "hf"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 64,
        dtype: str = "bfloat16",
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ):
        import torch

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("auto" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._is_multimodal = False
        self.processor = None
        self.tokenizer = None
        self.model = None
        self._input_device = None
        self._decoder = None
        torch_dtype = _resolve_torch_dtype(dtype)

        try:
            self._load_multimodal(torch_dtype, trust_remote_code)
        except Exception as exc:
            logger.info("Multimodal loader failed for %s (%s); trying causal LM fallback.",
                        model_name, exc)
            self._load_causal(torch_dtype, trust_remote_code)

    def _load_multimodal(self, torch_dtype, trust_remote_code: bool) -> None:
        from transformers import AutoProcessor

        model_cls = None
        try:
            from transformers import AutoModelForMultimodalLM as model_cls  # type: ignore
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText as model_cls  # type: ignore
            except Exception as exc:
                raise RuntimeError("No multimodal model class available in transformers") from exc

        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cpu":
            model = model_cls.from_pretrained(self.model_name, **load_kwargs)
            model = model.to("cpu")
        else:
            load_kwargs["device_map"] = "auto"
            model = model_cls.from_pretrained(self.model_name, **load_kwargs)

        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is not None:
            if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        self.model = model
        self._input_device = next(self.model.parameters()).device
        self._decoder = getattr(self.processor, "decode", None)
        if self._decoder is None and self.tokenizer is not None:
            self._decoder = self.tokenizer.decode
        self._is_multimodal = True
        self.model.eval()
        logger.info("Loaded multimodal generator: %s", self.model_name)

    def _load_causal(self, torch_dtype, trust_remote_code: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            model = model.to("cpu")
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        self.tokenizer = tokenizer
        self.processor = None
        self.model = model
        self._input_device = next(self.model.parameters()).device
        self._decoder = self.tokenizer.decode
        self._is_multimodal = False
        self.model.eval()
        logger.info("Loaded causal generator: %s", self.model_name)

    @staticmethod
    def _as_multimodal_messages(messages: list[dict]) -> list[dict]:
        out = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            out.append({**msg, "content": content})
        return out

    @staticmethod
    def _normalize_completion(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return "Answer: "
        lower = text.lower()
        for marker in ("answer:", "final answer:", "the answer is"):
            idx = lower.rfind(marker)
            if idx >= 0:
                return text[idx:].splitlines()[0].strip()
        return text.splitlines()[0].strip()

    def _build_prompt_texts(self, messages_batch: list[list[dict]]) -> list[str]:
        if self._is_multimodal:
            return [
                self.processor.apply_chat_template(  # type: ignore[union-attr]
                    self._as_multimodal_messages(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for messages in messages_batch
            ]
        return [
            self.tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in messages_batch
        ]

    def _generate_from_prompts(self, prompts: list[str]) -> list[str]:
        import torch

        if not prompts:
            return []
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self._input_device)  # type: ignore[union-attr]
        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions: list[str] = []
        for row in out:
            completion = self._decoder(row[prompt_len:], skip_special_tokens=True)  # type: ignore[misc]
            completions.append(self._normalize_completion(completion))
        return completions

    def generate_batch(self, queries, evidence_blocks, metas=None, facts_list=None) -> list[str]:
        metas = metas or [None] * len(queries)
        messages_batch = [build_chat_messages(q, e, m) for q, e, m in zip(queries, evidence_blocks, metas)]
        prompts = self._build_prompt_texts(messages_batch)
        return self._generate_from_prompts(prompts)

    def generate(self, query, evidence_block, meta=None, facts=None) -> str:
        out = self.generate_batch([query], [evidence_block], [meta], [facts])
        return out[0] if out else "Answer: "


def build_generator(kind: str = "extractive", **kwargs) -> BaseGenerator:
    kind = (kind or "extractive").lower()
    if kind in ("extractive", "baseline", "rule"):
        return ExtractiveGenerator()
    if kind in ("hf", "qwen", "llm"):
        return HFGenerator(**kwargs)
    raise ValueError(f"Unknown generator kind: {kind}")
