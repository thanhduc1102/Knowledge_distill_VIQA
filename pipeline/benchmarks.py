"""Benchmark registry utilities for the verifier-native reasoning suite.

This module centralizes benchmark naming, source resolution, and the distinction
between trainable benchmarks and evaluation-only stress tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig


ALIASES = {
    "finqa": "finqa",
    "tatqa": "tatqa",
    "tat-qa": "tatqa",
    "convfinqa": "convfinqa",
    "conv-finqa": "convfinqa",
    "docmath": "docmath_eval",
    "docmath_eval": "docmath_eval",
    "docmath-eval": "docmath_eval",
    "finchain": "finchain",
    "vi": "vinumqa",
    "vinumqa": "vinumqa",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    kind: str
    source: dict[str, Any]
    trainable: bool
    program_supervision: bool
    metric_family: str

    def resolve_path(self, project_root: str, key: str) -> str | None:
        value = self.source.get(key)
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str(Path(project_root) / path)


def normalize_benchmark_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key not in ALIASES:
        raise KeyError(f"Unsupported benchmark: {name}")
    return ALIASES[key]


def get_benchmark_spec(cfg: PipelineConfig, name: str) -> BenchmarkSpec:
    canonical = normalize_benchmark_name(name)
    source = cfg.data.benchmark_sources.get(canonical)
    if source is None:
        raise KeyError(f"Missing benchmark source config for: {canonical}")
    return BenchmarkSpec(
        name=canonical,
        kind=source.get("kind", "local_dir"),
        source=source,
        trainable=bool(source.get("trainable", False)),
        program_supervision=bool(source.get("program_supervision", False)),
        metric_family=str(source.get("metric_family", "answer_only")),
    )


def get_enabled_benchmarks(cfg: PipelineConfig, phase: str) -> list[BenchmarkSpec]:
    if phase == "train":
        names = cfg.data.train_benchmarks
    elif phase == "eval":
        names = cfg.data.eval_benchmarks
    else:
        names = cfg.data.benchmark_suite
    return [get_benchmark_spec(cfg, name) for name in names]


def benchmark_status(cfg: PipelineConfig, name: str) -> dict[str, Any]:
    spec = get_benchmark_spec(cfg, name)
    status = {
        "name": spec.name,
        "kind": spec.kind,
        "trainable": spec.trainable,
        "available": False,
        "detail": "",
    }

    if spec.kind == "local_pair":
        train_path = spec.resolve_path(cfg.project_root, "train_path")
        valid_path = spec.resolve_path(cfg.project_root, "valid_path")
        status["available"] = bool(train_path and Path(train_path).exists() and valid_path and Path(valid_path).exists())
        status["detail"] = train_path or ""
        return status

    if spec.kind == "local_dir":
        data_dir = spec.resolve_path(cfg.project_root, "dir")
        status["available"] = bool(data_dir and Path(data_dir).exists())
        status["detail"] = data_dir or ""
        return status

    if spec.kind in {"hf", "hf_gated"}:
        status["available"] = True
        status["detail"] = spec.source.get("dataset_name", "")
        return status

    status["detail"] = "unknown benchmark source kind"
    return status


def summarize_benchmark_status(cfg: PipelineConfig, phase: str = "suite") -> list[dict[str, Any]]:
    return [benchmark_status(cfg, spec.name) for spec in get_enabled_benchmarks(cfg, phase)]


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _limit_samples(samples: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None or len(samples) <= max_samples:
        return samples
    return samples[:max_samples]


def _load_cached_rows(spec: BenchmarkSpec, cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]] | None:
    cache_dir = spec.resolve_path(cfg.project_root, "cache_dir")
    if cache_dir is None:
        return None
    split_file = spec.source.get("local_splits", {}).get(purpose)
    if split_file is None and purpose == "eval":
        split_file = spec.source.get("local_splits", {}).get("eval")
    if split_file is None:
        return None
    cache_path = Path(cache_dir) / split_file
    if not cache_path.exists():
        return None
    return _load_json(str(cache_path))


def _stringify_answer(value: Any, scale: str | None = None) -> str:
    if isinstance(value, list):
        text = " | ".join(_stringify_answer(v) for v in value)
    elif value is None:
        text = ""
    else:
        text = str(value)
    if scale and text and scale not in {"", "none"}:
        return f"{text} {scale}".strip()
    return text.strip()


def _format_table(table: list[list[Any]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def _build_direct_prompt(question: str, context_blocks: list[str], table: list[list[Any]] | None = None) -> str:
    parts = []
    if context_blocks:
        parts.append("Financial context:\n" + "\n\n".join(block for block in context_blocks if block))
    if table:
        parts.append("Table:\n" + _format_table(table))
    parts.append("Question:\n" + question)
    return "\n\n".join(part for part in parts if part)


def _annotate_program_samples(samples: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    annotated = []
    for sample in samples:
        clone = dict(sample)
        clone["benchmark"] = benchmark
        clone["eval"] = {"metric_family": "program"}
        annotated.append(clone)
    return annotated


def _load_vinumqa_split(cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]]:
    spec = get_benchmark_spec(cfg, "vinumqa")
    key = {
        "train": "train_path",
        "valid": "valid_path",
        "test": "test_path",
        "private_test": "private_test_path",
    }.get(purpose, "valid_path")
    path = spec.resolve_path(cfg.project_root, key)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"ViNumQA split not found: {purpose}")
    return _annotate_program_samples(_limit_samples(_load_json(path), cfg.data.max_samples), "vinumqa")


def _load_finqa_split(cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]]:
    spec = get_benchmark_spec(cfg, "finqa")
    data_dir = spec.resolve_path(cfg.project_root, "dir")
    if data_dir is None:
        raise FileNotFoundError("FinQA directory not configured")
    split_file = spec.source.get("splits", {}).get(purpose)
    if split_file is None:
        raise KeyError(f"FinQA split not configured: {purpose}")
    path = Path(data_dir) / split_file
    if not path.exists():
        raise FileNotFoundError(f"FinQA split not found: {path}")
    return _annotate_program_samples(_limit_samples(_load_json(str(path)), cfg.data.max_samples), "finqa")


def _load_tatqa_split(cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]]:
    spec = get_benchmark_spec(cfg, "tatqa")
    rows = _load_cached_rows(spec, cfg, purpose)
    if rows is None:
        from datasets import load_dataset

        hf_split = spec.source.get("split_map", {}).get(purpose)
        if hf_split is None:
            raise KeyError(f"TAT-QA split not configured: {purpose}")
        dataset = load_dataset(spec.source["dataset_name"], spec.source.get("config_name"), split=hf_split)
        rows = list(dataset)
    samples = []
    for row in rows:
        table = row.get("table", {}).get("table", [])
        paragraphs = [p.get("text", "") for p in row.get("paragraphs", []) if isinstance(p, dict)]
        for question in row.get("questions", []):
            qid = question.get("uid") or f"tatqa-{len(samples)}"
            samples.append({
                "id": f"tatqa::{qid}",
                "benchmark": "tatqa",
                "pre_text": paragraphs,
                "post_text": [],
                "table": table,
                "qa": {
                    "question": question.get("question", ""),
                    "program": "",
                    "exe_ans": _stringify_answer(question.get("answer"), question.get("scale")),
                },
                "eval": {
                    "metric_family": spec.metric_family,
                    "answer_type": question.get("answer_type", ""),
                    "scale": question.get("scale", ""),
                    "raw_answer": question.get("answer"),
                },
            })
    return _limit_samples(samples, cfg.data.max_samples)


def _load_convfinqa_split(cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]]:
    spec = get_benchmark_spec(cfg, "convfinqa")
    rows = _load_cached_rows(spec, cfg, purpose)
    if rows is None:
        from datasets import load_dataset

        hf_split = spec.source.get("split_map", {}).get(purpose)
        if hf_split is None:
            raise KeyError(f"ConvFinQA split not configured: {purpose}")
        dataset = load_dataset(spec.source["dataset_name"], spec.source.get("config_name"))
        if hf_split not in dataset:
            if purpose == "valid" and "valid" in dataset:
                hf_split = "valid"
            else:
                raise KeyError(f"ConvFinQA split unavailable: {purpose}")
        rows = list(dataset[hf_split])
    samples = []
    for row in rows:
        query = row.get("query", "")
        samples.append({
            "id": str(row.get("id", f"convfinqa-{len(samples)}")),
            "benchmark": "convfinqa",
            "prompt_override": query,
            "pre_text": [],
            "post_text": [],
            "table": [],
            "qa": {
                "question": query,
                "program": "",
                "exe_ans": _stringify_answer(row.get("answer")),
            },
            "eval": {
                "metric_family": spec.metric_family,
                "turn": row.get("turn"),
                "dialogue_id": row.get("dialogue_id"),
            },
        })
    return _limit_samples(samples, cfg.data.max_samples)


def _format_docmath_block(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, list):
        if block and isinstance(block[0], list):
            return _format_table(block)
        return "\n".join(str(item) for item in block)
    if isinstance(block, dict):
        if isinstance(block.get("table"), list):
            return _format_table(block["table"])
        if "text" in block:
            return str(block["text"])
        return json.dumps(block, ensure_ascii=False)
    return str(block)


def _load_docmath_eval_split(cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]]:
    spec = get_benchmark_spec(cfg, "docmath_eval")
    rows = _load_cached_rows(spec, cfg, purpose)
    if rows is None:
        from datasets import load_dataset

        hf_split = spec.source.get("split_map", {}).get(purpose) or spec.source.get("split_map", {}).get("eval")
        if hf_split is None:
            raise KeyError("DocMath-Eval split not configured")
        dataset = load_dataset(spec.source["dataset_name"], split=hf_split)
        rows = list(dataset)
    samples = []
    for row in rows:
        question = row.get("question", "")
        context_blocks = [_format_docmath_block(block) for block in row.get("paragraphs", [])]
        answer = row.get("ground_truth", row.get("answer", ""))
        samples.append({
            "id": str(row.get("question_id", f"docmath-{len(samples)}")),
            "benchmark": "docmath_eval",
            "prompt_override": _build_direct_prompt(question, context_blocks),
            "pre_text": [],
            "post_text": [],
            "table": [],
            "qa": {
                "question": question,
                "program": "",
                "exe_ans": _stringify_answer(answer),
            },
            "eval": {
                "metric_family": spec.metric_family,
                "python_solution": row.get("python_solution"),
                "table_evidence": row.get("table_evidence", []),
                "paragraph_evidence": row.get("paragraph_evidence", []),
            },
        })
    return _limit_samples(samples, cfg.data.max_samples)


def _load_finchain_split(cfg: PipelineConfig, purpose: str) -> list[dict[str, Any]]:
    spec = get_benchmark_spec(cfg, "finchain")
    data_dir = spec.resolve_path(cfg.project_root, "dir")
    if data_dir is None:
        raise FileNotFoundError("FinChain directory not configured")
    split_file = spec.source.get("splits", {}).get(purpose)
    if split_file is None:
        raise KeyError(f"FinChain split not configured: {purpose}")
    path = Path(data_dir) / split_file
    if not path.exists():
        raise FileNotFoundError(f"FinChain split not found: {path}")

    samples = []
    for idx, row in enumerate(_load_json(str(path))):
        qa = row.get("qa", {}) if isinstance(row.get("qa"), dict) else {}
        question = row.get("question") or qa.get("question") or row.get("query") or ""
        prompt_override = row.get("prompt_override")
        if prompt_override is None:
            if isinstance(row.get("prompt"), str):
                prompt_override = row["prompt"]
            elif isinstance(row.get("prompt"), list) and row["prompt"]:
                prompt_override = row["prompt"][0].get("content", "")
        sample = {
            "id": str(row.get("id", f"finchain-{idx}")),
            "benchmark": "finchain",
            "pre_text": row.get("pre_text", []),
            "post_text": row.get("post_text", []),
            "table": row.get("table", []),
            "qa": {
                "question": question,
                "program": row.get("program") or qa.get("program", ""),
                "exe_ans": _stringify_answer(row.get("answer", qa.get("exe_ans", row.get("gold_answer", "")))),
            },
            "eval": {
                "metric_family": "step" if row.get("gold_steps") or row.get("steps") else spec.metric_family,
                "gold_steps": row.get("gold_steps") or row.get("steps") or [],
            },
        }
        if prompt_override:
            sample["prompt_override"] = prompt_override
        elif row.get("context"):
            sample["prompt_override"] = _build_direct_prompt(question, [str(row.get("context", ""))])
        samples.append(sample)
    return _limit_samples(samples, cfg.data.max_samples)


def load_benchmark_split(cfg: PipelineConfig, name: str, purpose: str) -> list[dict[str, Any]]:
    canonical = normalize_benchmark_name(name)
    loaders = {
        "vinumqa": _load_vinumqa_split,
        "finqa": _load_finqa_split,
        "tatqa": _load_tatqa_split,
        "convfinqa": _load_convfinqa_split,
        "docmath_eval": _load_docmath_eval_split,
        "finchain": _load_finchain_split,
    }
    loader = loaders.get(canonical)
    if loader is None:
        raise KeyError(f"No loader defined for benchmark: {canonical}")
    return loader(cfg, purpose)