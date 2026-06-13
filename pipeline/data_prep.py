"""
Data preparation module for Knowledge Distillation pipeline.
Handles: ViNumQA + FinQA loading, merging, program_re augmentation,
table→markdown conversion, and train/val splitting.
"""

import re
import json
import random
import copy
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.benchmarks import load_benchmark_split


# ── Table Formatting ─────────────────────────────────────────────────

def table_to_markdown(table: list) -> str:
    """Convert 2D list table to Markdown format."""
    if not table or not table[0]:
        return ""
    # Calculate column widths
    widths = [0] * len(table[0])
    for row in table:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    lines = []
    for row_idx, row in enumerate(table):
        cells = [str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                 for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(lines)


def format_comma_spaces(s: str) -> str:
    return re.sub(r',(\s*)', ', ', s)


# ── Sample Formatting ────────────────────────────────────────────────

def build_user_prompt(sample: dict, prompt_template: str) -> str:
    """Build the user prompt from a raw sample and template."""
    if sample.get("prompt_override"):
        return sample["prompt_override"]

    pre_text = " ".join(sample["pre_text"])
    table_md = table_to_markdown(sample["table"])
    post_text = " ".join(sample["post_text"])
    question = sample["qa"]["question"]

    text = prompt_template
    text = text.replace("pre_text_placeholder", pre_text)
    text = text.replace("table_placeholder", table_md)
    text = text.replace("post_text_placeholder", post_text)
    text = text.replace("question_placeholder", question)
    return text


def build_assistant_response(program: str, exe_ans: Any) -> str:
    """Build formatted assistant response with program + answer."""
    prog = format_comma_spaces(program)
    return (
        f"**Chương trình tính toán:**\n{prog}\n\n"
        f"**Đáp án cuối cùng:**\n{str(exe_ans)}"
    )


# ── Data Loading ─────────────────────────────────────────────────────

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples → {path}")


def load_vinumqa(cfg: PipelineConfig) -> tuple[list, list]:
    """Load ViNumQA train and validation splits."""
    train, valid = [], []
    train_path = Path(cfg.project_root) / cfg.data.vinumqa_train
    valid_path = Path(cfg.project_root) / cfg.data.vinumqa_valid

    if train_path.exists():
        train = load_json(str(train_path))
        print(f"ViNumQA train: {len(train)} samples")
    else:
        print(f"WARNING: {train_path} not found")

    if valid_path.exists():
        valid = load_json(str(valid_path))
        print(f"ViNumQA valid: {len(valid)} samples")
    else:
        print(f"WARNING: {valid_path} not found")

    return train, valid


def load_finqa(cfg: PipelineConfig) -> list:
    """Load and merge all FinQA splits."""
    finqa_dir = Path(cfg.project_root) / cfg.data.finqa_dir
    if not finqa_dir.exists():
        print(f"WARNING: FinQA directory not found: {finqa_dir}")
        return []

    data = []
    for split in ["train.json", "dev.json", "test.json"]:
        path = finqa_dir / split
        if path.exists():
            samples = load_json(str(path))
            print(f"FinQA {split}: {len(samples)} samples")
            data.extend(samples)
    return data


# ── Dataset Building ─────────────────────────────────────────────────

def prepare_sft_dataset(
    samples: list, prompt_template: str, use_program_re: bool = True
) -> list:
    """
    Convert raw samples to SFT training format.
    Returns list of {messages: [{role, content}], id, metadata}.
    """
    dataset = []
    for sample in samples:
        qa = sample["qa"]
        if not qa.get("program"):
            continue
        user_content = build_user_prompt(sample, prompt_template)
        assistant_content = build_assistant_response(qa["program"], qa["exe_ans"])

        dataset.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "id": sample["id"],
            "metadata": {
                "table": sample["table"],
                "program": format_comma_spaces(qa["program"]),
                "answer": str(qa["exe_ans"]),
            },
        })

        # Augment with program_re if available and different
        if use_program_re and "program_re" in qa:
            prog_re = format_comma_spaces(qa["program_re"])
            prog_orig = format_comma_spaces(qa["program"])
            if prog_re != prog_orig:
                asst_re = build_assistant_response(qa["program_re"], qa["exe_ans"])
                dataset.append({
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": asst_re},
                    ],
                    "id": sample["id"] + "_re",
                    "metadata": {
                        "table": sample["table"],
                        "program": prog_re,
                        "answer": str(qa["exe_ans"]),
                    },
                })
    return dataset


def prepare_grpo_dataset(
    samples: list, prompt_template: str, use_program_re: bool = True
) -> list:
    """
    Convert raw samples to GRPO training format (verl-compatible).
    Returns list of {data_source, prompt, reward_model, id}.
    """
    dataset = []
    for sample in samples:
        qa = sample["qa"]
        if not qa.get("program"):
            continue
        user_content = build_user_prompt(sample, prompt_template)

        ground_truth = {
            "table": sample["table"],
            "program": format_comma_spaces(qa["program"]),
            "answer": str(qa["exe_ans"]),
        }

        dataset.append({
            "data_source": "vlsp2025/vinumqa",
            "prompt": [{"role": "user", "content": user_content}],
            "reward_model": {"ground_truth": ground_truth},
            "id": sample["id"],
        })

        if use_program_re and "program_re" in qa:
            prog_re = format_comma_spaces(qa["program_re"])
            prog_orig = format_comma_spaces(qa["program"])
            if prog_re != prog_orig:
                ground_truth_re = {
                    "table": sample["table"],
                    "program": prog_re,
                    "answer": str(qa["exe_ans"]),
                }
                dataset.append({
                    "data_source": "vlsp2025/vinumqa",
                    "prompt": [{"role": "user", "content": user_content}],
                    "reward_model": {"ground_truth": ground_truth_re},
                    "id": sample["id"] + "_re",
                })
    return dataset


def prepare_teacher_dataset(
    samples: list,
    prompt_template: str,
    guided: bool = False,
    guided_template: str = None,
) -> list:
    """
    Prepare dataset for teacher inference (knowledge distillation).

    Args:
        guided: If True, embed gold program+answer into the prompt using guided_template
                (think.py format). Teacher only generates reasoning, not program/answer.
                Benefits: ~100% match rate, ~2x faster generation, higher quality traces.
        guided_template: The think.py prompt string (required when guided=True).

    Returns list of {prompt, ground_truth, id, table, program, answer}.
    """
    dataset = []
    for sample in samples:
        qa = sample["qa"]
        if not qa.get("program"):
            continue
        gold_program = format_comma_spaces(qa["program"])
        gold_answer = str(qa["exe_ans"])

        if guided and guided_template:
            # Build prompt with gold program/answer embedded as hints.
            # Teacher generates reasoning AROUND the correct program.
            user_content = build_user_prompt(sample, guided_template)
            user_content = user_content.replace("cttt_placeholder", gold_program)
            user_content = user_content.replace("dacc_placeholder", gold_answer)
        else:
            user_content = build_user_prompt(sample, prompt_template)

        ground_truth = build_assistant_response(qa["program"], qa["exe_ans"])

        dataset.append({
            "prompt": [{"role": "user", "content": user_content}],
            "ground_truth": ground_truth,
            "id": sample["id"],
            "table": sample["table"],
            "program": gold_program,
            "answer": gold_answer,
            "_guided": guided and guided_template is not None,
        })
    return dataset


def annotate_program_samples(samples: list, benchmark: str) -> list:
    annotated = []
    for sample in samples:
        clone = copy.deepcopy(sample)
        clone["benchmark"] = benchmark
        clone.setdefault("eval", {"metric_family": "program"})
        annotated.append(clone)
    return annotated


def prepare_inference_dataset(samples: list, prompt_template: str) -> list:
    dataset = []
    for sample in samples:
        qa = sample.get("qa", {})
        user_content = build_user_prompt(sample, prompt_template)
        dataset.append({
            "id": sample["id"],
            "benchmark": sample.get("benchmark", "unknown"),
            "messages": [{"role": "user", "content": user_content}],
            "metadata": {
                "table": sample.get("table", []),
                "program": format_comma_spaces(qa.get("program", "")) if qa.get("program") else "",
                "answer": str(qa.get("exe_ans", "")),
                "answer_raw": sample.get("eval", {}).get("raw_answer", qa.get("exe_ans")),
                "metric_family": sample.get("eval", {}).get("metric_family", "program"),
                "answer_type": sample.get("eval", {}).get("answer_type", ""),
                "scale": sample.get("eval", {}).get("scale", ""),
                "python_solution": sample.get("eval", {}).get("python_solution"),
                "gold_steps": sample.get("eval", {}).get("gold_steps", []),
            },
        })
    return dataset


# ── Main Entry Point ─────────────────────────────────────────────────

def run_data_prep(cfg: PipelineConfig) -> dict:
    """
    Main data preparation pipeline.
    Returns dict of output file paths.
    """
    from src.assets.template import prompt as prompt_template

    random.seed(cfg.seed)
    output_dir = Path(cfg.project_root) / cfg.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vi_train, vi_valid = [], []
    if cfg.data.include_vinumqa_in_training:
        vi_train, vi_valid = load_vinumqa(cfg)
        vi_train = annotate_program_samples(vi_train, "vinumqa")
        vi_valid = annotate_program_samples(vi_valid, "vinumqa")

    # Load benchmark-aligned supervised training data
    supervised_train = []
    supervised_valid = []
    benchmark_manifest = {"train": {}, "eval": {}, "skipped": {}}

    for benchmark in cfg.data.train_benchmarks:
        try:
            train_samples = load_benchmark_split(cfg, benchmark, "train")
            valid_samples = load_benchmark_split(cfg, benchmark, "valid")
        except Exception as exc:
            if cfg.data.skip_missing_benchmarks:
                print(f"WARNING: skipping train benchmark {benchmark}: {exc}")
                benchmark_manifest["skipped"][benchmark] = {"phase": "train", "reason": str(exc)}
                continue
            raise

        train_with_program = [sample for sample in train_samples if sample.get("qa", {}).get("program")]
        valid_with_program = [sample for sample in valid_samples if sample.get("qa", {}).get("program")]
        if not train_with_program:
            print(f"WARNING: skipping train benchmark {benchmark}: no program supervision found")
            benchmark_manifest["skipped"][benchmark] = {
                "phase": "train",
                "reason": "no program supervision found",
            }
            continue

        supervised_train.extend(train_with_program)
        supervised_valid.extend(valid_with_program)
        benchmark_manifest["train"][benchmark] = {
            "train_samples": len(train_with_program),
            "valid_samples": len(valid_with_program),
        }

    # Combine for training
    all_train = vi_train + supervised_train
    if cfg.data.max_samples:
        all_train = all_train[:cfg.data.max_samples]

    print(f"\nTotal training samples (raw): {len(all_train)}")
    print(f"Validation samples (raw): {len(vi_valid) + len(supervised_valid)}")

    # Phase 1: Teacher distillation input
    use_guided = getattr(cfg.teacher, "use_guided_template", True)
    guided_tmpl = None
    if use_guided:
        try:
            from src.assets.think import prompt as guided_template
            guided_tmpl = guided_template
            print(f"Teacher mode: GUIDED (think.py) — gold program/answer embedded in prompt")
        except ImportError:
            print("WARNING: think.py not found, falling back to free-form template")
            use_guided = False
    else:
        print("Teacher mode: FREE-FORM (template.py)")

    teacher_data = prepare_teacher_dataset(
        all_train, prompt_template, guided=use_guided, guided_template=guided_tmpl
    )
    teacher_path = str(output_dir / "teacher_input.json")
    save_json(teacher_path, teacher_data)

    # Phase 2: SFT dataset (will be used with teacher output, but also as baseline)
    sft_train = prepare_sft_dataset(all_train, prompt_template, cfg.data.use_program_re)
    sft_valid_raw = vi_valid + supervised_valid
    sft_valid = prepare_sft_dataset(sft_valid_raw, prompt_template, use_program_re=False)
    random.shuffle(sft_train)

    sft_train_path = str(output_dir / "sft_train.json")
    sft_valid_path = str(output_dir / "sft_valid.json")
    save_json(sft_train_path, sft_train)
    save_json(sft_valid_path, sft_valid)

    # Phase 3: GRPO dataset
    grpo_train = prepare_grpo_dataset(all_train, prompt_template, cfg.data.use_program_re)
    grpo_valid = prepare_grpo_dataset(sft_valid_raw, prompt_template, use_program_re=False)
    random.shuffle(grpo_train)

    grpo_train_path = str(output_dir / "grpo_train.parquet")
    grpo_valid_path = str(output_dir / "grpo_valid.parquet")

    import pandas as pd
    pd.DataFrame(grpo_train).to_parquet(grpo_train_path, index=False)
    pd.DataFrame(grpo_valid).to_parquet(grpo_valid_path, index=False)
    print(f"Saved {len(grpo_train)} GRPO train → {grpo_train_path}")
    print(f"Saved {len(grpo_valid)} GRPO valid → {grpo_valid_path}")

    # Benchmark evaluation datasets
    benchmark_eval_dir = output_dir / "benchmarks"
    benchmark_eval_dir.mkdir(parents=True, exist_ok=True)
    benchmark_eval_paths = {}
    split_override = {"vinumqa": "valid", "docmath_eval": "eval", "finchain": "test"}
    for benchmark in cfg.data.eval_benchmarks:
        purpose = split_override.get(benchmark, "test")
        try:
            samples = load_benchmark_split(cfg, benchmark, purpose)
        except Exception as exc:
            if cfg.data.skip_missing_benchmarks:
                print(f"WARNING: skipping eval benchmark {benchmark}: {exc}")
                benchmark_manifest["skipped"][benchmark] = {"phase": "eval", "reason": str(exc)}
                continue
            raise

        inference_dataset = prepare_inference_dataset(samples, prompt_template)
        bench_dir = benchmark_eval_dir / benchmark
        bench_dir.mkdir(parents=True, exist_ok=True)
        eval_path = str(bench_dir / f"{purpose}.json")
        save_json(eval_path, inference_dataset)
        benchmark_eval_paths[benchmark] = eval_path
        benchmark_manifest["eval"][benchmark] = {
            "split": purpose,
            "samples": len(inference_dataset),
            "path": eval_path,
        }

    benchmark_manifest_path = output_dir / "benchmark_manifest.json"
    with open(benchmark_manifest_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark manifest → {benchmark_manifest_path}")

    paths = {
        "teacher_input": teacher_path,
        "sft_train": sft_train_path,
        "sft_valid": sft_valid_path,
        "grpo_train": grpo_train_path,
        "grpo_valid": grpo_valid_path,
        "benchmark_manifest": str(benchmark_manifest_path),
        "eval_datasets": benchmark_eval_paths,
    }

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"  SFT train:  {len(sft_train)} samples")
    print(f"  SFT valid:  {len(sft_valid)} samples")
    print(f"  GRPO train: {len(grpo_train)} samples")
    print(f"  GRPO valid: {len(grpo_valid)} samples")
    print(f"  Teacher:    {len(teacher_data)} samples")
    print(f"  Eval suite: {len(benchmark_eval_paths)} benchmarks")
    print(f"{'='*60}\n")

    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    from pipeline.config import load_config
    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config)
    run_data_prep(cfg)
