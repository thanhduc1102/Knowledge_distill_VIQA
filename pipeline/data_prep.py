"""
Data preparation module for Knowledge Distillation pipeline.
Handles: ViNumQA + FinQA loading, merging, program_re augmentation,
table→markdown conversion, and train/val splitting.
"""

import re
import json
import random
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig


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
    samples: list, prompt_template: str
) -> list:
    """
    Prepare dataset for teacher inference (knowledge distillation).
    Returns list of {prompt, ground_truth, id, table}.
    """
    dataset = []
    for sample in samples:
        qa = sample["qa"]
        user_content = build_user_prompt(sample, prompt_template)
        ground_truth = build_assistant_response(qa["program"], qa["exe_ans"])

        dataset.append({
            "prompt": [{"role": "user", "content": user_content}],
            "ground_truth": ground_truth,
            "id": sample["id"],
            "table": sample["table"],
            "program": format_comma_spaces(qa["program"]),
            "answer": str(qa["exe_ans"]),
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

    # Load all data
    vi_train, vi_valid = load_vinumqa(cfg)
    finqa_data = load_finqa(cfg) if cfg.data.use_finqa else []

    # Combine for training
    all_train = vi_train + finqa_data
    if cfg.data.max_samples:
        all_train = all_train[:cfg.data.max_samples]

    print(f"\nTotal training samples (raw): {len(all_train)}")
    print(f"Validation samples (raw): {len(vi_valid)}")

    # Prepare datasets for each phase
    # Phase 1: Teacher distillation input
    teacher_data = prepare_teacher_dataset(all_train, prompt_template)
    teacher_path = str(output_dir / "teacher_input.json")
    save_json(teacher_path, teacher_data)

    # Phase 2: SFT dataset (will be used with teacher output, but also as baseline)
    sft_train = prepare_sft_dataset(all_train, prompt_template, cfg.data.use_program_re)
    sft_valid = prepare_sft_dataset(vi_valid, prompt_template, use_program_re=False)
    random.shuffle(sft_train)

    sft_train_path = str(output_dir / "sft_train.json")
    sft_valid_path = str(output_dir / "sft_valid.json")
    save_json(sft_train_path, sft_train)
    save_json(sft_valid_path, sft_valid)

    # Phase 3: GRPO dataset
    grpo_train = prepare_grpo_dataset(all_train, prompt_template, cfg.data.use_program_re)
    grpo_valid = prepare_grpo_dataset(vi_valid, prompt_template, use_program_re=False)
    random.shuffle(grpo_train)

    grpo_train_path = str(output_dir / "grpo_train.parquet")
    grpo_valid_path = str(output_dir / "grpo_valid.parquet")

    import pandas as pd
    pd.DataFrame(grpo_train).to_parquet(grpo_train_path, index=False)
    pd.DataFrame(grpo_valid).to_parquet(grpo_valid_path, index=False)
    print(f"Saved {len(grpo_train)} GRPO train → {grpo_train_path}")
    print(f"Saved {len(grpo_valid)} GRPO valid → {grpo_valid_path}")

    paths = {
        "teacher_input": teacher_path,
        "sft_train": sft_train_path,
        "sft_valid": sft_valid_path,
        "grpo_train": grpo_train_path,
        "grpo_valid": grpo_valid_path,
    }

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"  SFT train:  {len(sft_train)} samples")
    print(f"  SFT valid:  {len(sft_valid)} samples")
    print(f"  GRPO train: {len(grpo_train)} samples")
    print(f"  GRPO valid: {len(grpo_valid)} samples")
    print(f"  Teacher:    {len(teacher_data)} samples")
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
