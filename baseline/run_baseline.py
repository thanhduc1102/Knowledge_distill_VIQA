"""
Baseline zero-shot inference pipeline for VLSP 2025 Financial Numerical Reasoning.
Runs direct inference (no training) with Qwen3.5 models and evaluates EA/PA on public test.
"""

import gc
import json
import os
import re
import sys
import time
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline.config import (
    BaselineModelConfig,
    get_model_configs,
    get_single_model_config,
)
from pipeline.program_executor import execute_program, format_answer, validate_program
from pipeline.evaluate import evaluate_predictions
from pipeline.data_prep import table_to_markdown


# ── Prompt Construction ──────────────────────────────────────────────

def build_prompt(sample: dict) -> str:
    """Build the user prompt from a raw ViNumQA sample."""
    from src.assets.template import prompt as prompt_template

    pre_text = " ".join(sample.get("pre_text", []))
    table_md = table_to_markdown(sample.get("table", []))
    post_text = " ".join(sample.get("post_text", []))
    question = sample["qa"]["question"]

    text = prompt_template
    text = text.replace("pre_text_placeholder", pre_text)
    text = text.replace("table_placeholder", table_md)
    text = text.replace("post_text_placeholder", post_text)
    text = text.replace("question_placeholder", question)
    return text


# ── Model Loading ────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: BaselineModelConfig):
    """Load model and tokenizer with appropriate quantization settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

    model_kwargs = {
        "trust_remote_code": cfg.trust_remote_code,
        "dtype": dtype,
        "device_map": "auto",
    }

    if cfg.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.quantization == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if cfg.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading model: {cfg.model_id}")
    print(f"  Quantization: {cfg.quantization or 'none (FP16/BF16)'}")
    print(f"  Dtype: {cfg.torch_dtype}")
    print(f"  Flash Attention: {cfg.use_flash_attention}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Show VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")

    return model, tokenizer


def unload_model(model, tokenizer):
    """Properly unload model to free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


# ── Inference ────────────────────────────────────────────────────────

def generate_candidates(
    model, tokenizer, prompt: str, cfg: BaselineModelConfig
) -> list[str]:
    """Generate N candidate outputs for a prompt.
    
    Handles Qwen3/3.5 thinking mode by stripping <think>...</think> tags.
    """
    # Use nothink mode to get direct output (faster, more consistent)
    messages = [{"role": "user", "content": prompt}]

    # Try to disable thinking mode via chat template kwargs
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=cfg.max_seq_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    candidates = []
    for _ in range(cfg.num_candidates):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Strip thinking tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        candidates.append(text)

    return candidates


def extract_program_and_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract program and answer from model output."""
    program_matches = list(re.finditer(
        r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    program = program_matches[-1].group(1).strip() if program_matches else None

    answer_matches = list(re.finditer(
        r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    answer = answer_matches[-1].group(1).strip() if answer_matches else None

    return program, answer


def majority_vote(candidates: list[str], table: Optional[list] = None) -> dict:
    """Apply majority voting on N candidate outputs."""
    answers = []
    programs = []

    for text in candidates:
        program, answer = extract_program_and_answer(text)

        if program and validate_program(program):
            result = execute_program(program, table)
            if result is not None:
                formatted = format_answer(result)
                answers.append(formatted)
                programs.append(program)
            elif answer:
                answers.append(answer.strip())
                programs.append(program)
        elif answer:
            answers.append(answer.strip())
            programs.append(program or "")

    if not answers:
        return {
            "program": None, "answer": None,
            "confidence": 0.0, "num_valid": 0, "num_candidates": len(candidates),
        }

    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]

    best_program = None
    for ans, prog in zip(answers, programs):
        if ans == best_answer:
            best_program = prog
            break

    return {
        "program": best_program,
        "answer": best_answer,
        "confidence": best_count / len(candidates),
        "num_valid": len(answers),
        "num_candidates": len(candidates),
    }


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_results(predictions: list, label: str = "") -> dict:
    """Evaluate predictions and print results."""
    total = len(predictions)
    ea_correct = 0
    pa_correct = 0
    valid_programs = 0

    for pred in predictions:
        pred_ans = pred.get("predicted_answer") or ""
        gold_ans = pred.get("gold_answer") or ""
        pred_prog = pred.get("predicted_program") or ""
        gold_prog = pred.get("gold_program") or ""

        # EA
        try:
            p = float(pred_ans)
            g = float(gold_ans)
            if g == 0:
                ea_match = abs(p - g) < 1e-5
            else:
                ea_match = abs(p - g) / max(abs(g), 1e-10) < 1e-4
        except (ValueError, TypeError):
            ea_match = pred_ans.strip() == gold_ans.strip()

        if ea_match:
            ea_correct += 1

        # PA
        if pred_prog and gold_prog:
            from pipeline.evaluate import programs_match
            if programs_match(pred_prog, gold_prog):
                pa_correct += 1

        if pred_prog and validate_program(pred_prog):
            valid_programs += 1

    ea = ea_correct / total if total else 0
    pa = pa_correct / total if total else 0
    vr = valid_programs / total if total else 0

    result = {
        "model": label,
        "total": total,
        "ea_correct": ea_correct,
        "pa_correct": pa_correct,
        "valid_programs": valid_programs,
        "execution_accuracy": ea,
        "program_accuracy": pa,
        "valid_rate": vr,
    }

    print(f"\n{'='*60}")
    print(f"  Baseline Results: {label}")
    print(f"{'='*60}")
    print(f"  Samples:              {total}")
    print(f"  Execution Accuracy:   {ea:.2%} ({ea_correct}/{total})")
    print(f"  Program Accuracy:     {pa:.2%} ({pa_correct}/{total})")
    print(f"  Valid Program Rate:   {vr:.2%} ({valid_programs}/{total})")
    print(f"{'='*60}\n")

    return result


# ── Main Pipeline ────────────────────────────────────────────────────

def run_single_baseline(
    model_key: str,
    gpu_profile: str = "rtx6000_96gb",
    data_path: str = "data/receive/test.json",
    output_dir: str = "outputs/baseline",
    max_samples: Optional[int] = None,
) -> dict:
    """Run baseline inference for a single model."""
    cfg = get_single_model_config(model_key, gpu_profile)
    data_path = os.path.join(PROJECT_ROOT, data_path)
    output_dir = os.path.join(PROJECT_ROOT, output_dir)

    print(f"\n{'#'*60}")
    print(f"# Baseline: {cfg.model_id}")
    print(f"# GPU Profile: {gpu_profile}")
    print(f"# Data: {data_path}")
    print(f"# Candidates per sample: {cfg.num_candidates}")
    print(f"{'#'*60}\n")

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]
    print(f"Test samples: {len(test_data)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Run inference
    predictions = []
    start_time = time.time()

    for i, sample in enumerate(tqdm(test_data, desc=f"Inference [{cfg.short_name}]")):
        prompt = build_prompt(sample)
        table = sample.get("table")

        candidates = generate_candidates(model, tokenizer, prompt, cfg)
        vote_result = majority_vote(candidates, table)

        gold_prog = sample["qa"].get("program", "")
        gold_ans = str(sample["qa"].get("exe_ans", ""))

        predictions.append({
            "id": sample["id"],
            "predicted_program": vote_result["program"],
            "predicted_answer": vote_result["answer"],
            "confidence": vote_result["confidence"],
            "num_valid": vote_result["num_valid"],
            "gold_program": gold_prog,
            "gold_answer": gold_ans,
        })

        # Periodic progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(test_data) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(test_data)} | "
                  f"Rate: {rate:.1f} samples/s | "
                  f"ETA: {remaining/60:.1f}min")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {elapsed:.1f}s ({elapsed/len(test_data):.2f}s/sample)")

    # Unload model
    unload_model(model, tokenizer)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"predictions_{cfg.short_name}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved: {pred_path}")

    # Evaluate
    result = evaluate_results(predictions, cfg.short_name)
    result["elapsed_seconds"] = elapsed

    # Save eval results
    eval_path = os.path.join(output_dir, f"eval_{cfg.short_name}.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def run_all_baselines(
    gpu_profile: str = "rtx6000_96gb",
    data_path: str = "data/receive/test.json",
    output_dir: str = "outputs/baseline",
    max_samples: Optional[int] = None,
    models: Optional[list[str]] = None,
) -> list[dict]:
    """Run baseline for all configured models sequentially."""
    configs = get_model_configs(gpu_profile)

    if models:
        model_keys = [m for m in models if m in configs]
    else:
        model_keys = list(configs.keys())

    print(f"\n{'='*60}")
    print(f" Running {len(model_keys)} baseline models on {gpu_profile}")
    print(f" Models: {', '.join(model_keys)}")
    print(f" Data: {data_path}")
    print(f"{'='*60}\n")

    all_results = []
    for model_key in model_keys:
        try:
            result = run_single_baseline(
                model_key, gpu_profile, data_path, output_dir, max_samples
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR: Model {model_key} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"model": model_key, "error": str(e)})

    # Summary table
    print(f"\n{'='*80}")
    print(f" BASELINE SUMMARY ({gpu_profile})")
    print(f"{'='*80}")
    print(f" {'Model':<25} {'EA':>8} {'PA':>8} {'Valid%':>8} {'Time':>10}")
    print(f" {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in all_results:
        if "error" in r:
            print(f" {r['model']:<25} {'ERROR':>8} {'':<8} {'':<8} {'':<10}")
        else:
            t = r.get('elapsed_seconds', 0)
            print(f" {r['model']:<25} {r['execution_accuracy']:>7.2%} "
                  f"{r['program_accuracy']:>7.2%} {r['valid_rate']:>7.2%} "
                  f"{t:>8.0f}s")
    print(f"{'='*80}\n")

    # Save summary
    output_dir_full = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir_full, exist_ok=True)
    summary_path = os.path.join(output_dir_full, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu_profile": gpu_profile,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLSP 2025 Baseline Zero-Shot Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines on RTX 6000 Pro 96GB
  python -m baseline.run_baseline --gpu-profile rtx6000_96gb --models all

  # Run single model
  python -m baseline.run_baseline --gpu-profile rtx6000_96gb --models qwen3.5-4b

  # Run on P100 (only small models)
  python -m baseline.run_baseline --gpu-profile p100_16gb --models qwen3.5-4b

  # Quick test with 5 samples
  python -m baseline.run_baseline --gpu-profile p100_16gb --models qwen3.5-4b --max-samples 5

  # Use specific test data
  python -m baseline.run_baseline --data data/receive/valid.json --models qwen3.5-9b
        """,
    )
    parser.add_argument(
        "--gpu-profile", default="rtx6000_96gb",
        choices=["rtx6000_96gb", "p100_16gb"],
        help="GPU profile (default: rtx6000_96gb)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Model keys to run (or 'all')",
    )
    parser.add_argument(
        "--data", default="data/receive/test.json",
        help="Path to test data JSON",
    )
    parser.add_argument(
        "--output-dir", default="outputs/baseline",
        help="Output directory for predictions and results",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to process (for quick testing)",
    )
    args = parser.parse_args()

    if "all" in args.models:
        run_all_baselines(
            gpu_profile=args.gpu_profile,
            data_path=args.data,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
    else:
        for model_key in args.models:
            run_single_baseline(
                model_key=model_key,
                gpu_profile=args.gpu_profile,
                data_path=args.data,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
            )


if __name__ == "__main__":
    main()
