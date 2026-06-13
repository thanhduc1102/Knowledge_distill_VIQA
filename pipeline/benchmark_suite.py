"""Suite runner for the main verifier-native reasoning benchmarks.

Runs the shared data-prep and SFT stages once, then evaluates:
- sft_only
- grpo_pcpo
- grpo_ecrl

on the benchmark manifest produced by ``pipeline.data_prep``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pipeline.config import PipelineConfig, load_config, save_config


def _suite_root(cfg: PipelineConfig) -> Path:
    return Path(cfg.project_root) / cfg.data.output_dir / "benchmark_suite"


def _variant_cfg(cfg: PipelineConfig, tag: str) -> PipelineConfig:
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg.sft.output_dir = f"checkpoints/benchmark_suite/sft_base"
    variant_cfg.grpo.output_dir = f"checkpoints/benchmark_suite/{tag}"
    return variant_cfg


def _load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model_on_suite(
    cfg: PipelineConfig,
    model_path: str,
    manifest: dict,
    variant_name: str,
) -> dict:
    from pipeline.evaluate import run_evaluation
    from pipeline.inference import run_inference

    variant_root = _suite_root(cfg) / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)

    per_benchmark = {}
    for benchmark, dataset_path in manifest.get("evaluation_paths", {}).items():
        bench_root = variant_root / benchmark
        bench_root.mkdir(parents=True, exist_ok=True)

        predictions_path = str(bench_root / "predictions.json")
        metrics_path = str(bench_root / "metrics.json")

        run_inference(
            cfg,
            model_path=model_path,
            test_data_path=dataset_path,
            output_path=predictions_path,
        )
        metrics = run_evaluation(predictions_path, metrics_path)
        per_benchmark[benchmark] = {
            "predictions_path": predictions_path,
            "metrics_path": metrics_path,
            "answer_accuracy": metrics.get("answer_accuracy"),
            "program_accuracy": metrics.get("program_accuracy"),
            "step_accuracy": metrics.get("step_accuracy"),
            "valid_rate": metrics.get("valid_rate"),
            "total": metrics.get("total"),
        }

    summary = {
        "variant": variant_name,
        "model_path": model_path,
        "benchmarks": per_benchmark,
    }
    summary_path = variant_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved suite summary → {summary_path}")
    return summary


def run_benchmark_suite(
    cfg: PipelineConfig,
    include_sft_only: bool = True,
    include_pcpo: bool = True,
    include_ecrl: bool = True,
) -> dict:
    from pipeline.data_prep import run_data_prep
    from pipeline.teacher_distill import run_teacher_distillation
    from pipeline.train_grpo import run_grpo_training
    from pipeline.train_sft import run_sft_training

    suite_root = _suite_root(cfg)
    suite_root.mkdir(parents=True, exist_ok=True)

    data_paths = run_data_prep(cfg)
    manifest = _load_manifest(data_paths["benchmark_manifest"])

    save_config(cfg, str(suite_root / "suite_config.yaml"))

    teacher_input = data_paths["teacher_input"]
    distilled_path = run_teacher_distillation(cfg, teacher_input)

    sft_cfg = _variant_cfg(cfg, "grpo_pcpo")
    sft_model_path = run_sft_training(
        sft_cfg,
        train_path=distilled_path,
        valid_path=data_paths["sft_valid"],
    )

    suite_results = {}
    if include_sft_only:
        suite_results["sft_only"] = evaluate_model_on_suite(
            sft_cfg, sft_model_path, manifest, "sft_only"
        )

    if include_pcpo:
        pcpo_cfg = _variant_cfg(cfg, "grpo_pcpo")
        pcpo_cfg.grpo.reward_type = "pcpo"
        pcpo_model_path = run_grpo_training(pcpo_cfg, sft_model_path)
        suite_results["grpo_pcpo"] = evaluate_model_on_suite(
            pcpo_cfg, pcpo_model_path, manifest, "grpo_pcpo"
        )

    if include_ecrl:
        ecrl_cfg = _variant_cfg(cfg, "grpo_ecrl")
        ecrl_cfg.grpo.reward_type = "ecrl"
        ecrl_model_path = run_grpo_training(ecrl_cfg, sft_model_path)
        suite_results["grpo_ecrl"] = evaluate_model_on_suite(
            ecrl_cfg, ecrl_model_path, manifest, "grpo_ecrl"
        )

    summary_path = suite_root / "suite_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark suite results → {summary_path}")
    return suite_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main reasoning benchmark suite")
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-sft-only", action="store_true")
    parser.add_argument("--skip-pcpo", action="store_true")
    parser.add_argument("--skip-ecrl", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.max_samples is not None:
        overrides["data"] = {"max_samples": args.max_samples}

    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config, overrides=overrides or None)
    run_benchmark_suite(
        cfg,
        include_sft_only=not args.skip_sft_only,
        include_pcpo=not args.skip_pcpo,
        include_ecrl=not args.skip_ecrl,
    )


if __name__ == "__main__":
    main()