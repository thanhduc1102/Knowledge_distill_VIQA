"""
End-to-end pipeline orchestrator for VLSP 2025 Financial Numerical Reasoning.
Knowledge Distillation pipeline: Data Prep → Teacher Distill → SFT → GRPO → Inference → Evaluate
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from pipeline.config import load_config, save_config, PipelineConfig


PHASES = ["data_prep", "teacher", "sft", "grpo", "inference", "evaluate", "all"]


def run_phase(phase: str, cfg: PipelineConfig, state: dict) -> dict:
    """Run a single pipeline phase."""
    if phase == "data_prep":
        from pipeline.data_prep import run_data_prep
        paths = run_data_prep(cfg)
        state.update(paths)

    elif phase == "teacher":
        from pipeline.teacher_distill import run_teacher_distillation
        teacher_input = state.get("teacher_input")
        distilled_path = run_teacher_distillation(cfg, teacher_input)
        state["distilled_sft"] = distilled_path

    elif phase == "sft":
        from pipeline.train_sft import run_sft_training
        train_path = state.get("distilled_sft") or state.get("sft_train")
        valid_path = state.get("sft_valid")
        model_path = run_sft_training(cfg, train_path, valid_path)
        state["sft_model"] = model_path

    elif phase == "grpo":
        from pipeline.train_grpo import run_grpo_training
        sft_model = state.get("sft_model")
        model_path = run_grpo_training(cfg, sft_model)
        state["grpo_model"] = model_path

    elif phase == "inference":
        from pipeline.inference import run_inference
        model_path = state.get("grpo_model") or state.get("sft_model")
        test_data = state.get("sft_valid")
        predictions_path = run_inference(cfg, model_path, test_data)
        state["predictions"] = predictions_path

    elif phase == "evaluate":
        from pipeline.evaluate import run_evaluation
        predictions = state.get("predictions")
        if predictions:
            output_path = str(Path(cfg.project_root) / cfg.data.output_dir / "eval_results.json")
            results = run_evaluation(predictions, output_path)
            state["eval_results"] = results
        else:
            print("No predictions found. Run inference first.")

    return state


def run_pipeline(cfg: PipelineConfig, phases: list[str], resume_state: dict = None) -> dict:
    """Run the full pipeline (or selected phases)."""
    state = resume_state or {}

    if "all" in phases:
        phases = ["data_prep", "teacher", "sft", "grpo", "inference", "evaluate"]

    state_path = Path(cfg.project_root) / cfg.data.output_dir / "pipeline_state.json"

    # Load existing state if resuming
    if state_path.exists() and not state:
        with open(state_path, "r") as f:
            state = json.load(f)
        print(f"Resumed pipeline state from {state_path}")

    for phase in phases:
        print(f"\n{'#'*60}")
        print(f"# Phase: {phase.upper()}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}\n")

        try:
            state = run_phase(phase, cfg, state)
            state[f"{phase}_completed"] = True

            # Save state after each phase
            state_save = {k: v for k, v in state.items() if not isinstance(v, dict) or k == "eval_results"}
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w") as f:
                json.dump(state_save, f, ensure_ascii=False, indent=2)

            print(f"\n✓ Phase {phase} completed successfully")

        except Exception as e:
            print(f"\n✗ Phase {phase} failed: {e}")
            import traceback
            traceback.print_exc()
            break

    return state


def main():
    parser = argparse.ArgumentParser(
        description="VLSP 2025 Knowledge Distillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on P100
  python -m pipeline.run --gpu-profile p100_16gb --phases all

  # Run only data prep + SFT
  python -m pipeline.run --phases data_prep sft

  # Run on A100 with custom config
  python -m pipeline.run --gpu-profile a100_80gb --config configs/custom.yaml

  # Resume from inference phase
  python -m pipeline.run --phases inference evaluate

Available phases: data_prep, teacher, sft, grpo, inference, evaluate, all
GPU profiles: p100_16gb, t4_16gb, a100_40gb, a100_80gb, h100_80gb
        """,
    )
    parser.add_argument(
        "--gpu-profile", default="p100_16gb",
        choices=["p100_16gb", "rtx6000_96gb", "rtx6000_96gb_35b", "rtx6000_96gb_122b", "a100_80gb"],
        help="GPU profile to use",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--phases", nargs="+", default=["all"], choices=PHASES,
        help="Pipeline phases to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit data samples (for testing)")

    args = parser.parse_args()

    overrides = {}
    if args.max_samples:
        overrides["data"] = {"max_samples": args.max_samples}

    cfg = load_config(gpu_profile=args.gpu_profile, config_path=args.config, overrides=overrides)

    print(f"Pipeline: {cfg.project_name}")
    print(f"GPU Profile: {args.gpu_profile}")
    print(f"Teacher: {cfg.model.teacher_model}")
    print(f"Student: {cfg.model.student_model}")
    print(f"Phases: {args.phases}")
    print()

    if args.dry_run:
        # Save config for inspection
        config_path = str(Path(cfg.project_root) / cfg.data.output_dir / "config.yaml")
        save_config(cfg, config_path)
        print(f"Dry run complete. Config saved to {config_path}")
        return

    run_pipeline(cfg, args.phases)


if __name__ == "__main__":
    main()
