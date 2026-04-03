"""
VLSP 2025 - Knowledge Distillation Pipeline for Vietnamese Financial Numerical Reasoning
=========================================================================================
Kaggle Notebook for OFFLINE execution on RTX 6000 Pro 96GB.

Required Kaggle Input Datasets:
  1. thanhduc1108/vlsp2025-kd-pipeline    → Pipeline code
  2. thanhduc1108/vlsp2025-kd-wheels      → Python wheels for offline install
  3. thanhduc1108/finqa-en                → FinQA English dataset
  4. thanhduc1108/vinumericalqa-private    → ViNumQA Vietnamese dataset
  5. thanhduc1108/qwen_35_27b             → Teacher model (Qwen3.5-27B)
  6. thanhduc1108/qwen_35_4b              → Student model (Qwen3.5-4B)

GPU: RTX 6000 Pro 96GB (Kaggle)
Estimated Runtime: ~6-8 hours for full pipeline
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies (Offline)
# ═══════════════════════════════════════════════════════════════════════
import subprocess, os, sys

# Install from pre-downloaded wheels (offline)
WHEELS_DIR = "/kaggle/input/vlsp2025-kd-wheels"
if os.path.exists(WHEELS_DIR):
    print("Installing dependencies from offline wheels...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", WHEELS_DIR,
        "transformers", "peft", "accelerate", "datasets",
        "bitsandbytes", "trl", "safetensors", "sentencepiece",
        "protobuf", "sympy", "pyyaml", "pyarrow", "tqdm", "pandas",
        "huggingface_hub", "tokenizers",
    ], check=False, capture_output=True)
    print("Offline installation complete.")
else:
    print("Wheels not found, installing from pip...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=5.0", "peft>=0.18", "accelerate>=1.0",
        "datasets>=4.0", "bitsandbytes>=0.49", "trl>=1.0",
        "safetensors", "sentencepiece", "protobuf", "sympy",
    ], check=False, capture_output=True)

# Try flash-attn (optional, may fail on some GPUs)
try:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation", "-q"
    ], check=True, capture_output=True, timeout=300)
    print("Flash Attention installed.")
except Exception:
    print("Flash Attention not available (optional, continuing without it).")

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Setup Project Code & Data
# ═══════════════════════════════════════════════════════════════════════
import shutil
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Copy pipeline code
CODE_SRC = Path("/kaggle/input/vlsp2025-kd-pipeline")
if CODE_SRC.exists():
    for d in ["pipeline", "src", "configs"]:
        src = CODE_SRC / d
        dst = WORK_DIR / d
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    for f in ["requirements.txt"]:
        src = CODE_SRC / f
        if src.exists():
            shutil.copy2(src, WORK_DIR / f)
    print(f"Pipeline code copied to {WORK_DIR}")

# Setup dataset paths
DATASET_DIR = WORK_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Link ViNumQA dataset
vinumqa_src = Path("/kaggle/input/vinumericalqa-private")
vinumqa_dst = DATASET_DIR / "viNumericalQA_private"
if vinumqa_src.exists() and not vinumqa_dst.exists():
    vinumqa_dst.symlink_to(vinumqa_src)
    print(f"ViNumQA linked: {vinumqa_dst}")

# Link FinQA dataset
finqa_src = Path("/kaggle/input/finqa-en")
finqa_dst = DATASET_DIR / "finqa_en"
if finqa_src.exists() and not finqa_dst.exists():
    finqa_dst.symlink_to(finqa_src)
    print(f"FinQA linked: {finqa_dst}")

# Add to Python path
sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

print(f"\nWorking directory: {WORK_DIR}")
print(f"Contents: {list(WORK_DIR.iterdir())}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Verify Environment
# ═══════════════════════════════════════════════════════════════════════
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import transformers, peft, accelerate, datasets
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"Datasets: {datasets.__version__}")

try:
    import flash_attn
    HAS_FLASH = True
    print(f"Flash Attention: {flash_attn.__version__}")
except ImportError:
    HAS_FLASH = False
    print("Flash Attention: Not available")

# Detect GPU profile
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

if "6000" in gpu_name or vram_gb > 90:
    GPU_PROFILE = "rtx6000_96gb"
elif "P100" in gpu_name:
    GPU_PROFILE = "p100_16gb"
elif "A100" in gpu_name and vram_gb > 70:
    GPU_PROFILE = "a100_80gb"
else:
    GPU_PROFILE = "p100_16gb"  # fallback

print(f"\nDetected GPU Profile: {GPU_PROFILE}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Resolve Model Paths (Offline vs Online)
# ═══════════════════════════════════════════════════════════════════════

def resolve_model_path(model_id: str) -> str:
    """Resolve model path - check Kaggle input first, then HuggingFace."""
    # Check common Kaggle input patterns
    model_name = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    # Also try kebab-case variant (qwen-35-27b style from kagglehub model upload)
    model_name_kebab = model_id.split("/")[-1].lower().replace("_", "-").replace(".", "-")

    # Check various Kaggle input locations
    candidates = [
        Path(f"/kaggle/input/{model_name}"),
        Path(f"/kaggle/input/{model_name_kebab}"),
        Path(f"/kaggle/input/{model_name}/1"),  # versioned
        Path(f"/kaggle/input/{model_name_kebab}/1"),
        Path(f"/kaggle/input/{model_name}/pytorch/default/1"),  # Kaggle model format
        Path(f"/kaggle/input/{model_name_kebab}/pyTorch/default/1"),
    ]

    for path in candidates:
        if path.exists():
            # Check if it has model files
            has_model = any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or (path / "config.json").exists()
            if has_model:
                print(f"  Model found offline: {path}")
                return str(path)
            # Check subdirectories
            for subdir in path.iterdir():
                if subdir.is_dir():
                    has_model = any(subdir.glob("*.safetensors")) or (subdir / "config.json").exists()
                    if has_model:
                        print(f"  Model found offline: {subdir}")
                        return str(subdir)

    print(f"  Model not found offline, using HuggingFace: {model_id}")
    return model_id


# Resolve teacher and student model paths
TEACHER_MODEL_ID = "Qwen/Qwen3.5-27B"
STUDENT_MODEL_ID = "Qwen/Qwen3.5-4B"

print("Resolving teacher model...")
TEACHER_PATH = resolve_model_path(TEACHER_MODEL_ID)
print("Resolving student model...")
STUDENT_PATH = resolve_model_path(STUDENT_MODEL_ID)

# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Configure Pipeline
# ═══════════════════════════════════════════════════════════════════════
from pipeline.config import load_config, save_config

# Build config with resolved model paths
config_overrides = {
    "model": {
        "teacher_model": TEACHER_PATH,
        "student_model": STUDENT_PATH,
        "use_flash_attention": HAS_FLASH,
    },
    "data": {
        "vinumqa_train": "dataset/viNumericalQA_private/train.json",
        "vinumqa_valid": "dataset/viNumericalQA_private/valid.json",
        "vinumqa_test": "dataset/viNumericalQA_private/test.json",
        "vinumqa_private_test": "dataset/viNumericalQA_private/private_test.json",
        "finqa_dir": "dataset/finqa_en",
    },
}

cfg = load_config(gpu_profile=GPU_PROFILE, overrides=config_overrides)

print(f"\n{'='*60}")
print(f"Pipeline Configuration")
print(f"{'='*60}")
print(f"  GPU Profile:    {GPU_PROFILE}")
print(f"  Teacher:        {cfg.model.teacher_model}")
print(f"  Student:        {cfg.model.student_model}")
print(f"  Teacher Quant:  {cfg.model.teacher_quantization}")
print(f"  Flash Attn:     {cfg.model.use_flash_attention}")
print(f"  SFT Epochs:     {cfg.sft.num_epochs}")
print(f"  GRPO Epochs:    {cfg.grpo.num_epochs}")
print(f"  Inference N:    {cfg.inference.num_candidates}")
print(f"{'='*60}")

# Save config for reproducibility
save_config(cfg, str(WORK_DIR / "data/pipeline/config.yaml"))

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Phase 1 - Data Preparation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: DATA PREPARATION")
print("="*60)

from pipeline.data_prep import run_data_prep
import time

t0 = time.time()
data_paths = run_data_prep(cfg)
print(f"Data prep completed in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Phase 2 - Teacher Distillation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2: TEACHER DISTILLATION (Knowledge Distillation)")
print("="*60)

from pipeline.teacher_distill import run_teacher_distillation

t0 = time.time()
distilled_path = run_teacher_distillation(cfg, data_paths["teacher_input"])
print(f"Teacher distillation completed in {time.time()-t0:.1f}s")

# Clean up GPU memory
import gc
gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Phase 3 - SFT Training
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 3: SUPERVISED FINE-TUNING (SFT)")
print("="*60)

from pipeline.train_sft import run_sft_training

t0 = time.time()
sft_model_path = run_sft_training(
    cfg,
    train_path=distilled_path,
    valid_path=data_paths["sft_valid"],
)
print(f"SFT training completed in {time.time()-t0:.1f}s")

gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Phase 4 - GRPO Training (Optional - comment out to skip)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 4: GRPO WITH PCPO REWARD")
print("="*60)

from pipeline.train_grpo import run_grpo_training

t0 = time.time()
grpo_model_path = run_grpo_training(cfg, sft_model_path)
print(f"GRPO training completed in {time.time()-t0:.1f}s")

gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Phase 5 - Inference with Majority Voting
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 5: MULTI-PATH INFERENCE + MAJORITY VOTING")
print("="*60)

from pipeline.inference import run_inference
import json

# Use GRPO model if available, otherwise SFT model
final_model = grpo_model_path if os.path.exists(grpo_model_path) else sft_model_path

# Prepare test data for inference
test_dataset_path = str(WORK_DIR / "dataset/viNumericalQA_private/test.json")
with open(test_dataset_path, 'r') as f:
    test_data_raw = json.load(f)

# Convert test data to inference format
from src.assets.template import prompt as prompt_template
from pipeline.data_prep import build_user_prompt, prepare_sft_dataset

test_sft = prepare_sft_dataset(test_data_raw, prompt_template, use_program_re=False)
test_input_path = str(WORK_DIR / "data/pipeline/test_input.json")
with open(test_input_path, 'w') as f:
    json.dump(test_sft, f, ensure_ascii=False, indent=2)

t0 = time.time()
predictions_path = run_inference(cfg, final_model, test_input_path)
print(f"Inference completed in {time.time()-t0:.1f}s")

gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Phase 6 - Evaluation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 6: EVALUATION (EA + PA)")
print("="*60)

from pipeline.evaluate import evaluate_against_dataset, run_evaluation

# Evaluate against original test dataset (full evaluation with table data)
eval_output = str(WORK_DIR / "data/pipeline/eval_results.json")
results = evaluate_against_dataset(
    predictions_path,
    test_dataset_path,
    eval_output,
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Baseline Comparison (Optional)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BASELINE COMPARISON: Zero-shot Student Model")
print("="*60)

# Run zero-shot inference with untrained student for comparison
baseline_predictions = run_inference(
    cfg,
    model_path=STUDENT_PATH,
    test_data_path=test_input_path,
)

# Rename to avoid overwriting
baseline_pred_path = str(WORK_DIR / "data/pipeline/predictions_baseline.json")
shutil.move(baseline_predictions, baseline_pred_path)

baseline_results = evaluate_against_dataset(
    baseline_pred_path,
    test_dataset_path,
    str(WORK_DIR / "data/pipeline/eval_baseline.json"),
)

# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Results Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n{'Model':<30} {'EA':>8} {'PA':>8} {'Valid%':>8}")
print("-" * 60)
print(f"{'Baseline (zero-shot)':<30} {baseline_results['execution_accuracy']:>7.2%} {baseline_results['program_accuracy']:>7.2%} {baseline_results['valid_rate']:>7.2%}")
print(f"{'After KD (SFT+GRPO)':<30} {results['execution_accuracy']:>7.2%} {results['program_accuracy']:>7.2%} {results['valid_rate']:>7.2%}")

ea_delta = results['execution_accuracy'] - baseline_results['execution_accuracy']
pa_delta = results['program_accuracy'] - baseline_results['program_accuracy']
print(f"\nImprovement: EA {ea_delta:+.2%}, PA {pa_delta:+.2%}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 14: Save Outputs
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SAVING OUTPUTS")
print("="*60)

output_dir = WORK_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Copy final model
final_model_output = output_dir / "final_model"
if Path(final_model).exists():
    if final_model_output.exists():
        shutil.rmtree(final_model_output)
    shutil.copytree(final_model, final_model_output)
    print(f"Final model saved to: {final_model_output}")

# Copy evaluation results
for f in ["eval_results.json", "eval_baseline.json", "predictions.json", "predictions_baseline.json"]:
    src = WORK_DIR / "data/pipeline" / f
    if src.exists():
        shutil.copy2(src, output_dir / f)

# Summary JSON
summary = {
    "gpu_profile": GPU_PROFILE,
    "teacher_model": TEACHER_MODEL_ID,
    "student_model": STUDENT_MODEL_ID,
    "baseline": {
        "EA": baseline_results['execution_accuracy'],
        "PA": baseline_results['program_accuracy'],
        "valid_rate": baseline_results['valid_rate'],
    },
    "after_kd": {
        "EA": results['execution_accuracy'],
        "PA": results['program_accuracy'],
        "valid_rate": results['valid_rate'],
    },
    "improvement": {
        "EA_delta": ea_delta,
        "PA_delta": pa_delta,
    },
}

with open(output_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to: {output_dir}")
print(f"Contents: {[f.name for f in output_dir.iterdir()]}")
print("\nDone! Check /kaggle/working/vlsp2025/outputs/ for results.")
