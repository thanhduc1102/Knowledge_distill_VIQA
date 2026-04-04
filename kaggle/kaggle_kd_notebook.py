"""
VLSP 2025 - Knowledge Distillation Pipeline for Vietnamese Financial Numerical Reasoning
=========================================================================================
Kaggle Notebook for OFFLINE execution on RTX 6000 Pro 96GB.

HOW TO USE THIS NOTEBOOK:
  1. Open a new Kaggle notebook (or this one if imported as a notebook)
  2. Set Accelerator → GPU RTX 6000 Pro / T4 / P100 in Settings
  3. Add ALL required Input Datasets (see CELL 0 checklist)
  4. Run cells ONE BY ONE from top to bottom (do NOT "Run All" at once)

Required Kaggle Input Datasets:
  - thanhduc1108/vlsp2025-kd-pipeline     → Pipeline source code
  - thanhduc1108/vlsp2025-kd-wheels       → Python wheels (offline install)
  - thanhduc1108/finqa-en                 → FinQA English dataset
  - thanhduc1108/vinumericalqa-private     → ViNumQA Vietnamese dataset
  - thanhduc1108/qwen-35-27b (Model)      → Teacher model (Qwen3.5-27B)
  - thanhduc1108/qwen-35-4b  (Model)      → Student model (Qwen3.5-4B)
"""


# ═══════════════════════════════════════════════════════════════════════
# CELL 0: Pre-flight Checklist — Run this FIRST to verify all inputs
# ═══════════════════════════════════════════════════════════════════════
import os
from pathlib import Path

REQUIRED_INPUTS = {
    "/kaggle/input/vlsp2025-kd-pipeline":   "Pipeline code   (Dataset: thanhduc1108/vlsp2025-kd-pipeline)",
    "/kaggle/input/finqa-en":               "FinQA dataset   (Dataset: thanhduc1108/finqa-en)",
    "/kaggle/input/vinumericalqa-private":  "ViNumQA dataset (Dataset: thanhduc1108/vinumericalqa-private)",
}
OPTIONAL_INPUTS = {
    "/kaggle/input/vlsp2025-kd-wheels":     "Offline wheels  (Dataset: thanhduc1108/vlsp2025-kd-wheels)",
}

print("=" * 65)
print("CELL 0: Pre-flight Input Checklist")
print("=" * 65)
all_ok = True
for path, label in REQUIRED_INPUTS.items():
    exists = os.path.exists(path)
    status = "✓ FOUND  " if exists else "✗ MISSING"
    print(f"  [{status}] {label}")
    print(f"            path: {path}")
    if not exists:
        all_ok = False

for path, label in OPTIONAL_INPUTS.items():
    exists = os.path.exists(path)
    status = "✓ found  " if exists else "○ absent "
    print(f"  [{status}] {label}  (optional)")

print()
if not all_ok:
    print("STOP: Add the MISSING datasets above before continuing.")
    print("  Notebook -> top-right panel -> '+ Add Data' -> search by dataset name")
    raise SystemExit("Missing required input datasets. Add them first, then re-run.")
else:
    print("All required inputs present. Proceed to Cell 1.")

import subprocess, sys
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                         "--format=csv,noheader"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"\nGPU: {result.stdout.strip()}")
else:
    print("\nWARNING: nvidia-smi not found. Enable GPU: Notebook -> Settings -> Accelerator -> GPU")


# ═══════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════
import subprocess, os, sys
from pathlib import Path
try:
    from packaging.version import Version
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "packaging"], check=True)
    from packaging.version import Version

WHEELS_DIR = "/kaggle/input/vlsp2025-kd-wheels"
# Qwen3.5 requires transformers >= 4.52.0 for native model_type recognition.
# Even older versions work IF trust_remote_code=True is properly handled,
# but >= 5.0 is recommended for full support.
MIN_TRANSFORMERS_VERSION = "4.52.0"

def _get_transformers_version() -> str:
    """Get currently installed transformers version."""
    try:
        import transformers
        return transformers.__version__
    except ImportError:
        return "0.0.0"

def _install_packages(packages: list[str], extra_args: list[str] = None):
    """Install packages via pip."""
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    return subprocess.run(cmd, capture_output=True, text=True)

# Step 1: Try offline wheels first
if os.path.exists(WHEELS_DIR):
    wheel_files = list(Path(WHEELS_DIR).glob("*.whl"))
    print(f"Installing {len(wheel_files)} wheels from offline cache...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "--no-index", "--find-links", WHEELS_DIR,
         "transformers", "peft", "accelerate", "datasets",
         "bitsandbytes", "trl", "safetensors", "sentencepiece",
         "protobuf", "sympy", "pyyaml", "pyarrow", "tqdm", "pandas",
         "huggingface_hub", "tokenizers"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Offline installation complete.")
    else:
        print(f"Offline wheels failed:\n{result.stderr[-300:]}")
        print("Falling back to pip install...")
        _install_packages([
            "transformers>=5.0", "peft>=0.18", "accelerate>=1.0",
            "datasets>=4.0", "bitsandbytes>=0.49", "trl>=1.0",
            "safetensors", "sentencepiece", "protobuf", "sympy"])
else:
    print("Wheels not available, installing from pip (requires internet)...")
    _install_packages([
        "transformers>=5.0", "peft>=0.18", "accelerate>=1.0",
        "datasets>=4.0", "bitsandbytes>=0.49", "trl>=1.0",
        "safetensors", "sentencepiece", "protobuf", "sympy"])

# Step 2: CRITICAL — Verify transformers version is new enough for Qwen3.5
# Kaggle pre-installs an old transformers (e.g., 4.46) that does NOT know
# model_type='qwen3_5'. We MUST upgrade if it's too old.
# Reimport after install to pick up new version
if "transformers" in sys.modules:
    del sys.modules["transformers"]
cur_ver = _get_transformers_version()
print(f"\ntransformers version: {cur_ver}")

if Version(cur_ver) < Version(MIN_TRANSFORMERS_VERSION):
    print(f"  ⚠ Version {cur_ver} < {MIN_TRANSFORMERS_VERSION} — Qwen3.5 not natively supported!")
    print("  Upgrading transformers from PyPI...")
    result = _install_packages(["transformers>=5.0"])
    if result.returncode != 0:
        print("  PyPI upgrade failed, trying from GitHub source...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "git+https://github.com/huggingface/transformers.git"],
            capture_output=True, text=True)
    # Clear cached module to pick up new version
    for mod_name in list(sys.modules):
        if mod_name.startswith("transformers"):
            del sys.modules[mod_name]
    cur_ver = _get_transformers_version()
    print(f"  After upgrade: transformers {cur_ver}")
    if Version(cur_ver) < Version(MIN_TRANSFORMERS_VERSION):
        print(f"  WARNING: Still {cur_ver} < {MIN_TRANSFORMERS_VERSION}.")
        print("  The pipeline has a fallback (trust_remote_code), but native support is preferred.")
        print("  Consider enabling Internet access in notebook settings.")
    else:
        print(f"  ✓ transformers {cur_ver} — Qwen3.5 supported natively.")
else:
    print(f"  ✓ transformers {cur_ver} — Qwen3.5 supported natively.")

# Step 3: Flash Attention (optional — RTX/A100/H100 only, not P100/T4)
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "flash-attn", "--no-build-isolation"],
                   check=True, capture_output=True, timeout=300)
    print("Flash Attention installed.")
except Exception:
    print("Flash Attention not available (optional, continuing without it).")


# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Setup Working Directory, sys.path, and Dataset Links
# ═══════════════════════════════════════════════════════════════════════
import os, sys, shutil
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Register WORK_DIR on sys.path BEFORE copying (slot is created now)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

# Step 2: Copy pipeline source code into WORK_DIR
CODE_SRC = Path("/kaggle/input/vlsp2025-kd-pipeline")
if not CODE_SRC.exists():
    raise FileNotFoundError(
        f"Pipeline code not found at {CODE_SRC}\n"
        "Add dataset 'thanhduc1108/vlsp2025-kd-pipeline' as notebook input."
    )

for d in ["pipeline", "src", "configs"]:
    src = CODE_SRC / d
    dst = WORK_DIR / d
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n = sum(1 for _ in dst.rglob("*") if _.is_file())
        print(f"  Copied {d}/ ({n} files)")
    else:
        print(f"  WARNING: {d}/ missing from pipeline dataset")

if (CODE_SRC / "requirements.txt").exists():
    shutil.copy2(CODE_SRC / "requirements.txt", WORK_DIR / "requirements.txt")

# Step 3: Verify the copy succeeded and pipeline is importable
pipeline_init = WORK_DIR / "pipeline" / "__init__.py"
if not pipeline_init.exists():
    raise FileNotFoundError(
        f"pipeline/__init__.py not found after copy.\n"
        "The vlsp2025-kd-pipeline dataset may be outdated. Re-upload with:\n"
        "  python scripts/kaggle_upload.py --upload-code"
    )
print(f"\nPipeline code ready at: {WORK_DIR / 'pipeline'}")

# Step 4: Set working directory so relative paths work
os.chdir(WORK_DIR)

# Step 5: Symlink datasets into expected locations
DATASET_DIR = WORK_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for src_path, dst_name, label in [
    ("/kaggle/input/vinumericalqa-private", "viNumericalQA_private", "ViNumQA"),
    ("/kaggle/input/finqa-en",              "finqa_en",              "FinQA"),
]:
    src = Path(src_path)
    dst = DATASET_DIR / dst_name
    if src.exists():
        if not dst.exists():
            dst.symlink_to(src)
        n = sum(1 for _ in src.rglob("*") if _.is_file())
        print(f"  {label} linked ({n} files): {dst}")
    else:
        print(f"  WARNING: {label} not found at {src_path}")

print(f"\nWorking directory : {WORK_DIR}")
print(f"sys.path[0]       : {sys.path[0]}")
print(f"Contents          : {[p.name for p in sorted(WORK_DIR.iterdir())]}")

# Quick import smoke-test
try:
    import importlib
    spec = importlib.util.spec_from_file_location(
        "pipeline.config", str(WORK_DIR / "pipeline" / "config.py"))
    print("Pipeline import check: OK")
except Exception as e:
    print(f"Pipeline import check FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Verify Environment & Detect GPU Profile
# ═══════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

# Idempotent sys.path guard (safe even if cells run out of order)
WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import torch

print(f"PyTorch     : {torch.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected!\n"
        "Enable GPU: Notebook -> Settings -> Accelerator -> GPU T4 / P100 / RTX 6000"
    )

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU         : {gpu_name}  ({vram_gb:.1f} GB VRAM)")

import transformers, peft, accelerate, datasets
from packaging.version import Version
print(f"Transformers: {transformers.__version__}")
if Version(transformers.__version__) < Version("4.52.0"):
    print("  ⚠ WARNING: transformers < 4.52 — Qwen3.5 may not load natively!")
    print("  The pipeline will try trust_remote_code fallback, but upgrading is recommended.")
print(f"PEFT        : {peft.__version__}")
print(f"Accelerate  : {accelerate.__version__}")
print(f"Datasets    : {datasets.__version__}")

try:
    import flash_attn
    HAS_FLASH = True
    print(f"Flash Attn  : {flash_attn.__version__}")
except ImportError:
    HAS_FLASH = False
    print("Flash Attn  : not available")

if "6000" in gpu_name or vram_gb > 90:
    GPU_PROFILE = "rtx6000_96gb"
elif "A100" in gpu_name and vram_gb > 70:
    GPU_PROFILE = "a100_80gb"
elif "P100" in gpu_name or vram_gb < 20:
    GPU_PROFILE = "p100_16gb"
else:
    GPU_PROFILE = "p100_16gb"

print(f"\nGPU Profile : {GPU_PROFILE}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Resolve Model Paths (Kaggle offline -> HuggingFace fallback)
# ═══════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

TEACHER_MODEL_ID = "Qwen/Qwen3.5-27B"
STUDENT_MODEL_ID = "Qwen/Qwen3.5-4B"


def resolve_model_path(model_id: str) -> str:
    """Check /kaggle/input/* and /kaggle/models/* for weights; fall back to HF id."""
    base_name = model_id.split("/")[-1]
    slugs = set()
    for name in [base_name, base_name.lower()]:
        slugs.add(name)
        slugs.add(name.replace(".", "-").replace("_", "-"))
        slugs.add(name.replace(".", "_").replace("-", "_"))

    for root in [Path("/kaggle/input"), Path("/kaggle/models")]:
        if not root.exists():
            continue
        for entry in root.iterdir():
            if entry.name.lower() in {s.lower() for s in slugs}:
                for candidate in [entry, *entry.rglob("config.json")]:
                    check_dir = candidate if candidate.is_dir() else candidate.parent
                    has_weights = (
                        any(check_dir.glob("*.safetensors")) or
                        any(check_dir.glob("*.bin")) or
                        (check_dir / "config.json").exists()
                    )
                    if has_weights:
                        print(f"  Found offline: {check_dir}")
                        return str(check_dir)

    print(f"  Not found offline -> using HuggingFace: {model_id}")
    return model_id


print("Resolving teacher model...")
TEACHER_PATH = resolve_model_path(TEACHER_MODEL_ID)
print("Resolving student model...")
STUDENT_PATH = resolve_model_path(STUDENT_MODEL_ID)
print(f"\nTeacher: {TEACHER_PATH}")
print(f"Student: {STUDENT_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Configure Pipeline
# ═══════════════════════════════════════════════════════════════════════
import os, sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.config import load_config, save_config

config_overrides = {
    "model": {
        "teacher_model": TEACHER_PATH,
        "student_model": STUDENT_PATH,
        "use_flash_attention": HAS_FLASH,
    },
    "data": {
        "vinumqa_train":        "dataset/viNumericalQA_private/train.json",
        "vinumqa_valid":        "dataset/viNumericalQA_private/valid.json",
        "vinumqa_test":         "dataset/viNumericalQA_private/test.json",
        "vinumqa_private_test": "dataset/viNumericalQA_private/private_test.json",
        "finqa_dir":            "dataset/finqa_en",
    },
}

cfg = load_config(gpu_profile=GPU_PROFILE, overrides=config_overrides)

print(f"\n{'='*60}")
print("Pipeline Configuration")
print(f"{'='*60}")
print(f"  GPU Profile   : {GPU_PROFILE}")
print(f"  Teacher model : {cfg.model.teacher_model}")
print(f"  Student model : {cfg.model.student_model}")
print(f"  Teacher quant : {cfg.model.teacher_quantization}")
print(f"  Flash Attn    : {cfg.model.use_flash_attention}")
print(f"  dtype         : {cfg.model.torch_dtype}")
print(f"  SFT epochs    : {cfg.sft.num_epochs}  |  LoRA r={cfg.sft.lora_r}")
print(f"  GRPO epochs   : {cfg.grpo.num_epochs}  |  N generations={cfg.grpo.num_generations}")
print(f"  Inference N   : {cfg.inference.num_candidates}")
print(f"{'='*60}")

save_config(cfg, str(WORK_DIR / "data/pipeline/config.yaml"))
print("Config saved.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Phase 1 — Data Preparation
# ═══════════════════════════════════════════════════════════════════════
import os, sys, time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.data_prep import run_data_prep

print("=" * 60)
print("PHASE 1: DATA PREPARATION")
print("=" * 60)

t0 = time.time()
data_paths = run_data_prep(cfg)
print(f"\nData prep completed in {time.time()-t0:.1f}s")
for k, v in data_paths.items():
    print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Phase 2 — Teacher Distillation
# ═══════════════════════════════════════════════════════════════════════
import gc, os, sys, time, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.teacher_distill import run_teacher_distillation

print("=" * 60)
print("PHASE 2: TEACHER DISTILLATION")
print("=" * 60)

t0 = time.time()
distilled_path = run_teacher_distillation(cfg, data_paths["teacher_input"])
print(f"\nTeacher distillation completed in {time.time()-t0:.1f}s")
print(f"  Distilled SFT data: {distilled_path}")

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Phase 3 — SFT Training
# ═══════════════════════════════════════════════════════════════════════
import gc, os, sys, time, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.train_sft import run_sft_training

print("=" * 60)
print("PHASE 3: SUPERVISED FINE-TUNING (SFT)")
print("=" * 60)

t0 = time.time()
sft_model_path = run_sft_training(
    cfg,
    train_path=distilled_path,
    valid_path=data_paths["sft_valid"],
)
print(f"\nSFT training completed in {time.time()-t0:.1f}s")
print(f"  SFT model: {sft_model_path}")

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Phase 4 — GRPO Training
# (Comment out / skip to use SFT-only: set grpo_model_path = sft_model_path)
# ═══════════════════════════════════════════════════════════════════════
import gc, os, sys, time, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.train_grpo import run_grpo_training

print("=" * 60)
print("PHASE 4: GRPO WITH PCPO REWARD")
print("=" * 60)

t0 = time.time()
grpo_model_path = run_grpo_training(cfg, sft_model_path)
print(f"\nGRPO training completed in {time.time()-t0:.1f}s")
print(f"  GRPO model: {grpo_model_path}")

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Phase 5 — Inference with Majority Voting
# ═══════════════════════════════════════════════════════════════════════
import gc, json, os, sys, time, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.inference import run_inference

print("=" * 60)
print("PHASE 5: MULTI-PATH INFERENCE + MAJORITY VOTING")
print("=" * 60)

# Use best available model: GRPO > SFT
final_model = grpo_model_path if Path(grpo_model_path).exists() else sft_model_path
print(f"Using model : {final_model}")

# data_paths["sft_valid"] was produced by run_data_prep — correct SFT format for inference
test_input_path = data_paths["sft_valid"]
print(f"Test input  : {test_input_path}")

t0 = time.time()
predictions_path = run_inference(cfg, final_model, test_input_path)
print(f"\nInference completed in {time.time()-t0:.1f}s")
print(f"  Predictions: {predictions_path}")

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Phase 6 — Evaluation (EA + PA)
# ═══════════════════════════════════════════════════════════════════════
import gc, os, sys, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.evaluate import evaluate_against_dataset

print("=" * 60)
print("PHASE 6: EVALUATION (EA + PA)")
print("=" * 60)

# Original ViNumQA valid JSON supplies table data needed for proper EA/PA
test_dataset_path = str(WORK_DIR / "dataset/viNumericalQA_private/valid.json")
eval_output       = str(WORK_DIR / "data/pipeline/eval_results.json")

results = evaluate_against_dataset(predictions_path, test_dataset_path, eval_output)

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Baseline Comparison — Zero-shot Student Model (optional)
# Skip this cell if you only need KD results.
# ═══════════════════════════════════════════════════════════════════════
import gc, os, shutil, sys, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.inference import run_inference
from pipeline.evaluate import evaluate_against_dataset

pipeline_out = WORK_DIR / "data/pipeline"

# run_inference always writes to predictions.json — back up KD results first
kd_backup = str(pipeline_out / "predictions_kd.json")
shutil.copy2(predictions_path, kd_backup)
print(f"KD predictions backed up -> {kd_backup}")

print(f"\nRunning zero-shot baseline with: {STUDENT_PATH}")
run_inference(cfg, model_path=STUDENT_PATH, test_data_path=test_input_path)

# Rename baseline output, restore KD predictions
baseline_pred_path = str(pipeline_out / "predictions_baseline.json")
shutil.copy2(str(pipeline_out / "predictions.json"), baseline_pred_path)
shutil.copy2(kd_backup, str(pipeline_out / "predictions.json"))
print(f"Baseline predictions saved -> {baseline_pred_path}")

test_dataset_path = str(WORK_DIR / "dataset/viNumericalQA_private/valid.json")
baseline_results = evaluate_against_dataset(
    baseline_pred_path,
    test_dataset_path,
    str(pipeline_out / "eval_baseline.json"),
)

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Results Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\n{'Model':<30} {'EA':>8} {'PA':>8} {'Valid%':>8}")
print("-" * 60)
print(f"{'Baseline (zero-shot)':<30} "
      f"{baseline_results['execution_accuracy']:>7.2%} "
      f"{baseline_results['program_accuracy']:>7.2%} "
      f"{baseline_results['valid_rate']:>7.2%}")
print(f"{'After KD (SFT+GRPO)':<30} "
      f"{results['execution_accuracy']:>7.2%} "
      f"{results['program_accuracy']:>7.2%} "
      f"{results['valid_rate']:>7.2%}")

ea_delta = results['execution_accuracy'] - baseline_results['execution_accuracy']
pa_delta = results['program_accuracy']   - baseline_results['program_accuracy']
print(f"\nImprovement: EA {ea_delta:+.2%},  PA {pa_delta:+.2%}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 14: Save All Outputs
# ═══════════════════════════════════════════════════════════════════════
import json, os, shutil, sys
from pathlib import Path

WORK_DIR     = Path("/kaggle/working/vlsp2025")
output_dir   = WORK_DIR / "outputs"
pipeline_out = WORK_DIR / "data/pipeline"
output_dir.mkdir(parents=True, exist_ok=True)

if Path(final_model).exists():
    final_model_output = output_dir / "final_model"
    if final_model_output.exists():
        shutil.rmtree(final_model_output)
    shutil.copytree(final_model, final_model_output)
    print(f"Final model saved -> {final_model_output}")

for fname in ["eval_results.json", "eval_baseline.json",
              "predictions.json", "predictions_kd.json", "predictions_baseline.json",
              "config.yaml"]:
    src = pipeline_out / fname
    if src.exists():
        shutil.copy2(src, output_dir / fname)
        print(f"  Copied {fname}")

summary = {
    "gpu_profile":   GPU_PROFILE,
    "teacher_model": TEACHER_MODEL_ID,
    "student_model": STUDENT_MODEL_ID,
    "baseline": {
        "EA":         baseline_results["execution_accuracy"],
        "PA":         baseline_results["program_accuracy"],
        "valid_rate": baseline_results["valid_rate"],
    },
    "after_kd": {
        "EA":         results["execution_accuracy"],
        "PA":         results["program_accuracy"],
        "valid_rate": results["valid_rate"],
    },
    "improvement": {"EA_delta": ea_delta, "PA_delta": pa_delta},
}
with open(output_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nAll outputs -> {output_dir}")
print(f"Files: {[p.name for p in sorted(output_dir.iterdir())]}")
print("\nDone! Check /kaggle/working/vlsp2025/outputs/ for final results.")
