"""
VLSP 2025 / AAAI-27 Qwen3.5-4B Stability Run
============================================
Kaggle notebook for directly executing a verifier-native benchmark-suite smoke run.

Primary execution target:
    - Qwen/Qwen3.5-4B for both teacher and student roles

Preferred Kaggle inputs:
    - thanhduc1108/vlsp2025-kd-pipeline             -> pipeline source code
    - thanhduc1108/vlsp2025-kd-wheels               -> offline Python wheels
    - thanhduc1108/financial-reasoning-benchmarks   -> FinQA, ViNumQA, benchmark cache

If those inputs are not attached, the notebook bootstraps them directly from Kaggle.
If the model is not attached, it falls back to HuggingFace with internet enabled.

The notebook runs three small stability variants on the same benchmark manifest:
    - sft_only
    - grpo_pcpo
    - grpo_ecrl
"""


# ═══════════════════════════════════════════════════════════════════════
# CELL 0: Pre-flight Checklist
# ═══════════════════════════════════════════════════════════════════════
import os
import subprocess

PREFERRED_INPUTS = {
    "/kaggle/input/vlsp2025-kd-pipeline": "Pipeline code dataset",
    "/kaggle/input/financial-reasoning-benchmarks": "Bundled reasoning benchmarks",
    "/kaggle/input/vlsp2025-kd-wheels": "Offline wheels cache",
}

print("=" * 68)
print("CELL 0: Pre-flight Input Checklist")
print("=" * 68)

missing_inputs = []
for path, label in PREFERRED_INPUTS.items():
    exists = os.path.exists(path)
    status = "FOUND" if exists else "bootstrap"
    print(f"[{status:7}] {label}: {path}")
    if not exists:
        missing_inputs.append(path)

if missing_inputs:
    print("Some Kaggle inputs are not attached. The notebook will download public assets directly.")
else:
    print("All preferred Kaggle inputs are attached.")

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print(f"GPU: {result.stdout.strip()}")
else:
    print("WARNING: nvidia-smi unavailable. Make sure GPU is enabled in Kaggle settings.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════
import os
import subprocess
import sys
from pathlib import Path

def resolve_wheels_dir() -> Path | None:
    candidates = [
        Path("/kaggle/input/vlsp2025-kd-wheels"),
        Path("/kaggle/input/datasets/thanhduc1108/vlsp2025-kd-wheels"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in input_root.rglob("vlsp2025-kd-wheels"):
            if candidate.is_dir():
                return candidate
    return None


WHEELS_DIR = resolve_wheels_dir()

def _install_packages(packages: list[str], extra_args: list[str] | None = None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    return subprocess.run(cmd, capture_output=True, text=True)

if WHEELS_DIR is not None:
    print(f"Installing offline wheels from {WHEELS_DIR}...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "--no-index", "--find-links", str(WHEELS_DIR),
            "transformers", "peft", "accelerate", "datasets", "bitsandbytes",
            "trl", "safetensors", "sentencepiece", "protobuf", "sympy",
            "pyyaml", "pyarrow", "tqdm", "pandas", "huggingface_hub", "tokenizers", "kagglehub",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Offline installation failed. Falling back to pip.")
        result = _install_packages([
            "kagglehub",
            "transformers>=5.0", "peft>=0.18", "accelerate>=1.0", "datasets>=4.0",
            "bitsandbytes>=0.49", "trl>=1.0", "safetensors", "sentencepiece",
            "protobuf", "sympy", "pyyaml", "pyarrow", "tqdm", "pandas",
            "huggingface_hub", "tokenizers",
        ])
else:
    print("Offline wheels not found. Installing from pip.")
    result = _install_packages([
        "kagglehub",
        "transformers>=5.0", "peft>=0.18", "accelerate>=1.0", "datasets>=4.0",
        "bitsandbytes>=0.49", "trl>=1.0", "safetensors", "sentencepiece",
        "protobuf", "sympy", "pyyaml", "pyarrow", "tqdm", "pandas",
        "huggingface_hub", "tokenizers",
    ])

if result.returncode != 0:
    raise RuntimeError(result.stderr[-800:] or result.stdout[-800:])

print("Skipping flash-attn installation. The notebook will use it only if preinstalled.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Setup Working Directory and Offline Dataset Links
# ═══════════════════════════════════════════════════════════════════════
import os
import shutil
import sys
from pathlib import Path

import kagglehub

WORK_DIR = Path("/kaggle/working/vlsp2025")


def resolve_dataset_dir(input_path: str, handle: str) -> Path:
    candidate = Path(input_path)
    if candidate.exists():
        print(f"Using attached dataset: {candidate}")
        return candidate
    downloaded = Path(kagglehub.dataset_download(handle))
    print(f"Downloaded dataset {handle} -> {downloaded}")
    return downloaded


CODE_SRC = resolve_dataset_dir("/kaggle/input/vlsp2025-kd-pipeline", "thanhduc1108/vlsp2025-kd-pipeline")
BUNDLE_SRC = resolve_dataset_dir(
    "/kaggle/input/financial-reasoning-benchmarks",
    "thanhduc1108/financial-reasoning-benchmarks",
)

WORK_DIR.mkdir(parents=True, exist_ok=True)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

for directory in ["pipeline", "src", "configs"]:
    src = CODE_SRC / directory
    dst = WORK_DIR / directory
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {directory}/")

requirements_src = CODE_SRC / "requirements.txt"
if requirements_src.exists():
    shutil.copy2(requirements_src, WORK_DIR / "requirements.txt")

DATASET_DIR = WORK_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for src_name, dst_name, required in [
    ("dataset_finqa_en", "dataset_finqa_en", True),
    ("viNumericalQA_private", "viNumericalQA_private", True),
    ("benchmark_cache", "benchmark_cache", False),
    ("finchain", "finchain", False),
]:
    src = BUNDLE_SRC / src_name
    dst = DATASET_DIR / dst_name
    if src.exists():
        if dst.exists() or dst.is_symlink():
            dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
        dst.symlink_to(src)
        print(f"Linked {dst_name} -> {src}")
    elif required:
        raise FileNotFoundError(f"Required bundled dataset component missing: {src}")
    else:
        print(f"Optional benchmark component missing: {src_name}")

manifest_path = BUNDLE_SRC / "bundle_manifest.json"
if manifest_path.exists():
    print(f"Bundle manifest: {manifest_path.read_text(encoding='utf-8')[:800]}")

os.chdir(WORK_DIR)
print(f"Working directory ready: {WORK_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Verify Environment and Detect GPU Profile
# ═══════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Enable GPU in Kaggle notebook settings.")

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

try:
    import flash_attn  # noqa: F401
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

if "6000" in gpu_name or vram_gb > 90:
    GPU_PROFILE = "rtx6000_96gb"
elif "A100" in gpu_name and vram_gb > 70:
    GPU_PROFILE = "a100_80gb"
elif "P100" in gpu_name or vram_gb < 20:
    GPU_PROFILE = "p100_16gb"
else:
    GPU_PROFILE = "p100_16gb"

print(f"GPU profile: {GPU_PROFILE}")
print(f"Flash attention: {HAS_FLASH}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Resolve Model Paths
# ═══════════════════════════════════════════════════════════════════════
from pathlib import Path

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_OWNER = "thanhduc1108"

def resolve_model_path(model_id: str) -> str:
    base_name = model_id.split("/")[-1]
    slugs = set()
    for name in [base_name, base_name.lower()]:
        slugs.add(name)
        slugs.add(name.replace(".", "-").replace("_", "-"))
        slugs.add(name.replace(".", "_").replace("-", "_"))

    preferred_model_dirs = [
        Path(f"/kaggle/input/models/{MODEL_OWNER}/qwen_35_4b/transformers/default/1"),
        Path(f"/kaggle/input/models/{MODEL_OWNER}/qwen-35-4b/transformers/default/1"),
    ]
    for preferred in preferred_model_dirs:
        if preferred.exists():
            return str(preferred)

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
                        return str(check_dir)
    return model_id

TEACHER_PATH = resolve_model_path(BASE_MODEL_ID)
STUDENT_PATH = resolve_model_path(BASE_MODEL_ID)
print(f"Teacher model: {TEACHER_PATH}")
print(f"Student model: {STUDENT_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Configure Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
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
        "teacher_quantization": "4bit",
        "student_quantization": "4bit",
        "use_flash_attention": HAS_FLASH,
    },
    "data": {
        "vinumqa_train": "dataset/viNumericalQA_private/train.json",
        "vinumqa_valid": "dataset/viNumericalQA_private/valid.json",
        "vinumqa_test": "dataset/viNumericalQA_private/test.json",
        "vinumqa_private_test": "dataset/viNumericalQA_private/private_test.json",
        "finqa_dir": "dataset/dataset_finqa_en",
        "include_vinumqa_in_training": False,
        "skip_missing_benchmarks": True,
        "max_samples": 8,
        "train_benchmarks": ["finqa"],
        "eval_benchmarks": ["finqa", "tatqa", "convfinqa", "vinumqa"],
        "use_program_re": False,
    },
    "teacher": {
        "max_workers": 1,
        "batch_size": 1,
        "max_new_tokens": 256,
        "temperature": 0.2,
    },
    "sft": {
        "num_epochs": 1,
        "max_train_samples": 32,
        "max_valid_samples": 8,
        "logging_steps": 1,
        "save_steps": 10,
        "eval_steps": 10,
        "max_seq_length": 1024,
    },
    "grpo": {
        "num_epochs": 1,
        "num_generations": 2,
        "max_completion_length": 512,
    },
    "inference": {
        "num_candidates": 2,
        "max_new_tokens": 512,
    },
}

cfg = load_config(gpu_profile=GPU_PROFILE, overrides=config_overrides)
save_config(cfg, str(WORK_DIR / "data/pipeline/config.yaml"))

print("Configured Qwen3.5-4B stability run:")
print(f"  GPU profile: {GPU_PROFILE}")
print(f"  Teacher:     {cfg.model.teacher_model}")
print(f"  Student:     {cfg.model.student_model}")
print(f"  Eval benchs: {cfg.data.eval_benchmarks}")
print(f"  Candidates:  {cfg.inference.num_candidates}")
print(f"  Max samples: {cfg.data.max_samples}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Inspect Benchmark Availability
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.benchmarks import summarize_benchmark_status

status_rows = summarize_benchmark_status(cfg, phase="eval")
print("Configured eval benchmarks:")
for row in status_rows:
    print(row)


# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Run SFT, GRPO-PCPO, and GRPO-ECRL Suite
# ═══════════════════════════════════════════════════════════════════════
import gc
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.benchmark_suite import run_benchmark_suite
import torch

print("=" * 68)
print("RUNNING BENCHMARK SUITE")
print("=" * 68)

t0 = time.time()
suite_results = run_benchmark_suite(
    cfg,
    include_sft_only=True,
    include_pcpo=True,
    include_ecrl=True,
)
print(f"Suite completed in {(time.time() - t0) / 3600:.2f} hours")

gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Summarize Variant and Benchmark Results
# ═══════════════════════════════════════════════════════════════════════
import json
import os
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

summary_path = WORK_DIR / "data" / "pipeline" / "benchmark_suite" / "suite_results.json"
if not summary_path.exists():
    raise FileNotFoundError(f"Suite summary not found: {summary_path}")

with open(summary_path, "r", encoding="utf-8") as f:
    suite_results = json.load(f)

def fmt_score(value):
    if value is None:
        return "n/a"
    return f"{value:.2%}"

print("\n" + "=" * 90)
print("VARIANT SUMMARY")
print("=" * 90)
for variant, variant_payload in suite_results.items():
    print(f"\n[{variant}]")
    for benchmark, metrics in variant_payload.get("benchmarks", {}).items():
        print(
            f"  {benchmark:<12} "
            f"answer={fmt_score(metrics.get('answer_accuracy')):<8} "
            f"program={fmt_score(metrics.get('program_accuracy')):<8} "
            f"step={fmt_score(metrics.get('step_accuracy')):<8} "
            f"valid={fmt_score(metrics.get('valid_rate')):<8} "
            f"n={metrics.get('total')}"
        )


# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Save Outputs for Kaggle Download
# ═══════════════════════════════════════════════════════════════════════
import json
import os
import shutil
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

output_dir = WORK_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

for src in [
    WORK_DIR / "data" / "pipeline" / "config.yaml",
    WORK_DIR / "data" / "pipeline" / "benchmark_manifest.json",
    WORK_DIR / "data" / "pipeline" / "benchmark_suite" / "suite_results.json",
]:
    if src.exists():
        shutil.copy2(src, output_dir / src.name)

suite_root = WORK_DIR / "data" / "pipeline" / "benchmark_suite"
if suite_root.exists():
    dst = output_dir / "benchmark_suite"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(suite_root, dst)

ckpt_root = WORK_DIR / "checkpoints" / "benchmark_suite"
if ckpt_root.exists():
    dst = output_dir / "checkpoints_benchmark_suite"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ckpt_root, dst)

artifact_summary = {
    "gpu_profile": GPU_PROFILE,
    "teacher_model": TEACHER_PATH,
    "student_model": STUDENT_PATH,
    "suite_summary": str(output_dir / "suite_results.json"),
    "output_dir": str(output_dir),
}

with open(output_dir / "artifact_summary.json", "w", encoding="utf-8") as f:
    json.dump(artifact_summary, f, ensure_ascii=False, indent=2)

print(f"Outputs saved to: {output_dir}")
print(sorted(p.name for p in output_dir.iterdir()))