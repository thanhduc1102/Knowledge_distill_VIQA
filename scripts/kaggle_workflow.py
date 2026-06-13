#!/usr/bin/env python3
"""Generate and push staged Kaggle workflow notebooks.

The workflow is stability-first:
- one notebook per experiment idea
- explicit stage cells with persisted manifests
- optional multi-account fan-out for parallel Kaggle execution
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from kaggle_upload import (
    discover_kaggle_profiles,
    download_wheels,
    load_credentials,
    prepare_code_package,
    prepare_reasoning_benchmark_bundle,
    upload_dataset_kagglehub,
)


@dataclass(frozen=True)
class NotebookSpec:
    name: str
    slug: str
    title: str
    goal: str
    time_estimate: str
    include_teacher: bool
    include_sft: bool
    include_grpo: bool
    reward_type: str | None
    max_samples: int
    train_benchmarks: tuple[str, ...]
    eval_benchmarks: tuple[str, ...]
    teacher_max_new_tokens: int = 256
    sft_max_train_samples: int = 32
    sft_max_valid_samples: int = 8
    inference_candidates: int = 2
    inference_max_new_tokens: int = 384
    grpo_num_generations: int = 2
    grpo_max_completion_length: int = 384


WORKFLOW_SPECS = [
    NotebookSpec(
        name="bootstrap_smoke",
        slug="vlsp-2025-00-bootstrap-smoke",
        title="VLSP 2025 00 Bootstrap Smoke",
        goal="Verify datasets, model download, teacher loading, and one benchmark evaluation on Qwen3.5-4B.",
        time_estimate="~45-90 min",
        include_teacher=True,
        include_sft=False,
        include_grpo=False,
        reward_type=None,
        max_samples=2,
        train_benchmarks=("finqa",),
        eval_benchmarks=("finqa", "vinumqa"),
        teacher_max_new_tokens=128,
        inference_candidates=1,
        inference_max_new_tokens=256,
    ),
    NotebookSpec(
        name="sft_only",
        slug="vlsp-2025-10-sft-only",
        title="VLSP 2025 10 SFT Only",
        goal="Run data prep, teacher distillation, SFT, and benchmark evaluation without RL.",
        time_estimate="~2-4 h",
        include_teacher=True,
        include_sft=True,
        include_grpo=False,
        reward_type=None,
        max_samples=8,
        train_benchmarks=("finqa",),
        eval_benchmarks=("finqa", "tatqa", "convfinqa", "vinumqa"),
    ),
    NotebookSpec(
        name="grpo_pcpo",
        slug="vlsp-2025-20-grpo-pcpo",
        title="VLSP 2025 20 GRPO PCPO",
        goal="Run data prep, distillation, SFT, GRPO with PCPO reward, then evaluate across the benchmark suite.",
        time_estimate="~3-5 h",
        include_teacher=True,
        include_sft=True,
        include_grpo=True,
        reward_type="pcpo",
        max_samples=8,
        train_benchmarks=("finqa",),
        eval_benchmarks=("finqa", "tatqa", "convfinqa", "vinumqa"),
    ),
    NotebookSpec(
        name="grpo_ecrl",
        slug="vlsp-2025-30-grpo-ecrl",
        title="VLSP 2025 30 GRPO ECRL",
        goal="Run data prep, distillation, SFT, GRPO with ECRL reward, then evaluate across the benchmark suite.",
        time_estimate="~3-5 h",
        include_teacher=True,
        include_sft=True,
        include_grpo=True,
        reward_type="ecrl",
        max_samples=8,
        train_benchmarks=("finqa",),
        eval_benchmarks=("finqa", "tatqa", "convfinqa", "vinumqa"),
    ),
]

PREFERRED_PROFILE_ORDER = [
    "thanhduc1108",
    "thanhduc1180",
    "thanhduc1102",
    "ttkhang202",
]

SHARED_MODEL_OWNER = "thanhduc1108"
SHARED_MODEL_INSTANCE = f"{SHARED_MODEL_OWNER}/qwen_35_4b/transformers/default/1"


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _code_cell(source: str) -> dict:
    source = source.lstrip("\n")
    lines = source.split("\n")
    payload = [line + "\n" for line in lines[:-1]]
    if lines and lines[-1]:
        payload.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": _cell_id(),
            "language": "python",
        },
        "outputs": [],
        "source": payload,
    }


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {
            "id": _cell_id(),
            "language": "markdown",
        },
        "source": [source],
    }


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _render(template: str, replacements: dict[str, str]) -> str:
    rendered = textwrap.dedent(template).strip("\n")
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _intro_markdown(spec: NotebookSpec, dataset_owner: str) -> str:
    return textwrap.dedent(
        f"""
        # {spec.title}

        **Goal**: {spec.goal}

        **Expected runtime**: {spec.time_estimate}

        **Dataset owner**: `{dataset_owner}`

        **Stages**
        1. Bootstrap dependencies and working directory
        2. Verify benchmark availability and write a run manifest
        3. Run data preparation
        4. Run optional teacher / SFT / GRPO stages for this idea
        5. Evaluate the selected model on the configured benchmark slice

        **Primary outputs**
        - /kaggle/working/vlsp2025/outputs/{spec.slug}/run_manifest.json
        - /kaggle/working/vlsp2025/outputs/{spec.slug}/run_summary.json
        - /kaggle/working/vlsp2025/data/pipeline/{spec.name}/benchmark_manifest.json
        """
    ).strip()


def _spec_payload(spec: NotebookSpec) -> dict:
    payload = asdict(spec)
    payload["train_benchmarks"] = list(spec.train_benchmarks)
    payload["eval_benchmarks"] = list(spec.eval_benchmarks)
    return payload


def _runtime_config_cell(spec: NotebookSpec, dataset_owner: str) -> str:
    run_spec_literal = repr(_spec_payload(spec))
    return _render(
        """
        RUN_SPEC = __RUN_SPEC__
        DATASET_OWNER = "__DATASET_OWNER__"

        print("=" * 72)
        print(RUN_SPEC["title"])
        print("=" * 72)
        print(f"slug              : {RUN_SPEC['slug']}")
        print(f"goal              : {RUN_SPEC['goal']}")
        print(f"dataset owner     : {DATASET_OWNER}")
        print(f"max samples       : {RUN_SPEC['max_samples']}")
        print(f"train benchmarks  : {RUN_SPEC['train_benchmarks']}")
        print(f"eval benchmarks   : {RUN_SPEC['eval_benchmarks']}")
        print(f"teacher           : {RUN_SPEC['include_teacher']}")
        print(f"sft               : {RUN_SPEC['include_sft']}")
        print(f"grpo              : {RUN_SPEC['include_grpo']}")
        print(f"reward            : {RUN_SPEC['reward_type']}")
        print("=" * 72)
        """,
        {
            "__RUN_SPEC__": run_spec_literal,
            "__DATASET_OWNER__": dataset_owner,
        },
    )


CELL_INSTALL = """
import os
import subprocess
import sys
from pathlib import Path

for offline_var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    if offline_var in os.environ:
        os.environ.pop(offline_var, None)
        print(f"Removed {offline_var} from environment")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _resolve_wheels_dir() -> Path | None:
    candidates = [
        Path("/kaggle/input/vlsp2025-kd-wheels"),
        Path(f"/kaggle/input/datasets/{DATASET_OWNER}/vlsp2025-kd-wheels"),
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


WHEELS_DIR = _resolve_wheels_dir()


def _install_packages(packages, extra_args=None):
    command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"]
    if extra_args:
        command.extend(extra_args)
    command.extend(packages)
    return subprocess.run(command, capture_output=True, text=True)


packages = [
    "kagglehub",
    "transformers>=5.0",
    "peft>=0.18",
    "accelerate>=1.0",
    "datasets>=4.0",
    "bitsandbytes>=0.49",
    "trl>=1.0",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "sympy",
    "pyyaml",
    "pyarrow",
    "tqdm",
    "pandas",
    "huggingface_hub>=1.0",
    "tokenizers",
]

if WHEELS_DIR is not None:
    print(f"Installing offline wheels from {WHEELS_DIR}...")
    result = _install_packages(packages, ["--no-index", "--find-links", str(WHEELS_DIR)])
    if result.returncode != 0:
        print("Offline wheel install failed. Falling back to pip.")
        fallback = _install_packages(packages)
        if fallback.returncode != 0:
            raise RuntimeError(fallback.stderr[-800:])
else:
    print("Offline wheels not attached. Installing from pip.")
    result = _install_packages(packages)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])

try:
    import flash_attn  # noqa: F401
    print("Flash Attention already available.")
except Exception:
    print("Flash Attention not preinstalled. Using sdpa fallback.")
"""


CELL_BOOTSTRAP = _render(
    """
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


    CODE_SRC = resolve_dataset_dir("/kaggle/input/vlsp2025-kd-pipeline", "__DATASET_OWNER__/vlsp2025-kd-pipeline")
    BUNDLE_SRC = resolve_dataset_dir(
        "/kaggle/input/financial-reasoning-benchmarks",
        "__DATASET_OWNER__/financial-reasoning-benchmarks",
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
            if dst.is_symlink() or dst.exists():
                dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
            dst.symlink_to(src)
            print(f"Linked {dst_name} -> {src}")
        elif required:
            raise FileNotFoundError(f"Required bundled dataset component missing: {src}")
        else:
            print(f"Optional component missing: {src_name}")

    manifest_path = BUNDLE_SRC / "bundle_manifest.json"
    if manifest_path.exists():
        print(f"Bundle manifest: {manifest_path.read_text(encoding='utf-8')[:800]}")

    os.chdir(WORK_DIR)
    print(f"Working directory ready: {WORK_DIR}")
    """,
    {"__DATASET_OWNER__": "__DATASET_OWNER__"},
)


CELL_GPU_MODEL = _render(
    """
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
else:
    GPU_PROFILE = "p100_16gb"

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"

PREFERRED_MODEL_DIRS = [
    Path("/kaggle/input/models/__MODEL_OWNER__/qwen_35_4b/transformers/default/1"),
    Path("/kaggle/input/models/__MODEL_OWNER__/qwen-35-4b/transformers/default/1"),
    Path("/kaggle/input/models/__MODEL_OWNER__/qwen_35_27b_distill_opus46/transformers/default/1"),
]


def resolve_model_path(model_id: str) -> str:
    base_name = model_id.split("/")[-1]
    slugs = set()
    for name in [base_name, base_name.lower()]:
        slugs.add(name)
        slugs.add(name.replace(".", "-").replace("_", "-"))
        slugs.add(name.replace(".", "_").replace("-", "_"))

    normalized = {slug.lower() for slug in slugs}
    for preferred in PREFERRED_MODEL_DIRS:
        if preferred.exists():
            return str(preferred)
    for root in [Path("/kaggle/input"), Path("/kaggle/models")]:
        if not root.exists():
            continue
        for entry in root.iterdir():
            if entry.name.lower() not in normalized:
                continue
            for candidate in [entry, *entry.rglob("config.json")]:
                check_dir = candidate if candidate.is_dir() else candidate.parent
                has_weights = (
                    any(check_dir.glob("*.safetensors"))
                    or any(check_dir.glob("*.bin"))
                    or (check_dir / "config.json").exists()
                )
                if has_weights:
                    return str(check_dir)
    return model_id


TEACHER_PATH = resolve_model_path(BASE_MODEL_ID)
STUDENT_PATH = resolve_model_path(BASE_MODEL_ID)
print(f"GPU profile: {GPU_PROFILE}")
print(f"Flash attention: {HAS_FLASH}")
print(f"Teacher model: {TEACHER_PATH}")
print(f"Student model: {STUDENT_PATH}")
""",
    {"__MODEL_OWNER__": SHARED_MODEL_OWNER},
)


CELL_CONFIG_AND_STATE = """
import json
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.config import load_config, save_config

RUN_ROOT = WORK_DIR / "outputs" / RUN_SPEC["slug"]
RUN_ROOT.mkdir(parents=True, exist_ok=True)
RUN_MANIFEST_PATH = RUN_ROOT / "run_manifest.json"

RUN_STATE = {
    "run_spec": RUN_SPEC,
    "dataset_owner": DATASET_OWNER,
    "input_handles": {
        "code": f"{DATASET_OWNER}/vlsp2025-kd-pipeline",
        "wheels": f"{DATASET_OWNER}/vlsp2025-kd-wheels",
        "bundle": f"{DATASET_OWNER}/financial-reasoning-benchmarks",
    },
    "paths": {
        "work_dir": str(WORK_DIR),
        "run_root": str(RUN_ROOT),
    },
    "stages": {},
}


def record_stage(stage_name: str, payload: dict):
    stage_payload = dict(payload)
    stage_payload["updated_at"] = time.time()
    RUN_STATE["stages"][stage_name] = stage_payload
    with open(RUN_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(RUN_STATE, f, ensure_ascii=False, indent=2)


data_output_dir = f"data/pipeline/{RUN_SPEC['name']}"
sft_output_dir = f"checkpoints/{RUN_SPEC['name']}/sft"
grpo_output_dir = f"checkpoints/{RUN_SPEC['name']}/grpo"

config_overrides = {
    "model": {
        "teacher_model": TEACHER_PATH,
        "student_model": STUDENT_PATH,
        "teacher_quantization": "4bit",
        "student_quantization": "4bit",
        "use_flash_attention": HAS_FLASH,
    },
    "data": {
        "output_dir": data_output_dir,
        "vinumqa_train": "dataset/viNumericalQA_private/train.json",
        "vinumqa_valid": "dataset/viNumericalQA_private/valid.json",
        "vinumqa_test": "dataset/viNumericalQA_private/test.json",
        "vinumqa_private_test": "dataset/viNumericalQA_private/private_test.json",
        "finqa_dir": "dataset/dataset_finqa_en",
        "include_vinumqa_in_training": False,
        "skip_missing_benchmarks": True,
        "max_samples": RUN_SPEC["max_samples"],
        "train_benchmarks": RUN_SPEC["train_benchmarks"],
        "eval_benchmarks": RUN_SPEC["eval_benchmarks"],
        "use_program_re": False,
    },
    "teacher": {
        "max_workers": 1,
        "batch_size": 1,
        "max_new_tokens": RUN_SPEC["teacher_max_new_tokens"],
        "temperature": 0.2,
    },
    "sft": {
        "output_dir": sft_output_dir,
        "num_epochs": 1,
        "max_train_samples": RUN_SPEC["sft_max_train_samples"],
        "max_valid_samples": RUN_SPEC["sft_max_valid_samples"],
        "logging_steps": 1,
        "save_steps": 10,
        "eval_steps": 10,
        "max_seq_length": 1024,
    },
    "grpo": {
        "output_dir": grpo_output_dir,
        "num_epochs": 1,
        "num_generations": RUN_SPEC["grpo_num_generations"],
        "max_completion_length": RUN_SPEC["grpo_max_completion_length"],
    },
    "inference": {
        "num_candidates": RUN_SPEC["inference_candidates"],
        "max_new_tokens": RUN_SPEC["inference_max_new_tokens"],
    },
}

cfg = load_config(gpu_profile=GPU_PROFILE, overrides=config_overrides)
save_config(cfg, str(RUN_ROOT / "config.yaml"))
RUN_STATE["paths"]["config_yaml"] = str(RUN_ROOT / "config.yaml")
record_stage(
    "config",
    {
        "gpu_profile": GPU_PROFILE,
        "teacher_model": cfg.model.teacher_model,
        "student_model": cfg.model.student_model,
        "data_output_dir": data_output_dir,
        "sft_output_dir": sft_output_dir,
        "grpo_output_dir": grpo_output_dir,
    },
)

data_paths = {}
distilled_path = None
sft_model_path = None
grpo_model_path = None
suite_summary = None
"""


CELL_BENCHMARK_STATUS = """
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

record_stage("benchmark_status", {"rows": status_rows})
"""


CELL_DATA_PREP = """
import json
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.data_prep import run_data_prep

print("=" * 72)
print("RUNNING DATA PREP")
print("=" * 72)

t0 = time.time()
data_paths = run_data_prep(cfg)
elapsed_minutes = (time.time() - t0) / 60.0
print(f"Data prep completed in {elapsed_minutes:.2f} minutes")
print(json.dumps(data_paths, ensure_ascii=False, indent=2))

RUN_STATE["paths"]["data_paths"] = data_paths
record_stage(
    "data_prep",
    {
        "elapsed_minutes": elapsed_minutes,
        "paths": data_paths,
    },
)
"""


CELL_TEACHER = """
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

if RUN_SPEC["include_teacher"]:
    from pipeline.teacher_distill import run_teacher_distillation

    print("=" * 72)
    print("RUNNING TEACHER DISTILLATION")
    print("=" * 72)
    t0 = time.time()
    distilled_path = run_teacher_distillation(cfg, data_paths["teacher_input"])
    elapsed_minutes = (time.time() - t0) / 60.0
    RUN_STATE["paths"]["distilled_path"] = distilled_path
    record_stage(
        "teacher",
        {
            "enabled": True,
            "teacher_input": data_paths["teacher_input"],
            "distilled_path": distilled_path,
            "elapsed_minutes": elapsed_minutes,
        },
    )
    print(f"Teacher distillation output: {distilled_path}")
else:
    distilled_path = None
    record_stage("teacher", {"enabled": False, "reason": "stage disabled"})
    print("Teacher stage disabled for this notebook.")
"""


CELL_SFT = """
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

if RUN_SPEC["include_sft"]:
    from pipeline.train_sft import run_sft_training

    print("=" * 72)
    print("RUNNING SFT")
    print("=" * 72)
    t0 = time.time()
    train_path = distilled_path or data_paths["sft_train"]
    sft_model_path = run_sft_training(
        cfg,
        train_path=train_path,
        valid_path=data_paths["sft_valid"],
    )
    elapsed_minutes = (time.time() - t0) / 60.0
    RUN_STATE["paths"]["sft_model_path"] = sft_model_path
    record_stage(
        "sft",
        {
            "enabled": True,
            "train_path": train_path,
            "valid_path": data_paths["sft_valid"],
            "sft_model_path": sft_model_path,
            "elapsed_minutes": elapsed_minutes,
        },
    )
    print(f"SFT model path: {sft_model_path}")
else:
    sft_model_path = None
    record_stage("sft", {"enabled": False, "reason": "stage disabled"})
    print("SFT stage disabled for this notebook.")
"""


CELL_GRPO = """
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

if RUN_SPEC["include_grpo"]:
    from pipeline.train_grpo import run_grpo_training

    print("=" * 72)
    print("RUNNING GRPO")
    print("=" * 72)
    cfg.grpo.reward_type = RUN_SPEC["reward_type"]
    t0 = time.time()
    grpo_model_path = run_grpo_training(cfg, sft_model_path)
    elapsed_minutes = (time.time() - t0) / 60.0
    RUN_STATE["paths"]["grpo_model_path"] = grpo_model_path
    record_stage(
        "grpo",
        {
            "enabled": True,
            "reward_type": RUN_SPEC["reward_type"],
            "input_model_path": sft_model_path,
            "grpo_model_path": grpo_model_path,
            "elapsed_minutes": elapsed_minutes,
        },
    )
    print(f"GRPO model path: {grpo_model_path}")
else:
    grpo_model_path = None
    record_stage("grpo", {"enabled": False, "reason": "stage disabled"})
    print("GRPO stage disabled for this notebook.")
"""


CELL_EVALUATE = """
import json
import os
import sys
import time
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from pipeline.benchmark_suite import evaluate_model_on_suite

print("=" * 72)
print("RUNNING EVALUATION")
print("=" * 72)

with open(data_paths["benchmark_manifest"], "r", encoding="utf-8") as f:
    manifest = json.load(f)

if grpo_model_path:
    eval_model_path = grpo_model_path
elif sft_model_path:
    eval_model_path = sft_model_path
else:
    eval_model_path = cfg.model.student_model

t0 = time.time()
suite_summary = evaluate_model_on_suite(cfg, eval_model_path, manifest, RUN_SPEC["name"])
elapsed_minutes = (time.time() - t0) / 60.0

summary_path = (
    Path(cfg.project_root)
    / cfg.data.output_dir
    / "benchmark_suite"
    / RUN_SPEC["name"]
    / "summary.json"
)
RUN_STATE["paths"]["evaluation_summary"] = str(summary_path)
record_stage(
    "evaluation",
    {
        "model_path": eval_model_path,
        "summary_path": str(summary_path),
        "elapsed_minutes": elapsed_minutes,
        "benchmarks": suite_summary.get("benchmarks", {}),
    },
)

print(json.dumps(suite_summary, ensure_ascii=False, indent=2))
"""


CELL_FINALIZE = """
import json

summary = {
    "run_spec": RUN_SPEC,
    "dataset_owner": DATASET_OWNER,
    "paths": RUN_STATE.get("paths", {}),
    "stages": RUN_STATE.get("stages", {}),
    "benchmarks": (suite_summary or {}).get("benchmarks", {}),
}
summary_path = RUN_ROOT / "run_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

RUN_STATE["paths"]["run_summary"] = str(summary_path)
record_stage("finalize", {"summary_path": str(summary_path)})

print("=" * 72)
print("FINAL SUMMARY")
print("=" * 72)
print(f"run manifest : {RUN_MANIFEST_PATH}")
print(f"run summary  : {summary_path}")
for benchmark, metrics in (suite_summary or {}).get("benchmarks", {}).items():
    answer_accuracy = metrics.get("answer_accuracy")
    label = "n/a" if answer_accuracy is None else f"{answer_accuracy:.2%}"
    print(f"{benchmark:<12} answer={label} total={metrics.get('total')}")
"""


def build_notebook(spec: NotebookSpec, dataset_owner: str) -> dict:
    cells = [
        _md_cell(_intro_markdown(spec, dataset_owner)),
        _md_cell("## Cell 0: Run Configuration"),
        _code_cell(_runtime_config_cell(spec, dataset_owner)),
        _md_cell("## Cell 1: Install Dependencies"),
        _code_cell(CELL_INSTALL),
        _md_cell("## Cell 2: Bootstrap Workspace and Datasets"),
        _code_cell(CELL_BOOTSTRAP.replace("__DATASET_OWNER__", dataset_owner)),
        _md_cell("## Cell 3: Detect GPU and Resolve Models"),
        _code_cell(CELL_GPU_MODEL),
        _md_cell("## Cell 4: Materialize Config and Manifest State"),
        _code_cell(CELL_CONFIG_AND_STATE),
        _md_cell("## Cell 5: Inspect Benchmark Availability"),
        _code_cell(CELL_BENCHMARK_STATUS),
        _md_cell("## Cell 6: Run Data Prep"),
        _code_cell(CELL_DATA_PREP),
        _md_cell("## Cell 7: Run Teacher Distillation"),
        _code_cell(CELL_TEACHER),
        _md_cell("## Cell 8: Run SFT"),
        _code_cell(CELL_SFT),
        _md_cell("## Cell 9: Run GRPO"),
        _code_cell(CELL_GRPO),
        _md_cell("## Cell 10: Evaluate Benchmarks"),
        _code_cell(CELL_EVALUATE),
        _md_cell("## Cell 11: Persist Run Summary"),
        _code_cell(CELL_FINALIZE),
    ]
    return _notebook(cells)


def write_notebook(spec: NotebookSpec, dataset_owner: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f"{spec.slug}.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(build_notebook(spec, dataset_owner), f, ensure_ascii=False, indent=2)
    return notebook_path


def ordered_profiles(all_profiles: dict[str, dict[str, str]], selected: list[str] | None = None) -> list[str]:
    available = list(all_profiles)
    if selected:
        filtered = [name for name in selected if name in all_profiles]
    else:
        filtered = available

    preferred = [name for name in PREFERRED_PROFILE_ORDER if name in filtered]
    remainder = [name for name in filtered if name not in preferred]
    return preferred + remainder


def build_plan(profile_names: list[str]) -> list[dict]:
    if not profile_names:
        raise ValueError("No Kaggle profiles available for workflow planning")

    plan = []
    for index, spec in enumerate(WORKFLOW_SPECS):
        profile_name = profile_names[index % len(profile_names)]
        plan.append(
            {
                "profile": profile_name,
                "dataset_owner": profile_name,
                "spec": _spec_payload(spec),
            }
        )
    return plan


def save_plan(plan: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "workflow_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return plan_path


def sync_shared_datasets(plan: list[dict], output_base: Path):
    package_root = output_base / "shared_packages"
    code_dir = package_root / "vlsp2025-kd-pipeline"
    wheels_dir = package_root / "vlsp2025-kd-wheels"

    print("Preparing shared upload directories...")
    prepare_code_package(code_dir)
    download_wheels(wheels_dir)
    bundle_dir = prepare_reasoning_benchmark_bundle(package_root)

    seen_profiles: list[str] = []
    for row in plan:
        profile_name = row["profile"]
        if profile_name in seen_profiles:
            continue
        seen_profiles.append(profile_name)

        username = load_credentials(profile_name)
        print(f"\n=== Syncing shared datasets to {username} ===")
        upload_dataset_kagglehub(
            f"{username}/vlsp2025-kd-pipeline",
            str(code_dir),
            "Pipeline code for staged Kaggle workflow",
        )
        upload_dataset_kagglehub(
            f"{username}/vlsp2025-kd-wheels",
            str(wheels_dir),
            "Offline wheels for staged Kaggle workflow",
        )
        upload_dataset_kagglehub(
            f"{username}/financial-reasoning-benchmarks",
            str(bundle_dir),
            "Reasoning benchmark bundle for staged Kaggle workflow",
        )


def push_kernel(username: str, dataset_owner: str, spec: NotebookSpec, notebook_path: Path, output_base: Path) -> dict:
    push_dir = output_base / "kernel_push" / username / spec.slug
    if push_dir.exists():
        shutil.rmtree(push_dir)
    push_dir.mkdir(parents=True, exist_ok=True)

    target_notebook = push_dir / notebook_path.name
    shutil.copy2(notebook_path, target_notebook)

    metadata = {
        "id": f"{username}/{spec.slug}",
        "title": spec.title,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [
            f"{dataset_owner}/vlsp2025-kd-pipeline",
            f"{dataset_owner}/vlsp2025-kd-wheels",
            f"{dataset_owner}/financial-reasoning-benchmarks",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [
            SHARED_MODEL_INSTANCE,
        ],
    }

    with open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(push_dir)],
        capture_output=True,
        text=True,
    )

    payload = {
        "kernel": f"{username}/{spec.slug}",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to push {username}/{spec.slug}: {payload['stdout']} {payload['stderr']}"
        )
    return payload


def push_workflow(plan: list[dict], output_base: Path) -> list[dict]:
    pushes = []
    for row in plan:
        profile_name = row["profile"]
        dataset_owner = row["dataset_owner"]
        spec = next(item for item in WORKFLOW_SPECS if item.name == row["spec"]["name"])

        username = load_credentials(profile_name)
        notebook_dir = output_base / "generated_notebooks" / username
        notebook_path = write_notebook(spec, dataset_owner, notebook_dir)

        print(f"\n=== Pushing {spec.slug} to {username} ===")
        push_result = push_kernel(username, dataset_owner, spec, notebook_path, output_base)
        pushes.append(push_result)
        print(push_result["stdout"] or f"Pushed {push_result['kernel']}")
    return pushes


def status_workflow(plan: list[dict]) -> list[dict]:
    statuses = []
    for row in plan:
        username = load_credentials(row["profile"])
        slug = row["spec"]["slug"]
        kernel = f"{username}/{slug}"
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True,
            text=True,
        )
        statuses.append(
            {
                "kernel": kernel,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
        print(result.stdout.strip() or result.stderr.strip())
    return statuses


def print_profiles(profiles: dict[str, dict[str, str]]):
    print("Discovered Kaggle profiles:")
    for name, payload in profiles.items():
        print(f"  - {name} (username={payload.get('username', name)})")


def print_plan(plan: list[dict]):
    print("Workflow plan:")
    for row in plan:
        print(
            f"  - {row['spec']['slug']:<28} profile={row['profile']:<14} "
            f"reward={row['spec']['reward_type'] or 'none'}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate and push staged Kaggle workflow notebooks")
    parser.add_argument("--list-profiles", action="store_true", help="List Kaggle profiles parsed from .env")
    parser.add_argument("--generate", action="store_true", help="Generate notebooks and workflow plan locally")
    parser.add_argument("--sync-shared", action="store_true", help="Upload shared datasets to every profile in the plan")
    parser.add_argument("--push", action="store_true", help="Push the staged notebooks in the workflow plan")
    parser.add_argument("--status", action="store_true", help="Check Kaggle kernel status for the workflow plan")
    parser.add_argument("--all", action="store_true", help="Generate, sync shared datasets, push notebooks, and check status")
    parser.add_argument("--profiles", default=None, help="Comma-separated subset of Kaggle profiles to use")
    parser.add_argument("--output-dir", default="/tmp/kaggle_workflow", help="Working directory for generated notebooks and metadata")
    args = parser.parse_args()

    has_action = args.list_profiles or args.generate or args.sync_shared or args.push or args.status or args.all
    if not has_action:
        parser.print_help()
        return

    profiles = discover_kaggle_profiles()
    if not profiles:
        raise SystemExit("No Kaggle profiles discovered in .env")

    if args.list_profiles:
        print_profiles(profiles)
        if not (args.generate or args.sync_shared or args.push or args.status or args.all):
            return

    selected_profiles = None
    if args.profiles:
        selected_profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]

    ordered = ordered_profiles(profiles, selected_profiles)
    if not ordered:
        raise SystemExit("No valid Kaggle profiles remain after applying --profiles")

    output_base = Path(args.output_dir)
    plan = build_plan(ordered)
    print_plan(plan)
    plan_path = save_plan(plan, output_base)
    print(f"Plan saved to: {plan_path}")

    if args.generate or args.all:
        for row in plan:
            spec = next(item for item in WORKFLOW_SPECS if item.name == row["spec"]["name"])
            notebook_dir = output_base / "generated_notebooks" / row["dataset_owner"]
            notebook_path = write_notebook(spec, row["dataset_owner"], notebook_dir)
            print(f"Generated notebook: {notebook_path}")

    if args.sync_shared or args.all:
        sync_shared_datasets(plan, output_base)

    if args.push or args.all:
        push_workflow(plan, output_base)

    if args.status or args.all:
        status_workflow(plan)


if __name__ == "__main__":
    main()