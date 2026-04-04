"""
Configuration management for the Knowledge Distillation pipeline.
Supports multiple GPU profiles for easy migration between hardware.
"""

import os
import yaml
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    teacher_model: str = "Qwen/Qwen3.5-4B"
    student_model: str = "Qwen/Qwen3.5-0.8B"
    teacher_quantization: Optional[str] = None  # "4bit", "8bit", None
    student_quantization: Optional[str] = None
    use_flash_attention: bool = False
    torch_dtype: str = "float16"
    max_seq_length: int = 4096
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    vinumqa_train: str = "dataset/viNumericalQA_private/train.json"
    vinumqa_valid: str = "dataset/viNumericalQA_private/valid.json"
    vinumqa_test: str = "dataset/viNumericalQA_private/test.json"
    vinumqa_private_test: str = "dataset/viNumericalQA_private/private_test.json"
    finqa_dir: str = "dataset/finqa_en"
    output_dir: str = "data/pipeline"
    use_finqa: bool = True
    use_program_re: bool = True
    max_samples: Optional[int] = None  # None = use all


@dataclass
class TeacherConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "no-need"
    max_workers: int = 16
    max_retries: int = 0       # 0 = try once, fall back to gold SFT data on fail
    batch_size: int = 8        # Samples per GPU batch for local teacher
    checkpoint_every: int = 100
    max_new_tokens: int = 512  # Hard cap on output tokens; reduces decode time 4x vs 2048
    use_guided_template: bool = True  # Use think.py (guided) for 100% match rate & faster gen
    temperature: float = 0.6
    top_p: float = 0.95
    use_local: bool = True  # True = load model locally; False = use API


@dataclass
class SFTConfig:
    output_dir: str = "checkpoints/sft"
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    bf16: bool = False
    fp16: bool = True


@dataclass
class GRPOConfig:
    output_dir: str = "checkpoints/grpo"
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    num_generations: int = 5
    max_completion_length: int = 2048
    kl_coef: float = 0.001
    reward_weights: dict = field(default_factory=lambda: {
        "alpha": 0.7,   # R_valid base
        "beta": 0.2,    # R_exec weight
        "gamma": 0.1,   # R_bonus weight
    })
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = True


@dataclass
class InferenceConfig:
    num_candidates: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048
    base_url: str = "http://localhost:8000/v1"
    use_local: bool = True
    batch_size: int = 4


@dataclass
class PipelineConfig:
    project_name: str = "vlsp2025-kd"
    seed: int = 42
    project_root: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        if not self.project_root:
            self.project_root = str(Path(__file__).parent.parent)


# ── GPU Profile Presets ──────────────────────────────────────────────

GPU_PROFILES = {
    # P100 16GB - Testing profile (Qwen3.5-4B teacher 4bit, Qwen3.5-0.8B student)
    "p100_16gb": {
        "model": {
            "teacher_model": "Qwen/Qwen3.5-4B",
            "student_model": "Qwen/Qwen3.5-0.8B",
            "teacher_quantization": "4bit",
            "student_quantization": None,
            "use_flash_attention": False,
            "torch_dtype": "float16",
            "max_seq_length": 4096,
        },
        "sft": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "bf16": False,
            "fp16": True,
            "use_lora": True,
            "lora_r": 64,
        },
        "grpo": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_generations": 3,
            "bf16": False,
            "fp16": True,
        },
        "teacher": {
            "max_workers": 4,
            "use_local": True,
            "batch_size": 2,
            "max_retries": 2,
            "checkpoint_every": 50,
        },
        "inference": {
            "num_candidates": 5,
            "batch_size": 2,
            "use_local": True,
        },
    },
    # RTX 6000 Pro 96GB - Production profile (Qwen3.5-27B teacher, Qwen3.5-4B student)
    "rtx6000_96gb": {
        "model": {
            "teacher_model": "Qwen/Qwen3.5-27B",
            "student_model": "Qwen/Qwen3.5-4B",
            "teacher_quantization": None,
            "student_quantization": None,
            "use_flash_attention": True,
            "torch_dtype": "bfloat16",
            "max_seq_length": 8192,
        },
        "sft": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "fp16": False,
            "use_lora": True,
            "lora_r": 128,
            "lora_alpha": 256,
        },
        "grpo": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_generations": 5,
            "bf16": True,
            "fp16": False,
        },
        "teacher": {
            "max_workers": 8,
            "use_local": True,
            "batch_size": 12,         # 95GB: 27B bf16 ~54GB → 12 samples safe
            "max_retries": 0,
            "max_new_tokens": 512,    # ~3x faster than 2048; guided output fits in 512
            "use_guided_template": True,
            "checkpoint_every": 100,
        },
        "inference": {
            "num_candidates": 15,
            "batch_size": 8,
            "use_local": True,
        },
    },
    # RTX 6000 Pro 96GB with larger teacher models
    "rtx6000_96gb_35b": {
        "model": {
            "teacher_model": "Qwen/Qwen3.5-35B-A3B",
            "student_model": "Qwen/Qwen3.5-4B",
            "teacher_quantization": None,
            "student_quantization": None,
            "use_flash_attention": True,
            "torch_dtype": "bfloat16",
            "max_seq_length": 8192,
        },
        "sft": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "fp16": False,
            "use_lora": True,
            "lora_r": 128,
            "lora_alpha": 256,
        },
        "grpo": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_generations": 5,
            "bf16": True,
            "fp16": False,
        },
        "teacher": {
            "max_workers": 8,
            "use_local": True,
            "batch_size": 4,
            "max_retries": 2,
            "checkpoint_every": 50,
        },
        "inference": {
            "num_candidates": 15,
            "batch_size": 8,
            "use_local": True,
        },
    },
    "rtx6000_96gb_122b": {
        "model": {
            "teacher_model": "Qwen/Qwen3.5-122B-A10B",
            "student_model": "Qwen/Qwen3.5-9B",
            "teacher_quantization": "4bit",
            "student_quantization": None,
            "use_flash_attention": True,
            "torch_dtype": "bfloat16",
            "max_seq_length": 16384,
        },
        "sft": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "bf16": True,
            "fp16": False,
            "use_lora": True,
            "lora_r": 128,
            "lora_alpha": 256,
        },
        "grpo": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_generations": 5,
            "bf16": True,
            "fp16": False,
        },
        "teacher": {
            "max_workers": 4,
            "use_local": True,
            "batch_size": 2,
            "max_retries": 1,
            "checkpoint_every": 50,
        },
        "inference": {
            "num_candidates": 10,
            "batch_size": 4,
            "use_local": True,
        },
    },
    # A100 80GB
    "a100_80gb": {
        "model": {
            "teacher_model": "Qwen/Qwen3.5-27B",
            "student_model": "Qwen/Qwen3.5-9B",
            "teacher_quantization": None,
            "student_quantization": None,
            "use_flash_attention": True,
            "torch_dtype": "bfloat16",
            "max_seq_length": 16384,
        },
        "sft": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "fp16": False,
            "lora_r": 128,
            "lora_alpha": 256,
        },
        "grpo": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_generations": 5,
            "bf16": True,
            "fp16": False,
        },
        "teacher": {
            "max_workers": 8,
            "use_local": True,
            "batch_size": 8,
            "max_retries": 2,
            "checkpoint_every": 100,
        },
        "inference": {
            "num_candidates": 15,
            "batch_size": 16,
        },
    },
}


def _deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    gpu_profile: str = "p100_16gb",
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> PipelineConfig:
    """Load pipeline configuration from GPU profile, optional YAML file, and overrides."""
    cfg = PipelineConfig()
    cfg_dict = asdict(cfg)

    # Apply GPU profile
    if gpu_profile in GPU_PROFILES:
        profile = GPU_PROFILES[gpu_profile]
        cfg_dict = _deep_update(cfg_dict, profile)

    # Apply YAML config file
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg_dict = _deep_update(cfg_dict, yaml_cfg)

    # Apply CLI overrides
    if overrides:
        cfg_dict = _deep_update(cfg_dict, overrides)

    # Reconstruct dataclass
    cfg = PipelineConfig(
        project_name=cfg_dict.get("project_name", "vlsp2025-kd"),
        seed=cfg_dict.get("seed", 42),
        project_root=cfg_dict.get("project_root", ""),
        model=ModelConfig(**cfg_dict.get("model", {})),
        data=DataConfig(**cfg_dict.get("data", {})),
        teacher=TeacherConfig(**cfg_dict.get("teacher", {})),
        sft=SFTConfig(**cfg_dict.get("sft", {})),
        grpo=GRPOConfig(**cfg_dict.get("grpo", {})),
        inference=InferenceConfig(**cfg_dict.get("inference", {})),
    )
    return cfg


def save_config(cfg: PipelineConfig, path: str) -> None:
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved: {path}")
