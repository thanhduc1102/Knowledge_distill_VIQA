"""
Baseline configuration for zero-shot inference across multiple Qwen3.5 models.
Designed for RTX 6000 Pro 96GB (primary) with P100 16GB fallback for testing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaselineModelConfig:
    """Configuration for a single model baseline run."""
    model_id: str
    short_name: str
    quantization: Optional[str] = None  # None, "4bit", "8bit"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    max_seq_length: int = 8192
    max_new_tokens: int = 2048
    num_candidates: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 1  # for generation
    estimated_vram_gb: float = 0.0  # approximate VRAM usage
    trust_remote_code: bool = True


# ── RTX 6000 Pro 96GB Model Configurations ──────────────────────────

RTX6000_MODELS = {
    "qwen3.5-4b": BaselineModelConfig(
        model_id="Qwen/Qwen3.5-4B",
        short_name="qwen3.5-4b",
        quantization=None,
        torch_dtype="bfloat16",
        use_flash_attention=True,
        max_seq_length=16384,
        max_new_tokens=2048,
        num_candidates=15,
        temperature=0.7,
        batch_size=4,
        estimated_vram_gb=9,
    ),
    "qwen3.5-9b": BaselineModelConfig(
        model_id="Qwen/Qwen3.5-9B",
        short_name="qwen3.5-9b",
        quantization=None,
        torch_dtype="bfloat16",
        use_flash_attention=True,
        max_seq_length=16384,
        max_new_tokens=2048,
        num_candidates=15,
        temperature=0.7,
        batch_size=2,
        estimated_vram_gb=19,
    ),
    "qwen3.5-27b": BaselineModelConfig(
        model_id="Qwen/Qwen3.5-27B",
        short_name="qwen3.5-27b",
        quantization=None,
        torch_dtype="bfloat16",
        use_flash_attention=True,
        max_seq_length=12288,
        max_new_tokens=2048,
        num_candidates=10,
        temperature=0.7,
        batch_size=1,
        estimated_vram_gb=55,
    ),
    "qwen3.5-35b-a3b": BaselineModelConfig(
        model_id="Qwen/Qwen3.5-35B-A3B",
        short_name="qwen3.5-35b-a3b",
        quantization=None,
        torch_dtype="bfloat16",
        use_flash_attention=True,
        max_seq_length=16384,
        max_new_tokens=2048,
        num_candidates=15,
        temperature=0.7,
        batch_size=2,
        estimated_vram_gb=72,
    ),
    "qwen3.5-122b-a10b": BaselineModelConfig(
        model_id="Qwen/Qwen3.5-122B-A10B",
        short_name="qwen3.5-122b-a10b",
        quantization="4bit",
        torch_dtype="bfloat16",
        use_flash_attention=True,
        max_seq_length=12288,
        max_new_tokens=2048,
        num_candidates=10,
        temperature=0.7,
        batch_size=1,
        estimated_vram_gb=70,
    ),
}

# ── P100 16GB Test Configurations (small models for pipeline verification) ───

P100_MODELS = {
    "qwen3-0.6b": BaselineModelConfig(
        model_id="Qwen/Qwen3-0.6B",
        short_name="qwen3-0.6b",
        quantization=None,
        torch_dtype="float16",
        use_flash_attention=False,
        max_seq_length=4096,
        max_new_tokens=1024,
        num_candidates=3,
        temperature=0.7,
        batch_size=1,
        estimated_vram_gb=1.5,
    ),
    "qwen3-1.7b": BaselineModelConfig(
        model_id="Qwen/Qwen3-1.7B",
        short_name="qwen3-1.7b",
        quantization=None,
        torch_dtype="float16",
        use_flash_attention=False,
        max_seq_length=4096,
        max_new_tokens=1024,
        num_candidates=3,
        temperature=0.7,
        batch_size=1,
        estimated_vram_gb=3.5,
    ),
    "qwen3-4b": BaselineModelConfig(
        model_id="Qwen/Qwen3-4B",
        short_name="qwen3-4b",
        quantization="4bit",
        torch_dtype="float16",
        use_flash_attention=False,
        max_seq_length=4096,
        max_new_tokens=1024,
        num_candidates=3,
        temperature=0.7,
        batch_size=1,
        estimated_vram_gb=3,
    ),
}


def get_model_configs(gpu_profile: str = "rtx6000_96gb") -> dict[str, BaselineModelConfig]:
    """Get model configurations for a GPU profile."""
    if gpu_profile == "rtx6000_96gb":
        return RTX6000_MODELS
    elif gpu_profile == "p100_16gb":
        return P100_MODELS
    else:
        raise ValueError(f"Unknown GPU profile: {gpu_profile}. Use 'rtx6000_96gb' or 'p100_16gb'.")


def get_single_model_config(model_key: str, gpu_profile: str = "rtx6000_96gb") -> BaselineModelConfig:
    """Get configuration for a specific model."""
    configs = get_model_configs(gpu_profile)
    if model_key not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Model '{model_key}' not available for {gpu_profile}. Available: {available}")
    return configs[model_key]
