"""
VLSP 2025 Baseline - Kaggle Notebook (Offline Execution)
=========================================================
This script is designed to run on Kaggle with RTX 6000 Pro 96GB
WITHOUT internet access.

Setup requirements:
1. Upload the offline package as a Kaggle Dataset (e.g., 'vlsp2025-offline')
2. Add the dataset to your notebook
3. GPU: RTX 6000 Pro 96GB
4. Internet: OFF

The offline package should contain:
  /kaggle/input/vlsp2025-offline/
    ├── wheels/        (pip wheels for offline install)
    ├── models/        (pre-downloaded HF model weights)
    ├── data/receive/  (ViNumQA dataset)
    └── code/          (project source code)
"""

import os
import sys
import subprocess
import json
import time
import gc
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these paths based on your Kaggle dataset name
# ============================================================================

# Path to the offline package dataset on Kaggle
OFFLINE_DATASET = "/kaggle/input/vlsp2025-offline"

# Working directory
WORK_DIR = "/kaggle/working"

# Models to run (comment out models you don't want to test)
MODELS_TO_RUN = [
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b",   # requires 4-bit quantization
]

# Max samples per model (None = all 497 public test samples)
MAX_SAMPLES = None  # Set to e.g. 10 for quick test

# ============================================================================
# STEP 1: Install dependencies offline
# ============================================================================

print("=" * 60)
print(" Step 1: Installing offline dependencies")
print("=" * 60)

wheels_dir = os.path.join(OFFLINE_DATASET, "wheels")
if os.path.exists(wheels_dir) and os.listdir(wheels_dir):
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", wheels_dir,
        "transformers", "accelerate", "peft", "bitsandbytes",
        "datasets", "pandas", "tqdm", "pyarrow", "pyyaml",
        "sentencepiece", "protobuf", "safetensors",
        "huggingface_hub", "tokenizers",
    ], check=False, capture_output=True)
    print("  Offline packages installed.")
else:
    print("  No wheels directory found. Using pre-installed packages.")
    print("  If packages are missing, the script will fail at import time.")

# Try installing flash-attn (may fail on some environments)
try:
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", wheels_dir,
        "flash-attn",
    ], check=True, capture_output=True, timeout=120)
    print("  flash-attn installed.")
except Exception:
    print("  flash-attn not available (will use default attention).")


# ============================================================================
# STEP 2: Setup project code
# ============================================================================

print("\n" + "=" * 60)
print(" Step 2: Setting up project code")
print("=" * 60)

code_dir = os.path.join(OFFLINE_DATASET, "code")
if os.path.exists(code_dir):
    # Copy code to working dir for write access
    os.makedirs(WORK_DIR, exist_ok=True)
    os.system(f"cp -r {code_dir}/* {WORK_DIR}/")
    print(f"  Code copied to {WORK_DIR}")
else:
    print(f"  WARNING: Code directory not found at {code_dir}")
    print("  Attempting to use current directory...")

sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)

# Setup data directory
data_src = os.path.join(OFFLINE_DATASET, "data", "receive")
data_dst = os.path.join(WORK_DIR, "data", "receive")
os.makedirs(data_dst, exist_ok=True)
if os.path.exists(data_src):
    os.system(f"cp -n {data_src}/*.json {data_dst}/")
    print(f"  Data copied to {data_dst}")

# Verify
for f in ["train.json", "valid.json", "test.json"]:
    p = os.path.join(data_dst, f)
    if os.path.exists(p):
        with open(p) as fh:
            print(f"    {f}: {len(json.load(fh))} samples")

# ============================================================================
# STEP 3: Verify environment
# ============================================================================

print("\n" + "=" * 60)
print(" Step 3: Environment verification")
print("=" * 60)

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM: {vram:.1f} GB")

import transformers
print(f"  Transformers: {transformers.__version__}")

try:
    import peft
    print(f"  PEFT: {peft.__version__}")
except ImportError:
    print("  PEFT: NOT INSTALLED")

try:
    import bitsandbytes
    print(f"  BitsAndBytes: {bitsandbytes.__version__}")
except ImportError:
    print("  BitsAndBytes: NOT INSTALLED (4-bit quant unavailable)")

# Check flash attention
HAS_FLASH_ATTN = False
try:
    import flash_attn
    HAS_FLASH_ATTN = True
    print(f"  Flash Attention: {flash_attn.__version__}")
except ImportError:
    print("  Flash Attention: NOT INSTALLED (using eager attention)")


# ============================================================================
# STEP 4: Model path resolution
# ============================================================================

print("\n" + "=" * 60)
print(" Step 4: Resolving model paths")
print("=" * 60)

models_dir = os.path.join(OFFLINE_DATASET, "models")

# Map from model_id to local path
MODEL_LOCAL_PATHS = {}
MODEL_CONFIGS = {
    "qwen3.5-4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "quantization": None,
        "torch_dtype": "bfloat16",
        "max_seq_length": 16384,
        "max_new_tokens": 2048,
        "num_candidates": 15,
        "estimated_vram_gb": 9,
    },
    "qwen3.5-9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "quantization": None,
        "torch_dtype": "bfloat16",
        "max_seq_length": 16384,
        "max_new_tokens": 2048,
        "num_candidates": 15,
        "estimated_vram_gb": 19,
    },
    "qwen3.5-27b": {
        "model_id": "Qwen/Qwen3.5-27B",
        "quantization": None,
        "torch_dtype": "bfloat16",
        "max_seq_length": 12288,
        "max_new_tokens": 2048,
        "num_candidates": 10,
        "estimated_vram_gb": 55,
    },
    "qwen3.5-35b-a3b": {
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "quantization": None,
        "torch_dtype": "bfloat16",
        "max_seq_length": 16384,
        "max_new_tokens": 2048,
        "num_candidates": 15,
        "estimated_vram_gb": 72,
    },
    "qwen3.5-122b-a10b": {
        "model_id": "Qwen/Qwen3.5-122B-A10B",
        "quantization": "4bit",
        "torch_dtype": "bfloat16",
        "max_seq_length": 12288,
        "max_new_tokens": 2048,
        "num_candidates": 10,
        "estimated_vram_gb": 70,
    },
}

for key, cfg in MODEL_CONFIGS.items():
    model_id = cfg["model_id"]
    # Check offline package first
    local_name = model_id.replace("/", "_")
    local_path = os.path.join(models_dir, local_name)

    if os.path.exists(local_path) and os.listdir(local_path):
        MODEL_LOCAL_PATHS[key] = local_path
        size_gb = sum(f.stat().st_size for f in Path(local_path).rglob('*') if f.is_file()) / 1e9
        print(f"  {key}: {local_path} ({size_gb:.1f} GB)")
    else:
        # Try Kaggle HF cache
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"  {key}: Will use HF model_id '{model_id}' (not found locally)")
        MODEL_LOCAL_PATHS[key] = model_id

# ============================================================================
# STEP 5: Run Baselines
# ============================================================================

print("\n" + "=" * 60)
print(" Step 5: Running baselines")
print("=" * 60)

import re
from collections import Counter
from tqdm import tqdm

# Import project modules
from pipeline.program_executor import execute_program, format_answer, validate_program
from pipeline.data_prep import table_to_markdown
from src.assets.template import prompt as prompt_template


def build_prompt(sample):
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


def extract_program_and_answer(text):
    prog_matches = list(re.finditer(
        r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    prog = prog_matches[-1].group(1).strip() if prog_matches else None
    ans_matches = list(re.finditer(
        r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text
    ))
    ans = ans_matches[-1].group(1).strip() if ans_matches else None
    return prog, ans


def majority_vote(candidates, table=None):
    answers, programs = [], []
    for text in candidates:
        prog, ans = extract_program_and_answer(text)
        if prog and validate_program(prog):
            result = execute_program(prog, table)
            if result is not None:
                answers.append(format_answer(result))
                programs.append(prog)
            elif ans:
                answers.append(ans.strip())
                programs.append(prog)
        elif ans:
            answers.append(ans.strip())
            programs.append(prog or "")

    if not answers:
        return {"program": None, "answer": None, "confidence": 0.0, "num_valid": 0}

    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]
    best_program = next((p for a, p in zip(answers, programs) if a == best_answer), None)
    return {
        "program": best_program, "answer": best_answer,
        "confidence": best_count / len(candidates), "num_valid": len(answers),
    }


def run_model_baseline(model_key, test_data, max_samples=None):
    """Run baseline for a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_key]
    model_path = MODEL_LOCAL_PATHS[model_key]

    print(f"\n{'#' * 60}")
    print(f"# Model: {cfg['model_id']}")
    print(f"# Path: {model_path}")
    print(f"# Quantization: {cfg['quantization'] or 'none'}")
    print(f"# Candidates: {cfg['num_candidates']}")
    print(f"{'#' * 60}\n")

    if max_samples:
        test_data = test_data[:max_samples]

    # Model loading
    dtype = torch.bfloat16 if cfg["torch_dtype"] == "bfloat16" else torch.float16
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
    }

    if cfg["quantization"] == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )

    if HAS_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM used: {vram:.1f} GB")

    # Inference
    predictions = []
    start = time.time()
    n_cand = cfg["num_candidates"]

    for i, sample in enumerate(tqdm(test_data, desc=model_key)):
        prompt = build_prompt(sample)
        table = sample.get("table")

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=cfg["max_seq_length"]
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        candidates = []
        for _ in range(n_cand):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=0.7, top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            candidates.append(text)

        vote = majority_vote(candidates, table)

        predictions.append({
            "id": sample["id"],
            "predicted_program": vote["program"],
            "predicted_answer": vote["answer"],
            "confidence": vote["confidence"],
            "num_valid": vote["num_valid"],
            "gold_program": sample["qa"].get("program", ""),
            "gold_answer": str(sample["qa"].get("exe_ans", "")),
        })

    elapsed = time.time() - start
    print(f"  Done: {elapsed:.0f}s ({elapsed/len(test_data):.1f}s/sample)")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return predictions, elapsed


# Load test data
test_path = os.path.join(WORK_DIR, "data", "receive", "test.json")
with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f"\nPublic test: {len(test_data)} samples")

# Run each model
all_results = []
output_dir = os.path.join(WORK_DIR, "outputs", "baseline")
os.makedirs(output_dir, exist_ok=True)

for model_key in MODELS_TO_RUN:
    if model_key not in MODEL_CONFIGS:
        print(f"  SKIP unknown model: {model_key}")
        continue

    try:
        predictions, elapsed = run_model_baseline(model_key, test_data, MAX_SAMPLES)

        # Save predictions
        pred_path = os.path.join(output_dir, f"predictions_{model_key}.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        # Evaluate
        total = len(predictions)
        ea_ok = sum(1 for p in predictions if _ea_match(p))
        pa_ok = sum(1 for p in predictions if _pa_match(p))
        valid = sum(1 for p in predictions if p["predicted_program"] and
                    validate_program(p["predicted_program"]))

        result = {
            "model": model_key,
            "total": total,
            "ea": ea_ok / total if total else 0,
            "pa": pa_ok / total if total else 0,
            "valid_rate": valid / total if total else 0,
            "elapsed": elapsed,
        }
        all_results.append(result)

        print(f"  EA: {result['ea']:.2%} | PA: {result['pa']:.2%} | Valid: {result['valid_rate']:.2%}")

    except Exception as e:
        print(f"  ERROR: {model_key} - {e}")
        import traceback
        traceback.print_exc()
        all_results.append({"model": model_key, "error": str(e)})


def _ea_match(pred):
    try:
        p = float(pred.get("predicted_answer", ""))
        g = float(pred.get("gold_answer", ""))
        return abs(p - g) / max(abs(g), 1e-10) < 1e-4 if g != 0 else abs(p - g) < 1e-5
    except (ValueError, TypeError):
        return (pred.get("predicted_answer") or "").strip() == (pred.get("gold_answer") or "").strip()


def _pa_match(pred):
    from pipeline.evaluate import programs_match
    return programs_match(pred.get("predicted_program", ""), pred.get("gold_program", ""))


# ============================================================================
# STEP 6: Summary
# ============================================================================

print("\n" + "=" * 80)
print(" BASELINE RESULTS SUMMARY")
print("=" * 80)
print(f" {'Model':<25} {'EA':>8} {'PA':>8} {'Valid%':>8} {'Time':>10}")
print(f" {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for r in all_results:
    if "error" in r:
        print(f" {r['model']:<25} {'ERROR':>8}")
    else:
        print(f" {r['model']:<25} {r['ea']:>7.2%} {r['pa']:>7.2%} "
              f"{r['valid_rate']:>7.2%} {r['elapsed']:>8.0f}s")
print("=" * 80)

# Save summary
summary_path = os.path.join(output_dir, "baseline_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nSummary saved: {summary_path}")

# Save outputs to /kaggle/working for download
print("\nAll outputs saved to /kaggle/working/outputs/baseline/")
print("Download from Kaggle Output tab after notebook completes.")
