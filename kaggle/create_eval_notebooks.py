#!/usr/bin/env python3
"""
Generate 3 evaluation baseline notebooks for VLSP 2025 KD pipeline.
Run: python kaggle/create_eval_notebooks.py

Outputs:
  kaggle/rtx6000-eval-distill-only.ipynb  — NB1: Teacher distillation + quality eval
  kaggle/rtx6000-eval-sft-only.ipynb      — NB2: SFT on gold data + EA/PA eval
  kaggle/rtx6000-eval-kd.ipynb            — NB3: Distill + SFT + EA/PA eval + comparison
"""
import json, uuid
from pathlib import Path

def _id(): return uuid.uuid4().hex[:8]

def code_cell(src):
    src = src.lstrip('\n')
    lines = src.split('\n')
    source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines and lines[-1] else [])
    return {"cell_type":"code","execution_count":None,"id":_id(),"metadata":{},"outputs":[],"source":source}

def md_cell(src):
    return {"cell_type":"markdown","id":_id(),"metadata":{},"source":[src]}

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }

# ═══════════════════════════════════════════════════════════════════
# SHARED CELL BODIES
# ═══════════════════════════════════════════════════════════════════

CELL_INSTALL = r"""
import subprocess, os, sys, shutil
from pathlib import Path

WORK_DIR     = Path("/kaggle/working/vlsp2025")
WHEELS_DIR   = Path("/kaggle/input/datasets/thanhduc1108/vlsp2025-kd-wheels")
PIPELINE_SRC = Path("/kaggle/input/datasets/thanhduc1108/vlsp2025-kd-pipeline")

os.environ.update({
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
WORK_DIR.mkdir(parents=True, exist_ok=True)

# --- Install wheels ---
if WHEELS_DIR.exists():
    wheels = sorted(WHEELS_DIR.glob("*.whl"))
    if wheels:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
             f"--find-links={WHEELS_DIR}"] + [str(w) for w in wheels],
            capture_output=True, text=True)
        print(f"Wheels: {len(wheels)} installed" if r.returncode == 0 else f"Wheel warn: {r.stderr[:100]}")
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers", "accelerate", "peft", "bitsandbytes", "sympy", "tqdm"],
                   check=False)
    print("Installed from PyPI (online mode)")

# --- Copy pipeline code (pipeline/, src/, configs/) ---
if PIPELINE_SRC.exists():
    for d in ["pipeline", "src", "configs"]:
        src_d = PIPELINE_SRC / d
        dst_d = WORK_DIR / d
        if src_d.exists():
            if dst_d.exists():
                shutil.rmtree(dst_d)
            shutil.copytree(src_d, dst_d)
            n = sum(1 for _ in dst_d.rglob("*") if _.is_file())
            print(f"  Copied {d}/ ({n} files)")
        else:
            print(f"  WARNING: {d}/ not found in pipeline dataset")
    # Verify critical module exists
    if not (WORK_DIR / "pipeline" / "__init__.py").exists():
        raise RuntimeError("pipeline/__init__.py missing after copy — re-upload vlsp2025-kd-pipeline dataset")
elif (WORK_DIR / "pipeline").exists():
    print(f"Pipeline already present: {WORK_DIR / 'pipeline'}")
else:
    raise RuntimeError(f"Pipeline not found at {PIPELINE_SRC}\nAdd dataset: thanhduc1108/vlsp2025-kd-pipeline")

if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)
print(f"WORK_DIR: {WORK_DIR}")
"""

CELL_GPU = r"""
import sys, os, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path: sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU! Enable RTX 6000 accelerator in notebook settings.")

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU:  {gpu_name}  ({vram_gb:.0f} GB)")

import transformers, peft, accelerate
print(f"Transformers: {transformers.__version__}  PEFT: {peft.__version__}")

try:
    import flash_attn
    HAS_FLASH = True
    print(f"Flash Attn: {flash_attn.__version__}")
except ImportError:
    HAS_FLASH = False
    print("Flash Attn: not available (sdpa fallback)")

GPU_PROFILE = "rtx6000_96gb" if vram_gb > 80 else ("a100_80gb" if vram_gb > 60 else "p100_16gb")
print(f"GPU Profile: {GPU_PROFILE}")

# Resolve offline model paths (Kaggle offline mode)
def resolve_model(hf_id):
    name = hf_id.split("/")[-1].lower()
    for root in [Path("/kaggle/input"), Path("/kaggle/models")]:
        if not root.exists(): continue
        for d in root.rglob("config.json"):
            parent = d.parent
            if name.replace("-","").replace("_","") in parent.name.lower().replace("-","").replace("_",""):
                print(f"  Found offline: {parent}")
                return str(parent)
    return hf_id

TEACHER_PATH = resolve_model("Qwen/Qwen3.5-27B")
STUDENT_PATH = resolve_model("Qwen/Qwen3.5-4B")
# Fallback to Kaggle model mount paths
if TEACHER_PATH == "Qwen/Qwen3.5-27B":
    p = Path("/kaggle/input/models/thanhduc1108/qwen_35_27b/transformers/default/1")
    if p.exists(): TEACHER_PATH = str(p)
if STUDENT_PATH == "Qwen/Qwen3.5-4B":
    p = Path("/kaggle/input/models/thanhduc1108/qwen_35_4b/transformers/default/1")
    if p.exists(): STUDENT_PATH = str(p)

print(f"Teacher: {TEACHER_PATH}")
print(f"Student: {STUDENT_PATH}")
"""

CELL_CONFIG_TEMPLATE = """
import sys, os
from pathlib import Path
from pipeline.config import load_config, save_config

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path: sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

cfg_overrides = {{
    "model": {{
        "teacher_model": TEACHER_PATH,
        "student_model": STUDENT_PATH,
        "use_flash_attention": HAS_FLASH,
    }},
    "data": {{
        "vinumqa_train":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/train.json",
        "vinumqa_valid":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/valid.json",
        "vinumqa_test":         "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/test.json",
        "vinumqa_private_test": "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/private_test.json",
        "finqa_dir":            "/kaggle/input/datasets/thanhduc1108/finqa-en",
        "max_samples":          TEST_SAMPLES if TEST_MODE else None,
    }},
    "sft": {sft_overrides},
    "inference": {{"num_candidates": 1, "temperature": 0.0, "batch_size": 8}},
}}
cfg = load_config(gpu_profile=GPU_PROFILE, overrides=cfg_overrides)

print(f"{'='*60}")
print(f"TEST_MODE     : {{TEST_MODE}} (n={{}}")
print(f"GPU Profile   : {{GPU_PROFILE}}")
print(f"Teacher       : {{cfg.model.teacher_model.split('/')[-1]}}")
print(f"Student       : {{cfg.model.student_model.split('/')[-1]}}")
print(f"SFT seq len   : {{cfg.sft.max_seq_length}}  |  LoRA r={{cfg.sft.lora_r}}")
print(f"{'='*60}")
save_config(cfg, str(WORK_DIR / "data/pipeline/config_{notebook_id}.yaml"))
"""

CELL_DATAPREP = r"""
import os, sys, time, json
from pathlib import Path
from pipeline.data_prep import run_data_prep

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path: sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)
t0 = time.time()
data_paths = run_data_prep(cfg)
print(f"\nCompleted in {time.time()-t0:.1f}s")

# Print sample counts
for k, v in data_paths.items():
    try:
        d = json.load(open(v))
        print(f"  {k}: {len(d)} samples  ({v})")
    except Exception:
        print(f"  {k}: {v}")
"""

CELL_GREEDY_EVAL = r"""
# ═════════════════════════════════════════════════════════════════════
# GREEDY EVALUATION — fast single-pass batched inference
# Time: ~40-60 min for 584 valid samples with Qwen3.5-4B + RTX 6000
# vs 35h for majority voting (num_candidates=15)
# ═════════════════════════════════════════════════════════════════════
import gc, json, time, torch, warnings
from pathlib import Path
from tqdm import tqdm
from pipeline import _load_model_robust, _load_tokenizer_robust
from pipeline.evaluate import answers_match, programs_match
from pipeline.program_executor import validate_program
import re

# Suppress non-critical deprecation warnings (torch_dtype, warmup_ratio)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*warmup_ratio.*")

WORK_DIR = Path("/kaggle/working/vlsp2025")

VALID_PATH = globals().get("data_paths", {}).get("sft_valid") or str(WORK_DIR / "data/pipeline/sft_valid.json")
with open(VALID_PATH, "r", encoding="utf-8") as f:
    valid_data = json.load(f)

if TEST_MODE:
    valid_data = valid_data[:min(TEST_SAMPLES, len(valid_data))]
    print(f"TEST MODE: using {len(valid_data)} validation samples")

# Resolve display name for the model (avoid showing "1" from .../default/1)
_model_display = sft_model_path
for _part in reversed(sft_model_path.replace("\\", "/").split("/")):
    if len(_part) > 3 and _part not in ("final", "default", "1"):
        _model_display = _part; break
print(f"Evaluating {len(valid_data)} samples | Model: {_model_display}")

# --- Load model ---
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16
# Support both old (torch_dtype) and new (dtype) transformers API
import inspect as _inspect
from transformers import AutoModelForCausalLM as _ACM
_dtype_key = "dtype" if "dtype" in _inspect.signature(_ACM.from_pretrained).parameters else "torch_dtype"
mkw = {"trust_remote_code": True, _dtype_key: dtype, "device_map": "auto", "low_cpu_mem_usage": True}
if cfg.model.use_flash_attention:
    try:
        import flash_attn
        mkw["attn_implementation"] = "flash_attention_2"
    except ImportError:
        mkw["attn_implementation"] = "sdpa"

tokenizer = _load_tokenizer_robust(sft_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

try:
    from peft import PeftModel
    base = _load_model_robust(cfg.model.student_model, mkw)
    model = PeftModel.from_pretrained(base, sft_model_path)
    print("Loaded as PEFT (LoRA) model")
except Exception as _e:
    print(f"PEFT load skipped ({type(_e).__name__}), loading as full model")
    model = _load_model_robust(sft_model_path, mkw)

model.eval()
model.config.pad_token_id = tokenizer.pad_token_id
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f"Model loaded. GPU free: {free_gb:.1f} GB")

# --- Helper: extract program/answer from output text ---
def _extract(text):
    pm = list(re.finditer(r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text))
    am = list(re.finditer(r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", text))
    prog = pm[-1].group(1).strip() if pm else None
    ans  = am[-1].group(1).strip() if am else None
    return prog, ans

# --- Batched greedy inference ---
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512  # output format ~100-300 tokens; 512 is safe ceiling

eval_results = []
batches = [valid_data[i:i+BATCH_SIZE] for i in range(0, len(valid_data), BATCH_SIZE)]
t0 = time.time()

for b_idx, batch in enumerate(tqdm(batches, desc="Greedy eval")):
    prompts  = [s["messages"][0]["content"] for s in batch]
    gold_prog = [s.get("metadata", {}).get("program", "") for s in batch]
    gold_ans  = [str(s.get("metadata", {}).get("answer", "")) for s in batch]

    # Chat template
    texts_in = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:
            t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts_in.append(t)

    enc = tokenizer(texts_in, return_tensors="pt", truncation=True, max_length=3584, padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)

    for i, (txt, samp) in enumerate(zip(gen, batch)):
        pp, pa = _extract(txt)
        ea = answers_match(pa or "", gold_ans[i])
        pa_m = programs_match(pp or "", gold_prog[i])
        eval_results.append({
            "id": samp.get("id", f"s{len(eval_results)}"),
            "pred_program": pp or "",
            "pred_answer":  pa or "",
            "gold_program": gold_prog[i],
            "gold_answer":  gold_ans[i],
            "ea": ea, "pa": pa_m,
            "valid": validate_program(pp) if pp else False,
            "raw_output": txt[:300],
        })

    # Progress log every 10 batches
    if (b_idx + 1) % 10 == 0:
        el = time.time() - t0
        eta = (len(batches) - b_idx - 1) / ((b_idx + 1) / el)
        n_done = min((b_idx + 1) * BATCH_SIZE, len(valid_data))
        ea_now = sum(r["ea"] for r in eval_results) / len(eval_results) * 100
        print(f"  [{n_done}/{len(valid_data)}] EA={ea_now:.1f}%  ETA={eta/60:.0f}min")

elapsed = time.time() - t0
N = len(eval_results)
EA = sum(r["ea"] for r in eval_results) / N * 100 if N else 0
PA = sum(r["pa"] for r in eval_results) / N * 100 if N else 0
VR = sum(r["valid"] for r in eval_results) / N * 100 if N else 0

print(f"\n{'='*60}")
print(f"GREEDY EVALUATION ({N} samples,  {elapsed:.0f}s / {elapsed/60:.1f}min)")
print(f"{'='*60}")
print(f"  Execution Accuracy (EA): {EA:.2f}%   ({sum(r['ea'] for r in eval_results)}/{N})")
print(f"  Program Accuracy   (PA): {PA:.2f}%   ({sum(r['pa'] for r in eval_results)}/{N})")
print(f"  Valid Program Rate:      {VR:.2f}%")
print(f"{'='*60}")

eval_summary = {
    "execution_accuracy": EA / 100,
    "program_accuracy":   PA / 100,
    "valid_rate":         VR / 100,
    "total": N,
    "ea_correct": sum(r["ea"] for r in eval_results),
    "pa_correct": sum(r["pa"] for r in eval_results),
    "details": eval_results,
}

# Free GPU after eval — model no longer needed
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU free after eval cleanup: {free_gb:.1f} GB")
"""

CELL_ERROR_ANALYSIS = r"""
# ═════════════════════════════════════════════════════════════════════
# DETAILED ERROR ANALYSIS — breakdown of prediction quality
# ═════════════════════════════════════════════════════════════════════
import json
from pathlib import Path

N = len(eval_results)
perfect   = sum(1 for r in eval_results if r["ea"] and r["pa"])
ea_only   = sum(1 for r in eval_results if r["ea"] and not r["pa"])
valid_err = sum(1 for r in eval_results if not r["ea"] and r["valid"])
inv_prog  = sum(1 for r in eval_results if not r["ea"] and not r["valid"] and r["pred_program"])
no_prog   = sum(1 for r in eval_results if not r["pred_program"])

print(f"\n{'='*60}")
print(f"ERROR ANALYSIS  (N={N})")
print(f"{'='*60}")
print(f"{'Category':<35} {'Count':>6}  {'%':>6}")
print(f"{'-'*50}")
print(f"{'EA + PA correct (perfect)' :<35} {perfect:>6}  {perfect/N*100:>5.1f}%")
print(f"{'EA correct, PA wrong (alt solution)':<35} {ea_only:>6}  {ea_only/N*100:>5.1f}%")
print(f"{'EA wrong, valid program':<35} {valid_err:>6}  {valid_err/N*100:>5.1f}%")
print(f"{'EA wrong, invalid program':<35} {inv_prog:>6}  {inv_prog/N*100:>5.1f}%")
print(f"{'No program generated':<35} {no_prog:>6}  {no_prog/N*100:>5.1f}%")
print(f"{'-'*50}")
print(f"{'Total':<35} {N:>6}  100.0%")
print(f"{'='*60}")

# Show 5 failure examples
print("\nFAILURE EXAMPLES (EA wrong):")
failures = [r for r in eval_results if not r["ea"]][:5]
for i, r in enumerate(failures, 1):
    print(f"\n[{i}] id={r['id']}")
    print(f"  Gold  : program={r['gold_program'][:60]}  answer={r['gold_answer']}")
    print(f"  Pred  : program={r['pred_program'][:60]}  answer={r['pred_answer']}")
    print(f"  Valid : {r['valid']}")
"""

CELL_SAVE_RESULTS_TEMPLATE = r"""
import gc, json, shutil, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)
pipeline_out = WORK_DIR / "data/pipeline"

# Build and save eval_results.json
eval_out = dict(notebook=NOTEBOOK_ID, test_mode=TEST_MODE, model_path=sft_model_path)
for k, v in eval_summary.items():
    if k != "details":
        eval_out[k] = v
eval_out["details"] = eval_summary["details"]
with open(out / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_out, f, ensure_ascii=False, indent=2)
print(f"eval_results.json saved  (EA={eval_out['execution_accuracy']:.2%}  PA={eval_out['program_accuracy']:.2%})")

# Compact predictions for cross-notebook comparison
preds_compact = [
    dict(id=r["id"], pred_answer=r["pred_answer"], gold_answer=r["gold_answer"],
         ea=r["ea"], pa=r["pa"])
    for r in eval_summary["details"]
]
with open(out / "predictions.json", "w", encoding="utf-8") as f:
    json.dump(preds_compact, f, ensure_ascii=False, indent=2)

# Save model adapter (LoRA weights)
model_save = out / "sft_adapter"
if model_save.exists():
    shutil.rmtree(model_save)
if Path(sft_model_path).exists():
    shutil.copytree(sft_model_path, model_save)
    sz = sum(p.stat().st_size for p in model_save.rglob("*") if p.is_file()) / 1024**2
    print(f"Model adapter saved -> {model_save}  ({sz:.0f} MB)")
else:
    print(f"WARNING: sft_model_path not found: {sft_model_path}")

# Copy data / config files
for fname in ["sft_train.json", "sft_valid.json", "config_{notebook_id}.yaml"]:
    src = pipeline_out / fname
    if src.exists():
        shutil.copy2(src, out / fname)

print(f"\nAll outputs -> {OUTPUT_DIR}")
print(f"Files: {sorted(p.name for p in out.iterdir())}")
print(f"\nSUMMARY  ({'TEST MODE - ' + str(TEST_SAMPLES) + ' samples' if TEST_MODE else 'FULL RUN'}):")
print(f"  EA (Execution Accuracy): {eval_summary['execution_accuracy']:.2%}")
print(f"  PA (Program Accuracy)  : {eval_summary['program_accuracy']:.2%}")
print(f"  Valid Program Rate     : {eval_summary['valid_rate']:.2%}")
if TEST_MODE:
    print(f"\n  ⚠ TEST_MODE=True — numbers above are meaningless (only {TEST_SAMPLES} training samples).")
    print(f"  ⚠ Set TEST_MODE=False, restart kernel, and rerun for real results.")
"""

# ═══════════════════════════════════════════════════════════════════
# NOTEBOOK 1: DISTILLATION ONLY
# ═══════════════════════════════════════════════════════════════════

def make_nb1():
    cells = []

    cells.append(md_cell("""# VLSP 2025 KD — Notebook 1: Teacher Distillation Only

**Goal**: Run teacher distillation (Qwen3.5-27B → reasoning traces), evaluate quality
**Output**: `distilled_sft.json` for NB3, match rate stats
**No student training** in this notebook — just measure what the teacher produces

| Step | Time (causal_conv1d) | Time (no causal_conv1d) |
|------|---------------------|------------------------|
| Test (50 samples) | ~5 min | ~14 min |
| Full (~3000 samples) | ~5-7h | ~13-15h ⚠️ |

> ⚠️ **WITHOUT causal_conv1d**: distillation may NOT finish in 12h.
> In that case, the notebook auto-saves checkpoints every 100 samples — start Session 2 and resume.

## Execution Order
1. **Run ALL cells with `TEST_MODE=True`** (~15 min) — verify no errors
2. Check Cell 7 stats (match rate should be >80%)
3. Set `TEST_MODE=False`, restart kernel, run all cells (full run, ~5-7h)
4. Download `distilled_sft.json` from outputs — use as input to NB3"""))

    cells.append(code_cell("""# ════════════════════════════════════════════════════════════════
# CELL 1: CONFIGURATION — read this before running!
# ════════════════════════════════════════════════════════════════

# ▶ TEST BEFORE FULL RUN
TEST_MODE    = True     # ← SET TO False AFTER TEST PASSES
TEST_SAMPLES = 50       # samples in test mode (distillation only)

NOTEBOOK_ID = "distill-only"
OUTPUT_DIR  = f"/kaggle/working/outputs/{NOTEBOOK_ID}"

import os
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"Notebook  : {NOTEBOOK_ID}")
print(f"TEST_MODE : {TEST_MODE}  (n={TEST_SAMPLES if TEST_MODE else 'ALL'})")
print(f"Output    : {OUTPUT_DIR}")
print(f"{'='*60}")
"""))

    cells.append(code_cell(CELL_INSTALL))
    cells.append(code_cell(CELL_GPU))

    cells.append(code_cell("""import sys, os
from pathlib import Path
from pipeline.config import load_config, save_config

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path: sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

cfg = load_config(gpu_profile=GPU_PROFILE, overrides={
    "model": {"teacher_model": TEACHER_PATH, "student_model": STUDENT_PATH,
              "use_flash_attention": HAS_FLASH},
    "data": {
        "vinumqa_train":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/train.json",
        "vinumqa_valid":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/valid.json",
        "vinumqa_test":         "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/test.json",
        "vinumqa_private_test": "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/private_test.json",
        "finqa_dir":            "/kaggle/input/datasets/thanhduc1108/finqa-en",
        "max_samples":          TEST_SAMPLES if TEST_MODE else None,
    },
})
print(f"Teacher: {cfg.model.teacher_model.split('/')[-1]}")
print(f"batch_size={cfg.teacher.batch_size}  max_new_tokens={cfg.teacher.max_new_tokens}")
print(f"use_guided_template={cfg.teacher.use_guided_template}")
save_config(cfg, str(WORK_DIR / "data/pipeline/config_distill.yaml"))
"""))

    cells.append(code_cell(CELL_DATAPREP))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 6: TIMING ESTIMATE — check before committing to full run
# ════════════════════════════════════════════════════════════════
import json, torch
from pathlib import Path

WORK_DIR     = Path("/kaggle/working/vlsp2025")
pipeline_out = WORK_DIR / "data/pipeline"

total = done = 0
if (pipeline_out / "teacher_input.json").exists():
    dataset = json.load(open(pipeline_out / "teacher_input.json"))
    total = len(dataset)
if (pipeline_out / "teacher_raw_checkpoint.json").exists():
    ckpt = json.load(open(pipeline_out / "teacher_raw_checkpoint.json"))
    done = len(ckpt)
    matched = sum(1 for r in ckpt if r.get("matched", False))
    print(f"Checkpoint: {done}/{total} done, {matched} matched ({matched/max(done,1)*100:.0f}%)")

remaining = total - done
try:
    import causal_conv1d
    tps = 40  # tokens/sec
    conv1d_status = "INSTALLED (fast)"
except ImportError:
    tps = 17  # tokens/sec
    conv1d_status = "MISSING (2.3x slower — risk timeout!)"

batch = cfg.teacher.batch_size
avg_out = 280  # guided template output avg tokens
s_per_sample = avg_out / tps  # seconds per sample (batch parallelism)
eta_h = remaining * s_per_sample / 3600

print(f"\ncausal_conv1d : {conv1d_status}")
print(f"batch_size    : {batch}  |  max_new_tokens: {cfg.teacher.max_new_tokens}")
print(f"Throughput est: {tps} tok/s  =>  {s_per_sample:.1f}s/sample")
print(f"Remaining     : {remaining} samples")
print(f"Est. time     : {eta_h:.1f}h")
if eta_h > 10.5:
    print(f"\nWARNING: May exceed 12h! Consider:")
    print(f"  - Reducing total samples (cfg.data.max_samples)")
    print(f"  - Using 2-session approach (checkpoint auto-saves every 100 samples)")
"""))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 7: TEACHER DISTILLATION
# Checkpoints every 100 samples — safe to interrupt and resume
# ════════════════════════════════════════════════════════════════
import gc, os, sys, time, torch
from pathlib import Path
from pipeline.teacher_distill import run_teacher_distillation

WORK_DIR = Path("/kaggle/working/vlsp2025")
os.chdir(WORK_DIR)

print("=" * 65)
print("TEACHER DISTILLATION")
print(f"  guided_template : {cfg.teacher.use_guided_template}")
print(f"  max_new_tokens  : {cfg.teacher.max_new_tokens}")
print(f"  batch_size      : {cfg.teacher.batch_size}")
print(f"  checkpoint_every: {cfg.teacher.checkpoint_every}")
print("=" * 65)

t0 = time.time()
distilled_path = run_teacher_distillation(
    cfg,
    teacher_input_path=data_paths["teacher_input"],
    resume=True,   # safe to restart — picks up from checkpoint
)
elapsed = time.time() - t0
print(f"\nDistillation done in {elapsed/3600:.2f}h  ({elapsed:.0f}s)")
print(f"Output: {distilled_path}")

# ── Aggressive GPU cleanup ──────────────────────────────────────
for _n in list(globals()):
    if "teacher" in _n.lower():
        try: del globals()[_n]
        except: pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU free after cleanup: {free_b/1024**3:.1f} / {total_b/1024**3:.1f} GB")
"""))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 8: EVALUATE DISTILLATION QUALITY
# Shows how well teacher traces match gold answers
# ════════════════════════════════════════════════════════════════
import json
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
pipeline_out = WORK_DIR / "data/pipeline"

# Load distilled data
with open(distilled_path, "r", encoding="utf-8") as f:
    distilled = json.load(f)

# Load raw checkpoint for full stats
ckpt_path = pipeline_out / "teacher_raw_checkpoint.json"
raw_stats = {"exact_match": 0, "answer_match": 0, "program_valid": 0, "invalid": 0, "error": 0}
total_raw = 0
if ckpt_path.exists():
    raw = json.load(open(ckpt_path))
    total_raw = len(raw)
    for r in raw:
        mt = r.get("match_type", "invalid")
        raw_stats[mt] = raw_stats.get(mt, 0) + 1

print(f"\n{'='*65}")
print(f"DISTILLATION QUALITY REPORT")
print(f"{'='*65}")
print(f"Total processed    : {total_raw}")
print(f"Distilled (usable) : {len(distilled)}  ({len(distilled)/max(total_raw,1)*100:.1f}%)")
print()
print(f"{'Match type':<25} {'Count':>7}  {'%':>6}")
print(f"{'-'*42}")
for mt, cnt in sorted(raw_stats.items(), key=lambda x: -x[1]):
    pct = cnt / max(total_raw, 1) * 100
    bar = '█' * int(pct / 3)
    print(f"  {mt:<23} {cnt:>7}  {pct:>5.1f}%  {bar}")
print(f"{'='*65}")
print(f"\nQuality breakdown:")
usable = raw_stats.get("exact_match", 0) + raw_stats.get("answer_match", 0)
print(f"  Usable for SFT (matched): {usable}/{total_raw} = {usable/max(total_raw,1)*100:.1f}%")
print(f"  Avg output length: ~280 tokens (guided template)")
print(f"\nData saved to: {distilled_path}")
"""))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 9: SAVE OUTPUTS
# Saves distilled_sft.json for use in NB3
# ════════════════════════════════════════════════════════════════
import json, shutil
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)
pipeline_out = WORK_DIR / "data/pipeline"

# Copy files to /kaggle/working/outputs/distill-only/
FILES_TO_SAVE = [
    "distilled_sft.json",       # ← import this in NB3!
    "teacher_raw_checkpoint.json",
    "teacher_raw_output.json",
    "sft_train.json",
    "sft_valid.json",
    "config_distill.yaml",
]
for fname in FILES_TO_SAVE:
    src = pipeline_out / fname
    if src.exists():
        shutil.copy2(src, out / fname)
        sz = src.stat().st_size / 1024
        print(f"  Saved: {fname}  ({sz:.0f} KB)")
    else:
        print(f"  (skip) {fname} not found")

# Summary
import json as _json
raw_ckpt_path = pipeline_out / "teacher_raw_checkpoint.json"
if raw_ckpt_path.exists():
    raw = _json.load(open(raw_ckpt_path))
    matched = sum(1 for r in raw if r.get("matched", False))
    summary = {
        "notebook": NOTEBOOK_ID,
        "test_mode": TEST_MODE,
        "total_processed": len(raw),
        "matched": matched,
        "match_rate": matched / max(len(raw), 1),
        "distilled_path": str(out / "distilled_sft.json"),
    }
    with open(out / "distill_summary.json", "w", encoding="utf-8") as f:
        _json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary: {matched}/{len(raw)} matched ({matched/max(len(raw),1)*100:.1f}%)")

print(f"\nAll outputs -> {OUTPUT_DIR}")
print(f"Files: {[p.name for p in sorted(out.iterdir())]}")
print("\nNEXT STEP: Use distilled_sft.json in NB3 (rtx6000-eval-kd.ipynb)")
"""))

    return make_nb(cells)


# ═══════════════════════════════════════════════════════════════════
# NOTEBOOK 2: SFT ONLY (GOLD DATA BASELINE)
# ═══════════════════════════════════════════════════════════════════

def make_nb2():
    cells = []

    cells.append(md_cell("""# VLSP 2025 KD — Notebook 2: SFT Only (Gold Data Baseline)

**Goal**: Train student (Qwen3.5-4B) on gold SFT data (no teacher distillation)
**Baseline**: What EA/PA does SFT alone achieve?
**Output**: SFT adapter + EA/PA metrics for comparison with NB3

| Step | Time |
|------|------|
| Test mode (50 samples) | ~15-20 min |
| Full run | ~1.5-2h |

This is the **fastest** notebook — ideal to run first to establish a baseline.

## Execution Order
1. Run ALL cells with `TEST_MODE=True` (~15 min)
2. Verify Cell 6 shows loss < 0.5 and Cell 7 shows valid EA/PA
3. Set `TEST_MODE=False`, restart kernel, full run (~1.5-2h)"""))

    cells.append(code_cell("""# ════════════════════════════════════════════════════════════════
# CELL 1: CONFIGURATION
# ════════════════════════════════════════════════════════════════

TEST_MODE    = True     # ← SET TO False AFTER TEST PASSES
TEST_SAMPLES = 50       # samples in test mode

NOTEBOOK_ID = "sft-only"
OUTPUT_DIR  = f"/kaggle/working/outputs/{NOTEBOOK_ID}"

import os
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"Notebook  : {NOTEBOOK_ID}  (SFT on gold data — no teacher distillation)")
print(f"TEST_MODE : {TEST_MODE}  (n={TEST_SAMPLES if TEST_MODE else 'ALL'})")
print(f"Output    : {OUTPUT_DIR}")
print(f"{'='*60}")
"""))

    cells.append(code_cell(CELL_INSTALL))
    cells.append(code_cell(CELL_GPU))

    cells.append(code_cell("""import sys, os
from pathlib import Path
from pipeline.config import load_config, save_config

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path: sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

cfg = load_config(gpu_profile=GPU_PROFILE, overrides={
    "model": {"teacher_model": TEACHER_PATH, "student_model": STUDENT_PATH,
              "use_flash_attention": HAS_FLASH},
    "data": {
        "vinumqa_train":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/train.json",
        "vinumqa_valid":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/valid.json",
        "vinumqa_test":         "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/test.json",
        "vinumqa_private_test": "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/private_test.json",
        "finqa_dir":            "/kaggle/input/datasets/thanhduc1108/finqa-en",
        "max_samples":          TEST_SAMPLES if TEST_MODE else None,
    },
    "sft": {"num_epochs": 3, "lora_r": 64},  # smaller r=64 for gold-only (less data)
    "inference": {"num_candidates": 1, "batch_size": 8},
})
print(f"Student: {cfg.model.student_model.split('/')[-1]}")
print(f"SFT: epochs={cfg.sft.num_epochs}  batch={cfg.sft.per_device_train_batch_size}x{cfg.sft.gradient_accumulation_steps}  LoRA r={cfg.sft.lora_r}  seq={cfg.sft.max_seq_length}")
save_config(cfg, str(WORK_DIR / "data/pipeline/config_sft_only.yaml"))
"""))

    cells.append(code_cell(CELL_DATAPREP))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 6: SFT TRAINING ON GOLD DATA
# Uses sft_train.json (gold program-answer pairs, no teacher)
# Time: ~30-40 min for full run (3 epochs, 3000 samples, LoRA r=64)
# ════════════════════════════════════════════════════════════════
import gc, os, sys, time, torch
from pathlib import Path
from pipeline.train_sft import run_sft_training

WORK_DIR = Path("/kaggle/working/vlsp2025")
pipeline_out = WORK_DIR / "data/pipeline"

# Use gold SFT data (NOT distilled) — this is the SFT-only baseline
train_path = data_paths.get("sft_train", str(pipeline_out / "sft_train.json"))
valid_path = data_paths.get("sft_valid", str(pipeline_out / "sft_valid.json"))

print("=" * 60)
print("SFT TRAINING (GOLD DATA)")
print("=" * 60)
print(f"  Train: {train_path}")
print(f"  Valid: {valid_path}")
print(f"  LoRA r={cfg.sft.lora_r}  seq_len={cfg.sft.max_seq_length}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  GPU free: {free_gb:.1f} GB")

# Resume from checkpoint if it exists
sft_ckpt_dir = WORK_DIR / "checkpoints/sft_only"
resume_ckpt = None
if sft_ckpt_dir.exists():
    ckpts = sorted([d for d in sft_ckpt_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[-1]))
    if ckpts:
        resume_ckpt = str(ckpts[-1])
        print(f"  Resuming from: {resume_ckpt}")

# Override output dir so it doesn't conflict with main notebook
import dataclasses
cfg.sft = dataclasses.replace(cfg.sft, output_dir="checkpoints/sft_only")

t0 = time.time()
sft_model_path = run_sft_training(
    cfg,
    train_path=train_path,
    valid_path=valid_path,
    resume_from_checkpoint=resume_ckpt,
)
elapsed = time.time() - t0
print(f"\nSFT done in {elapsed/3600:.2f}h ({elapsed:.0f}s)")
print(f"Model: {sft_model_path}")

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""))

    cells.append(code_cell(CELL_GREEDY_EVAL))
    cells.append(code_cell(CELL_ERROR_ANALYSIS))

    save_cell = CELL_SAVE_RESULTS_TEMPLATE.replace("{notebook_id}", "sft_only")
    cells.append(code_cell(save_cell))

    return make_nb(cells)


# ═══════════════════════════════════════════════════════════════════
# NOTEBOOK 3: KNOWLEDGE DISTILLATION (DISTILL + SFT)
# ═══════════════════════════════════════════════════════════════════

def make_nb3():
    cells = []

    cells.append(md_cell("""# VLSP 2025 KD — Notebook 3: Knowledge Distillation (Distill + SFT)

**Goal**: Full KD pipeline — teacher distillation → SFT on distilled traces → evaluate
**Comparison**: vs NB2 (SFT-only) — does KD actually improve EA/PA?
**Time**: ~7-9h (if running distillation fresh with causal_conv1d)

| Approach | How to run | Time |
|----------|-----------|------|
| **Fastest** | Load distilled data from NB1 output | ~1.5-2h |
| **Full** | Run distillation from scratch | ~7-9h |

> **Recommended**: Run NB1 first, save `distilled_sft.json`, then set
> `LOAD_DISTILLED_FROM` below to skip distillation in this notebook.

## Execution Order
1. (Optional) Set `LOAD_DISTILLED_FROM` to NB1 output path
2. Run ALL cells with `TEST_MODE=True` (~20-40 min)
3. Verify no errors, check Cell 9 EA/PA
4. Set `TEST_MODE=False`, restart kernel, full run"""))

    cells.append(code_cell("""# ════════════════════════════════════════════════════════════════
# CELL 1: CONFIGURATION
# ════════════════════════════════════════════════════════════════

TEST_MODE    = True     # ← SET TO False AFTER TEST PASSES
TEST_SAMPLES = 50       # samples in test mode

NOTEBOOK_ID = "kd"
OUTPUT_DIR  = f"/kaggle/working/outputs/{NOTEBOOK_ID}"

# ─── If you ran NB1 already, set this to skip re-distillation ──────
# Set to path of distilled_sft.json from NB1, or leave None to run fresh
LOAD_DISTILLED_FROM = None  # e.g. "/kaggle/input/distill-only-output/distilled_sft.json"
# ───────────────────────────────────────────────────────────────────

import os
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"Notebook           : {NOTEBOOK_ID}  (KD = Distill + SFT)")
print(f"TEST_MODE          : {TEST_MODE}  (n={TEST_SAMPLES if TEST_MODE else 'ALL'})")
print(f"LOAD_DISTILLED_FROM: {LOAD_DISTILLED_FROM or '(run distillation fresh)'}")
print(f"Output             : {OUTPUT_DIR}")
print(f"{'='*60}")
"""))

    cells.append(code_cell(CELL_INSTALL))
    cells.append(code_cell(CELL_GPU))

    cells.append(code_cell("""import sys, os
from pathlib import Path
from pipeline.config import load_config, save_config

WORK_DIR = Path("/kaggle/working/vlsp2025")
if str(WORK_DIR) not in sys.path: sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

cfg = load_config(gpu_profile=GPU_PROFILE, overrides={
    "model": {"teacher_model": TEACHER_PATH, "student_model": STUDENT_PATH,
              "use_flash_attention": HAS_FLASH},
    "data": {
        "vinumqa_train":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/train.json",
        "vinumqa_valid":        "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/valid.json",
        "vinumqa_test":         "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/test.json",
        "vinumqa_private_test": "/kaggle/input/datasets/thanhduc1108/vinumericalqa-private/private_test.json",
        "finqa_dir":            "/kaggle/input/datasets/thanhduc1108/finqa-en",
        "max_samples":          TEST_SAMPLES if TEST_MODE else None,
    },
    "sft": {"num_epochs": 3, "lora_r": 128},  # larger r=128 for KD (distilled data)
    "inference": {"num_candidates": 1, "batch_size": 8},
})
print(f"Teacher: {cfg.model.teacher_model.split('/')[-1]}")
print(f"Student: {cfg.model.student_model.split('/')[-1]}")
print(f"SFT: epochs={cfg.sft.num_epochs}  LoRA r={cfg.sft.lora_r}  seq={cfg.sft.max_seq_length}")
save_config(cfg, str(WORK_DIR / "data/pipeline/config_kd.yaml"))
"""))

    cells.append(code_cell(CELL_DATAPREP))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 6: TEACHER DISTILLATION (or load from NB1)
# ════════════════════════════════════════════════════════════════
import gc, os, sys, time, json, shutil, torch
from pathlib import Path

WORK_DIR = Path("/kaggle/working/vlsp2025")
pipeline_out = WORK_DIR / "data/pipeline"

# --- Try to load distilled data from NB1 output ---
distilled_path = None

# Priority 1: explicit path from config
if LOAD_DISTILLED_FROM and Path(LOAD_DISTILLED_FROM).exists():
    distilled_path = LOAD_DISTILLED_FROM
    print(f"Loaded distilled data from NB1: {distilled_path}")
    d = json.load(open(distilled_path))
    print(f"  Samples: {len(d)}")
    # Copy to pipeline_out for SFT to pick up
    dst = pipeline_out / "distilled_sft.json"
    if not dst.exists():
        shutil.copy2(distilled_path, dst)

# Priority 2: already exists in working dir
if distilled_path is None:
    for fname in ["distilled_sft.json"]:
        p = pipeline_out / fname
        if p.exists():
            distilled_path = str(p)
            d = json.load(open(distilled_path))
            print(f"Found existing distilled data: {distilled_path}  ({len(d)} samples)")
            break

# Priority 3: run teacher distillation from scratch
if distilled_path is None:
    print("No distilled data found — running teacher distillation now...")
    print(f"Estimated time: {cfg.teacher.checkpoint_every} samples/checkpoint, ~5-7h total")
    from pipeline.teacher_distill import run_teacher_distillation
    t0 = time.time()
    distilled_path = run_teacher_distillation(
        cfg,
        teacher_input_path=data_paths["teacher_input"],
        resume=True,
    )
    elapsed = time.time() - t0
    print(f"\nDistillation done in {elapsed/3600:.2f}h")
    # Aggressive GPU cleanup after 27B model
    for _n in list(globals()):
        if "teacher" in _n.lower():
            try: del globals()[_n]
            except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        free_b, tot_b = torch.cuda.mem_get_info()
        print(f"GPU free after cleanup: {free_b/1024**3:.1f}/{tot_b/1024**3:.1f} GB")

print(f"\nDistilled path: {distilled_path}")
"""))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 7: SFT ON DISTILLED DATA
# Training student on teacher-generated reasoning traces
# Time: ~35-50 min (3 epochs, ~2500 distilled samples)
# ════════════════════════════════════════════════════════════════
import gc, os, sys, time, torch, dataclasses
from pathlib import Path
from pipeline.train_sft import run_sft_training

WORK_DIR = Path("/kaggle/working/vlsp2025")
pipeline_out = WORK_DIR / "data/pipeline"

valid_path = data_paths.get("sft_valid", str(pipeline_out / "sft_valid.json"))

print("=" * 60)
print("SFT TRAINING ON DISTILLED DATA")
print("=" * 60)
print(f"  Train (distilled): {distilled_path}")
print(f"  Valid            : {valid_path}")
print(f"  LoRA r={cfg.sft.lora_r}  seq_len={cfg.sft.max_seq_length}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  GPU free: {free_gb:.1f} GB")

# Resume from checkpoint if it exists
sft_ckpt_dir = WORK_DIR / "checkpoints/sft_kd"
resume_ckpt = None
if sft_ckpt_dir.exists():
    ckpts = sorted([d for d in sft_ckpt_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[-1]))
    if ckpts:
        resume_ckpt = str(ckpts[-1])
        print(f"  Resuming from: {resume_ckpt}")

# Use separate output dir from SFT-only
cfg.sft = dataclasses.replace(cfg.sft, output_dir="checkpoints/sft_kd")

t0 = time.time()
sft_model_path = run_sft_training(
    cfg,
    train_path=distilled_path,
    valid_path=valid_path,
    resume_from_checkpoint=resume_ckpt,
)
elapsed = time.time() - t0
print(f"\nSFT done in {elapsed/3600:.2f}h ({elapsed:.0f}s)")
print(f"Model: {sft_model_path}")

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""))

    cells.append(code_cell(CELL_GREEDY_EVAL))
    cells.append(code_cell(CELL_ERROR_ANALYSIS))

    cells.append(code_cell(r"""# ════════════════════════════════════════════════════════════════
# CELL 10: COMPARISON WITH SFT-ONLY BASELINE (NB2)
# ════════════════════════════════════════════════════════════════
import json
from pathlib import Path

# Try to load NB2 results for comparison
NB2_RESULTS_PATH = "/kaggle/working/outputs/sft-only/eval_results.json"

print(f"\n{'='*65}")
print(f"COMPARISON: SFT-only (NB2) vs Knowledge Distillation (NB3)")
print(f"{'='*65}")

kd_ea = eval_summary["execution_accuracy"]
kd_pa = eval_summary["program_accuracy"]
kd_vr = eval_summary["valid_rate"]

if Path(NB2_RESULTS_PATH).exists():
    nb2 = json.load(open(NB2_RESULTS_PATH))
    base_ea = nb2["execution_accuracy"]
    base_pa = nb2["program_accuracy"]
    base_vr = nb2["valid_rate"]
    base_n  = nb2.get("total", "?")

    print(f"\n{'Model':<35} {'EA':>8} {'PA':>8} {'Valid%':>8}")
    print(f"{'-'*62}")
    print(f"{'SFT-only (NB2, gold data)':<35} {base_ea:>7.2%} {base_pa:>7.2%} {base_vr:>7.2%}")
    print(f"{'KD (NB3, distilled data)':<35} {kd_ea:>7.2%} {kd_pa:>7.2%} {kd_vr:>7.2%}")
    print(f"{'-'*62}")
    delta_ea = kd_ea - base_ea
    delta_pa = kd_pa - base_pa
    sign_ea = "+" if delta_ea >= 0 else ""
    sign_pa = "+" if delta_pa >= 0 else ""
    print(f"{'KD Improvement':<35} {sign_ea}{delta_ea:>6.2%}  {sign_pa}{delta_pa:>6.2%}")
    print(f"{'='*65}")

    if delta_ea > 0:
        print(f"\nKD IMPROVED EA by {delta_ea:.2%} ({delta_ea*100:.1f} percentage points)")
    elif delta_ea < 0:
        print(f"\nKD DECREASED EA by {abs(delta_ea):.2%} — investigate why!")
    else:
        print("\nKD showed no change in EA (same as SFT-only)")
else:
    print(f"NB2 results not found at {NB2_RESULTS_PATH}")
    print(f"Run NB2 (rtx6000-eval-sft-only.ipynb) first to get comparison.")
    print(f"\nKD-only results:")
    print(f"  EA: {kd_ea:.2%}  PA: {kd_pa:.2%}  Valid: {kd_vr:.2%}")
"""))

    save_cell = CELL_SAVE_RESULTS_TEMPLATE.replace("{notebook_id}", "kd")
    # Also save comparison
    save_cell += r"""
# Save comparison results if NB2 available
import json
NB2_RESULTS_PATH = "/kaggle/working/outputs/sft-only/eval_results.json"
if Path(NB2_RESULTS_PATH).exists():
    nb2 = json.load(open(NB2_RESULTS_PATH))
    comparison = {
        "sft_only": {k: nb2[k] for k in ["execution_accuracy","program_accuracy","valid_rate"]},
        "kd":       {k: eval_summary[k] for k in ["execution_accuracy","program_accuracy","valid_rate"]},
        "delta_ea": eval_summary["execution_accuracy"] - nb2["execution_accuracy"],
        "delta_pa": eval_summary["program_accuracy"]   - nb2["program_accuracy"],
    }
    with open(out / "comparison_nb2_vs_nb3.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Comparison saved: {out / 'comparison_nb2_vs_nb3.json'}")
"""
    cells.append(code_cell(save_cell))

    return make_nb(cells)


# ═══════════════════════════════════════════════════════════════════
# GENERATE ALL NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = Path(__file__).parent

    notebooks = [
        ("rtx6000-eval-distill-only.ipynb", make_nb1()),
        ("rtx6000-eval-sft-only.ipynb",     make_nb2()),
        ("rtx6000-eval-kd.ipynb",           make_nb3()),
    ]

    for fname, nb_data in notebooks:
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, ensure_ascii=False, indent=1)
        n_cells = len(nb_data["cells"])
        print(f"Created: {path}  ({n_cells} cells)")

    print("\nAll notebooks generated. Upload to Kaggle:")
    print("  kaggle kernels push -p kaggle/")
