# GitHub Copilot Instructions — VLSP 2025 Knowledge Distillation Pipeline

## Project Overview
Knowledge Distillation pipeline for Vietnamese financial numerical reasoning (VLSP 2025). A large **teacher model** (Qwen3.5-27B) generates structured reasoning traces that train a compact **student model** (Qwen3.5-4B) via SFT + GRPO. Primary metric is **PA (Program Accuracy)**, not EA.

## Pipeline Architecture (6 Phases)
```
data_prep → teacher → sft → grpo → inference → evaluate
```
- **`pipeline/`** — Core pipeline; each phase is a standalone module (`data_prep.py`, `teacher_distill.py`, `train_sft.py`, `train_grpo.py`, `inference.py`, `evaluate.py`)
- **`pipeline/config.py`** — All configuration via `PipelineConfig` dataclass + GPU-specific `GPU_PROFILES` dict; no ad-hoc config files
- **`pipeline/program_executor.py`** — Executes the DSL (add/subtract/multiply/divide/exp/greater/table_sum/table_average/table_max/table_min)
- **`pipeline/reward.py`** — PCPO reward: `R = R_valid × (0.7 + 0.2×R_exec + 0.1×R_bonus)`
- **`src/assets/template.py`** — The Vietnamese prompt template; all three output sections are in Vietnamese

## Running the Pipeline

```bash
# Test run (P100 16GB, 50 samples)
python -m pipeline.run --gpu-profile p100_16gb --phases all --max-samples 50

# Production (RTX 6000 Pro 96GB)
python -m pipeline.run --gpu-profile rtx6000_96gb --phases all

# Specific phases only
python -m pipeline.run --gpu-profile p100_16gb --phases data_prep teacher sft

# Pipeline resumes automatically from pipeline_state.json if interrupted
```

## GPU Profiles
Defined in `pipeline/config.py → GPU_PROFILES`. Never hard-code VRAM-sensitive settings — always pass `--gpu-profile`:
| Profile | Teacher | Student | Flash Attn | dtype |
|---|---|---|---|---|
| `p100_16gb` | Qwen3.5-4B (4bit) | Qwen3.5-0.8B | ❌ | fp16 |
| `rtx6000_96gb` | Qwen3.5-27B | Qwen3.5-4B | ✅ | bf16 |

Flash Attention 2 requires separate install and is **not available on P100**:
```bash
pip install flash-attn --no-build-isolation  # RTX/A100/H100 only
```

## Output Format Convention
All model outputs must follow this **exact three-section Vietnamese structure** (parsed by regex in `reward.py`, `teacher_distill.py`, `evaluate.py`):
```
**Phân tích lập luận:**
<step-by-step reasoning>

**Chương trình tính toán:**
divide(914, 391), multiply(#0, const_100)

**Đáp án cuối cùng:**
30.28486
```
Back-references use `#0`, `#1`, … syntax. Constants are `const_100`, `const_1000`, `const_m1`, etc.

## Data Flow
- Raw datasets: `dataset/viNumericalQA_private/` (train/valid/test/private_test) + `dataset/finqa_en/`
- Prepared data output: `data/pipeline/` (JSON files for each phase)
- SFT/GRPO checkpoints: `checkpoints/sft/`, `checkpoints/grpo/`
- Inference & eval outputs: `data/pipeline/eval_results.json`, `predictions.json`
- State persistence: `data/pipeline/pipeline_state.json` (auto-saved after each phase)

## Key Conventions
- **Table handling**: Raw tables (2D list) are converted to Markdown via `data_prep.table_to_markdown()` before being inserted into prompts
- **Program augmentation**: `use_program_re=True` in `DataConfig` adds reversed-operation variants to training data (boosts from ~11K to 14.6K samples)
- **Vietnamese text**: Always use `encoding="utf-8"` (`src/utils.py` sets `VIETNAMESE_ENCODING`); `json.dump(..., ensure_ascii=False)`
- **Answer format**: 5 decimal places max, trailing zeros stripped, integer results as `5.0` not `5`
- **Teacher validation**: Three tiers — `exact_match` > `answer_match` > `program_valid`; falls back to gold SFT data when teacher fails (`teacher_distill.py`)

## Kaggle Offline Deployment
```bash
# Upload pipeline code as Kaggle dataset
python scripts/kaggle_upload.py --upload-code

# Upload Python wheels for offline install
python scripts/kaggle_upload.py --upload-wheels

# Check upload status
python scripts/kaggle_upload.py --check
```
Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`. See `KAGGLE_DEPLOY.md` for full notebook setup with required input datasets (`thanhduc1108/vlsp2025-kd-pipeline`, model datasets, etc.).

## Evaluation
- **EA**: Numeric comparison with 0.01% tolerance (`evaluate.py → answers_match`)
- **PA**: Symbolic program equivalence via sympy — handles commutativity, e.g., `add(a,b) == add(b,a)` (`evaluate.py → programs_match`)
- PA is the **primary contest ranking metric**
