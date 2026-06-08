# GitHub Copilot Instructions — VLSP 2025 KD + CLEF 2026 FinMMEval

## Workspace Overview
This repo contains **two independent competition solutions** that share hardware profiles but otherwise do not import from each other:

| Sub-project | Root dir | Competition | Primary metric |
|---|---|---|---|
| VLSP 2025 KD | `pipeline/` + `src/` | Vietnamese financial numerical reasoning | PA (Program Accuracy) |
| CLEF 2026 FinMMEval | `clef2026_finmmeval/` | Multilingual financial AI (Tasks 1-3) | Task1 Acc / Task2 ROUGE-1 / Task3 CR |

---

## VLSP 2025 — Knowledge Distillation Pipeline

### Architecture (6 Phases)
```
data_prep → teacher → sft → grpo → inference → evaluate
```
- **`pipeline/config.py`** — All config via `PipelineConfig` dataclass + `GPU_PROFILES` dict; never hard-code VRAM settings
- **`pipeline/program_executor.py`** — Financial DSL: `add/subtract/multiply/divide/exp/greater/table_sum/table_average/table_max/table_min`
- **`pipeline/reward.py`** — PCPO reward: `R = R_valid × (0.7 + 0.2×R_exec + 0.1×R_bonus)`
- **`src/assets/template.py`** — Vietnamese prompt template (all three output sections must be in Vietnamese)

### Running
```bash
python -m pipeline.run --gpu-profile p100_16gb --phases all --max-samples 50   # test
python -m pipeline.run --gpu-profile rtx6000_96gb --phases all                 # production
# Resumes automatically from data/pipeline/pipeline_state.json if interrupted
```

### GPU Profiles (defined in `pipeline/config.py → GPU_PROFILES`)
| Profile | Teacher | Student | Flash Attn | dtype |
|---|---|---|---|---|
| `p100_16gb` | Qwen3.5-4B (4bit) | Qwen3.5-0.8B | ❌ | fp16 |
| `rtx6000_96gb` | Qwen3.5-27B | Qwen3.5-4B | ✅ | bf16 |

Flash Attention 2: `pip install flash-attn --no-build-isolation` — RTX/A100/H100 only.

### Output Format (parsed by regex in `reward.py`, `teacher_distill.py`, `evaluate.py`)
```
**Phân tích lập luận:**
<reasoning>
**Chương trình tính toán:**
divide(914, 391), multiply(#0, const_100)
**Đáp án cuối cùng:**
30.28486
```
Back-references: `#0`, `#1`, … — Constants: `const_100`, `const_1000`, `const_m1`.

### Key Conventions
- **Table handling**: 2D lists → Markdown via `data_prep.table_to_markdown()` before prompt insertion
- **Program augmentation**: `use_program_re=True` boosts training data from ~11K → 14.6K samples
- **Vietnamese text**: `encoding="utf-8"` everywhere; `json.dump(..., ensure_ascii=False)`
- **Answer format**: 5 decimal places max, trailing zeros stripped; integers as `5.0` not `5`
- **Teacher validation tiers**: `exact_match` > `answer_match` > `program_valid`; falls back to gold data on failure
- **PA vs EA**: PA (sympy symbolic, handles commutativity) is the contest metric; EA is secondary

### Kaggle Deployment
```bash
python scripts/kaggle_upload.py --upload-code    # upload pipeline as dataset
python scripts/kaggle_upload.py --upload-wheels  # offline Python wheels
```
Requires `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`. See `KAGGLE_DEPLOY.md`.

---

## CLEF 2026 FinMMEval (`clef2026_finmmeval/`)

### Five Tiers — Never Mix Imports Between Them
| Tier | Entry point | Model | Target GPU |
|---|---|---|---|
| `src/` | `scripts/run_all_tasks.py` | Qwen3.5-4B | P100 16GB |
| `baseline/` | `baseline/run_baseline.py` | Qwen2.5-7B | P100/T4 |
| `advanced/` | `advanced/run_advanced.py` | Qwen2.5-7B | RTX 48GB |
| `pro/` | `pro/run_pro.py` | Qwen2.5-72B | RTX 96GB |
| `pipeline/` | `pipeline/` modules | Qwen3.5-27B→4B | RTX 96GB |

`baseline/runner.py` and `advanced/runner.py` are **fully self-contained** (no `src/` imports). `pro/` imports from `src/`.

### Running (from inside `clef2026_finmmeval/`)
```bash
python scripts/run_all_tasks.py --tasks 1 --max-samples 10   # src/ tier quick test
python baseline/run_baseline.py --tasks 1 2 3
python advanced/run_advanced.py --tasks 1 2 3
python pro/run_pro.py --tasks 1 2 3
# Arabic SOTA (SFT+GRPO+BM25+self-consistency)
python scripts/run_task1_arabic_sota.py --max-samples 5 --no-train
# Task 3 API (port 62237)
python -m api.trading_api
```

### Critical Qwen3.5 / P100 Gotchas
- **`enable_thinking=False`** in `apply_chat_template()` is REQUIRED — prevents 500+ token internal CoT
- **`max_context_length: 4096`** in Task 2 config is REQUIRED — longer sequences hang on P100
- **`transformers >= 5.4.0`** is REQUIRED — older versions lack `qwen3_5` in CONFIG_MAPPING
- **CFA/CPA datasets**: loaded via `hf_hub_download` + parquet, NOT `load_dataset` (HF repo schema bug)
- **BBF (Hindi)** removed — gated dataset requiring HF auth

### Key Architecture Patterns
- **Prompts as constants** in `src/prompts.py` (`TASK1_COT_TEMPLATE`, etc.) — add strategy: (1) add constant, (2) add branch in `format_task*_prompt()`, (3) set `prompting.strategy` in YAML
- **Task 3 config nesting**: `model.llm.name` (not `model.name`) because `model.sentiment` also exists for FinBERT — always check `config["model"].get("llm")` first
- **Task 3 SOTA is rule-based** (no LLM): `compute_rule_based_signal()` uses SMA-10 vs SMA-20 with asymmetric thresholds (crypto: buy≥0.30; stocks: buy≥0.02)
- **Task 2 BM25 RAG**: `src/task2_rag_chunking.py → chunk_financial_context()` — requires `pip install rank-bm25`; falls back to naive truncation if missing
- **Answer constraints**: Task 2 answers ≤100 words — `truncate_answer()` in `task2_financial_qa.py` uses sentence-boundary-aware truncation
- **Fallback defaults**: `"A"` for MCQ, `"HOLD"` for trading, simple ROUGE for Task 2
