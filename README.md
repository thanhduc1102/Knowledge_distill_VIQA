# VLSP 2025 - Verifier-Native Financial Reasoning Suite

Verifier-native financial numerical reasoning pipeline centered on executable programs, symbolic evaluation, and benchmark-suite comparison. The repo now supports a shared reasoning stack over FinQA, TAT-QA, ConvFinQA, DocMath-Eval, and FinChain, with ViNumQA kept as an external portability probe by default.

## Quick Start

### 1. Install dependencies
```bash
pip install transformers>=5.0 peft>=0.18 accelerate datasets bitsandbytes trl sympy pyyaml
```

### 2. Run pipeline on P100 (testing)
```bash
# Run the main benchmark suite with small samples
python -m pipeline.benchmark_suite --gpu-profile p100_16gb --max-samples 50

# Legacy single-run pipeline entrypoint
python -m pipeline.run --gpu-profile p100_16gb --phases data_prep teacher sft
```

### 3. Run on RTX 6000 Pro 96GB (production)
```bash
python -m pipeline.run --gpu-profile rtx6000_96gb --phases all
```

## Architecture

```
Data Prep → Teacher Distill → SFT → GRPO/PCPO or GRPO/ECRL → Inference → Evaluate
  (suite)      (Qwen3.5-27B)   (LoRA)       (reward)          (per benchmark)
```

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed methodology, formulas, and analysis.

For the AAAI-27 research blueprint centered on verifier-native financial reasoning, benchmark strategy, ablations, and implementation roadmap, see [docs/AAAI27_RESEARCH_BLUEPRINT.md](docs/AAAI27_RESEARCH_BLUEPRINT.md). For the 2024-2026 SOTA survey, benchmark selection, baseline matrix, and frontier API-model gap analysis, see [docs/AAAI27_SOTA_SURVEY_AND_BENCHMARKS.md](docs/AAAI27_SOTA_SURVEY_AND_BENCHMARKS.md). For the positioning reset and the detailed core research strategy, see [docs/AAAI27_CORE_RESEARCH_DIRECTION_AND_STRATEGY.md](docs/AAAI27_CORE_RESEARCH_DIRECTION_AND_STRATEGY.md).

## Repository Structure

```
.
├── pipeline/                    # Core KD pipeline (6 phases)
│   ├── config.py               # Configuration + GPU profiles
│   ├── benchmarks.py           # Benchmark registry + loaders
│   ├── benchmark_suite.py      # SFT vs GRPO-PCPO vs GRPO-ECRL suite runner
│   ├── run.py                  # Pipeline orchestrator
│   ├── data_prep.py            # Data loading, formatting, augmentation
│   ├── teacher_distill.py      # Teacher reasoning trace generation
│   ├── train_sft.py            # Supervised Fine-Tuning with LoRA
│   ├── train_grpo.py           # GRPO with PCPO or ECRL-Fin rewards
│   ├── inference.py            # Multi-path inference + majority voting
│   ├── evaluate.py             # Answer / program / step evaluation
│   ├── program_executor.py     # Financial DSL executor
│   └── reward.py               # PCPO + ECRL-Fin reward functions
├── src/                         # Utilities
│   ├── assets/template.py      # Vietnamese prompt template
│   └── program_tokenizer.py    # Program tokenization
├── configs/                     # GPU-specific YAML configs
│   ├── p100_16gb.yaml
│   └── rtx6000pro_96gb.yaml
├── kaggle/                      # Kaggle notebook files
│   ├── kaggle_kd_notebook.py   # Full KD pipeline notebook
│   └── kaggle_baseline_notebook.py
├── scripts/
│   ├── build_benchmark_cache.py # Mirror public benchmarks for offline Kaggle use
│   ├── kaggle_upload.py        # Upload to Kaggle via API
│   ├── download_data.sh        # Download datasets
│   └── setup.sh                # Install dependencies
├── dataset/                     # Datasets
│   ├── viNumericalQA_private/  # Vietnamese financial QA
│   ├── dataset_finqa_en/       # FinQA benchmark anchor
│   └── benchmark_cache/        # Offline mirrors for public benchmarks
├── TECHNICAL_REPORT.md         # Detailed technical documentation
└── requirements.txt
```

## GPU Profiles

| Profile | GPU | Teacher | Student | VRAM Usage |
|---------|-----|---------|---------|------------|
| `p100_16gb` | Tesla P100 16GB | Qwen3.5-4B (4bit) | Qwen3.5-0.8B | ~12 GB |
| `rtx6000_96gb` | RTX 6000 Pro 96GB | Qwen3.5-27B | Qwen3.5-4B | ~70 GB |
| `rtx6000_96gb_35b` | RTX 6000 Pro 96GB | Qwen3.5-35B-A3B | Qwen3.5-4B | ~80 GB |
| `rtx6000_96gb_122b` | RTX 6000 Pro 96GB | Qwen3.5-122B-A10B (4bit) | Qwen3.5-9B | ~90 GB |

## Benchmarks

- **FinQA**: core program-supervised benchmark and default train anchor
- **TAT-QA**: answer-only benchmark with table and paragraph evidence
- **ConvFinQA**: conversational financial QA benchmark, currently treated as answer-only in the public mirror
- **DocMath-Eval**: optional gated benchmark, loaded when a local mirror is available
- **FinChain**: optional local-first chain/step benchmark
- **ViNumQA**: external portability probe; excluded from default training

## Evaluation Metrics

- **Answer Accuracy**: answer match with numeric tolerance and list-aware normalization
- **Program Accuracy (PA)**: symbolic program equivalence via sympy / structural fallback
- **Step Accuracy**: intermediate-step agreement for step-supervised benchmarks such as FinChain
- **Valid Program Rate**: fraction of generations that compile under the financial DSL

## Kaggle Deployment (Offline)

### Step 1: Upload assets
```bash
# Build public benchmark cache for offline execution
python scripts/build_benchmark_cache.py --benchmarks tatqa convfinqa

# Upload pipeline code, wheels, bundled benchmarks, and push the notebook
python scripts/kaggle_upload.py --all --push-notebook
```

### Step 2: Kaggle notebook setup
1. Create notebook → Select GPU RTX 6000 Pro → Internet OFF
2. Add input datasets:
   - `thanhduc1108/vlsp2025-kd-pipeline`
   - `thanhduc1108/vlsp2025-kd-wheels`
  - `thanhduc1108/financial-reasoning-benchmarks`
   - `thanhduc1108/qwen_35_27b`
   - `thanhduc1108/qwen_35_4b`
3. Copy and run `kaggle/kaggle_kd_notebook.py`

### Step 3: Check outputs
Results saved to `/kaggle/working/vlsp2025/outputs/`:
- `benchmark_suite/` - per-variant, per-benchmark predictions and metrics
- `suite_results.json` - aggregated variant summary
- `checkpoints_benchmark_suite/` - LoRA adapters / checkpoints for the suite run
- `artifact_summary.json` - run metadata and output pointers

## Key Technical Features

1. **PCPO Reward**: $R = R_{valid} \times (0.7 + 0.2R_{exec} + 0.1R_{bonus})$ prioritizes valid executable reasoning.
2. **ECRL-Fin Reward**: set `grpo.reward_type: ecrl` to reward syntax validity, execution, symbolic equivalence, intermediate steps, answer match, and brevity.
3. **Benchmark-aware data prep**: training and evaluation are routed through a benchmark registry rather than hardcoded dataset paths.
4. **Program/answer/step evaluation**: the evaluator handles mixed supervision families in one suite summary.
5. **Offline Kaggle execution**: public benchmarks can be mirrored into `dataset/benchmark_cache/` and shipped as a single benchmark bundle.
6. **LoRA-based training**: memory-efficient SFT and GRPO remain the default adaptation path.
