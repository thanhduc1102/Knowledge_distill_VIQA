# VLSP 2025 Financial Numerical Reasoning

Knowledge Distillation pipeline for Vietnamese financial numerical reasoning (VLSP 2025 challenge). Includes baseline zero-shot inference, SFT training, GRPO/PCPO optimization, and majority voting.

## Quick Start

### 1. Download data
```bash
bash scripts/download_data.sh
```

### 2. Run baseline on current GPU
```bash
# P100 16GB (pipeline verification)
python -m baseline.run_baseline --gpu-profile p100_16gb --models qwen3-0.6b --max-samples 5

# RTX 6000 Pro 96GB (full baseline)
python -m baseline.run_baseline --gpu-profile rtx6000_96gb --models all
```

### 3. Run KD training pipeline
```bash
python -m pipeline.run --gpu-profile p100_16gb --phases all
```

## Repository Structure

```
.
├── baseline/                # Zero-shot baseline inference
│   ├── config.py           # Model configs per GPU profile
│   └── run_baseline.py     # Main baseline runner
├── pipeline/               # Knowledge Distillation pipeline
│   ├── config.py           # Pipeline configuration system
│   ├── data_prep.py        # Data preparation (ViNumQA + FinQA)
│   ├── teacher_distill.py  # Teacher model reasoning traces
│   ├── train_sft.py        # Supervised Fine-Tuning
│   ├── train_grpo.py       # GRPO with PCPO reward
│   ├── inference.py        # Multi-path inference + majority voting
│   ├── evaluate.py         # EA/PA evaluation
│   ├── program_executor.py # Financial DSL executor
│   ├── reward.py           # PCPO reward function
│   └── run.py              # Pipeline orchestrator
├── configs/                # GPU-specific YAML configs
│   ├── p100_16gb.yaml
│   ├── rtx6000pro_96gb.yaml
│   ├── a100_80gb.yaml
│   └── h100_80gb.yaml
├── kaggle/                 # Kaggle notebook files
│   └── kaggle_baseline_notebook.py
├── scripts/
│   ├── download_data.sh    # Download ViNumQA + FinQA
│   ├── prepare_kaggle_offline.sh  # Build offline package
│   ├── setup.sh            # Install dependencies
│   └── run_pipeline.sh     # Run pipeline wrapper
├── src/                    # Original source utilities
│   ├── assets/template.py  # Prompt template
│   └── program_tokenizer.py
├── data/receive/           # ViNumQA dataset
├── KAGGLE_DEPLOY.md        # Detailed Kaggle deployment guide
└── requirements.txt
```

## Baseline Models (RTX 6000 Pro 96GB)

| Model | Quantization | Est. VRAM | Candidates |
|-------|-------------|-----------|------------|
| Qwen/Qwen3.5-4B | BF16 | ~9 GB | 15 |
| Qwen/Qwen3.5-9B | BF16 | ~19 GB | 15 |
| Qwen/Qwen3.5-27B | BF16 | ~55 GB | 10 |
| Qwen/Qwen3.5-35B-A3B | BF16 | ~72 GB | 15 |
| Qwen/Qwen3.5-122B-A10B | 4-bit NF4 | ~70 GB | 10 |

## Kaggle Deployment (Offline)

See [KAGGLE_DEPLOY.md](KAGGLE_DEPLOY.md) for step-by-step guide to run on Kaggle RTX 6000 Pro 96GB without internet.

## Dataset

- **ViNumQA**: 2993 train / 584 valid / 497 public test / 1625 private test
- **FinQA**: English financial QA (used for multilingual augmentation)

## Evaluation Metrics

- **EA** (Execution Accuracy): Is the numerical answer correct?
- **PA** (Program Accuracy): Is the reasoning program structurally correct?
- PA is the primary ranking metric (auditability > correct result)