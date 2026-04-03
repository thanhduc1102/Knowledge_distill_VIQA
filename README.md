# VLSP 2025 - Knowledge Distillation for Vietnamese Financial Numerical Reasoning

Knowledge Distillation pipeline for Vietnamese financial numerical reasoning (VLSP 2025 challenge). Implements the full HUSTUET-inspired pipeline: multilingual data integration, teacher reasoning distillation, SFT, GRPO/PCPO optimization, and majority voting inference.

## Quick Start

### 1. Install dependencies
```bash
pip install transformers>=5.0 peft>=0.18 accelerate datasets bitsandbytes trl sympy pyyaml
```

### 2. Run pipeline on P100 (testing)
```bash
# Full pipeline with small sample
python -m pipeline.run --gpu-profile p100_16gb --phases all --max-samples 50

# Run specific phases
python -m pipeline.run --gpu-profile p100_16gb --phases data_prep teacher sft
```

### 3. Run on RTX 6000 Pro 96GB (production)
```bash
python -m pipeline.run --gpu-profile rtx6000_96gb --phases all
```

## Architecture

```
Data Prep → Teacher Distill → SFT → GRPO/PCPO → Inference → Evaluate
  (14.6K)     (Qwen3.5-27B)   (LoRA)  (Reward)   (N-path)    (EA+PA)
```

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed methodology, formulas, and analysis.

## Repository Structure

```
.
├── pipeline/                    # Core KD pipeline (6 phases)
│   ├── config.py               # Configuration + GPU profiles
│   ├── run.py                  # Pipeline orchestrator
│   ├── data_prep.py            # Data loading, formatting, augmentation
│   ├── teacher_distill.py      # Teacher reasoning trace generation
│   ├── train_sft.py            # Supervised Fine-Tuning with LoRA
│   ├── train_grpo.py           # GRPO with PCPO reward function
│   ├── inference.py            # Multi-path inference + majority voting
│   ├── evaluate.py             # EA/PA evaluation (sympy symbolic comparison)
│   ├── program_executor.py     # Financial DSL executor
│   └── reward.py               # PCPO reward function
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
│   ├── kaggle_upload.py        # Upload to Kaggle via API
│   ├── download_data.sh        # Download datasets
│   └── setup.sh                # Install dependencies
├── dataset/                     # Datasets
│   ├── viNumericalQA_private/  # Vietnamese financial QA
│   └── finqa_en/               # English financial QA
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

## Dataset

- **ViNumQA**: 2993 train / 584 valid / 497 test / 1625 private test
- **FinQA**: 6251 train + 883 dev + 1147 test (English, used for multilingual augmentation)
- **Total SFT training**: 14,661 samples (with program_re augmentation)

## Evaluation Metrics

- **EA** (Execution Accuracy): Numerical answer correctness (tolerance: 0.01%)
- **PA** (Program Accuracy): Symbolic program equivalence via sympy (primary ranking metric)

## Kaggle Deployment (Offline)

### Step 1: Upload assets
```bash
# Upload pipeline code to Kaggle
python scripts/kaggle_upload.py --upload-code

# Upload Python wheels for offline install
python scripts/kaggle_upload.py --upload-wheels
```

### Step 2: Kaggle notebook setup
1. Create notebook → Select GPU RTX 6000 Pro → Internet OFF
2. Add input datasets:
   - `thanhduc1108/vlsp2025-kd-pipeline`
   - `thanhduc1108/vlsp2025-kd-wheels`
   - `thanhduc1108/finqa-en`
   - `thanhduc1108/vinumericalqa-private`
   - `thanhduc1108/qwen_35_27b`
   - `thanhduc1108/qwen_35_4b`
3. Copy and run `kaggle/kaggle_kd_notebook.py`

### Step 3: Check outputs
Results saved to `/kaggle/working/vlsp2025/outputs/`:
- `final_model/` - Distilled student model
- `eval_results.json` - EA/PA scores
- `predictions.json` - All predictions
- `summary.json` - Comparison with baseline

## Key Technical Features

1. **PCPO Reward**: R = R_valid * (0.7 + 0.2 * R_exec + 0.1 * R_bonus) - prioritizes program validity over correct answer
2. **Multilingual Data**: English FinQA + Vietnamese ViNumQA without translation
3. **Symbolic PA**: sympy-based program equivalence (handles commutativity, etc.)
4. **Multi-level teacher validation**: exact match + answer match + valid-only
5. **LoRA**: Memory-efficient fine-tuning (5-8% trainable parameters)
