---
name: knowledge-distillation
description: "End-to-end Knowledge Distillation pipeline for Vietnamese financial numerical reasoning (VLSP 2025). Use when: training financial QA models, running KD pipeline, configuring GPU profiles, tuning PCPO rewards, debugging SFT/GRPO training, preparing ViNumQA/FinQA data, running majority voting inference, evaluating EA/PA metrics."
argument-hint: "Specify GPU profile (p100_16gb, a100_80gb, h100_80gb) and phase (data_prep, teacher, sft, grpo, inference, evaluate, all)"
---

# Knowledge Distillation for Financial Numerical Reasoning

## When to Use
- Training or fine-tuning models for Vietnamese financial numerical reasoning
- Running any phase of the KD pipeline (data prep, teacher distillation, SFT, GRPO, inference, evaluation)
- Switching GPU profiles or scaling to different hardware
- Debugging training issues (gradient checkpointing, LoRA, quantization)
- Evaluating model performance with EA/PA metrics
- Working with ViNumQA or FinQA datasets

## System Architecture

```
Data Prep → Teacher Distill → SFT Training → GRPO/PCPO → Inference + Vote → Evaluate
  (Phase 1)     (Phase 2)       (Phase 3)     (Phase 4)     (Phase 5)       (Phase 6)
```

### Phase 1: Data Preparation (`pipeline/data_prep.py`)
- Loads ViNumQA (Vietnamese) + FinQA (English) for multilingual training
- Augments with `program_re` field (alternative equivalent programs)
- Converts tables to Markdown format
- Outputs: SFT format, GRPO/Parquet format, teacher input format

### Phase 2: Teacher Distillation (`pipeline/teacher_distill.py`)
- Large teacher generates structured reasoning traces (Chain of Numerical Reasoning)
- Validates output matches gold program + answer
- Supports local model (quantized) or API-based teacher
- Output: distilled SFT training data with reasoning

### Phase 3: SFT Training (`pipeline/train_sft.py`)
- Fine-tunes student model on teacher traces
- Uses LoRA/QLoRA for memory efficiency on consumer GPUs
- Requires `model.enable_input_require_grads()` for gradient checkpointing + LoRA
- P100: Qwen3-0.6B FP16 + LoRA r=64 → ~1.4GB VRAM

### Phase 4: GRPO with PCPO Reward (`pipeline/train_grpo.py`)
- Program-Centric Policy Optimization reward function:
  ```
  R(p, x) = R_valid × (0.7 + 0.2 × R_exec + 0.1 × R_bonus)
  ```
  - R_valid: 0 if syntax error, 1 if valid (absolute gate)
  - R_exec: 1 if answer matches gold, 0 otherwise
  - R_bonus: 1.0 (shorter), 0.5 (equal), 0.1 (longer than gold)
- Uses TRL GRPOTrainer when available, manual REINFORCE fallback otherwise
- Generates N rollouts per prompt, normalizes rewards within group

### Phase 5: Inference + Majority Voting (`pipeline/inference.py`)
- Generates N candidate programs per question (temperature sampling)
- Executes each program with the DSL executor
- Majority vote on executed answers for final selection
- Returns confidence score based on agreement ratio

### Phase 6: Evaluation (`pipeline/evaluate.py`)
- **EA** (Execution Accuracy): Is the final numerical answer correct?
- **PA** (Program Accuracy): Is the reasoning program structurally correct?
- PA is the primary ranking metric (auditability > just getting the right answer)

## GPU Profiles

| Profile | Teacher | Student | Quantization | Max Seq | Candidates |
|---------|---------|---------|-------------|---------|------------|
| `p100_16gb` | Qwen3-1.7B | Qwen3-0.6B | 4bit teacher | 4096 | 5 |
| `t4_16gb` | Qwen3-4B | Qwen3-1.7B | 4bit both | 4096 | 5 |
| `a100_40gb` | Qwen3-14B | Qwen3-4B | 4bit teacher | 8192 | 10 |
| `a100_80gb` | Qwen3-32B | Qwen3-8B | 4bit teacher | 16384 | 15 |
| `h100_80gb` | Qwen3-235B-A22B | Qwen3-8B | None (API) | 32768 | 15 |

## Quick Start

```bash
# Setup
bash scripts/setup.sh --full

# Full pipeline on current GPU
python -m pipeline.run --gpu-profile p100_16gb --phases all

# Run specific phases
python -m pipeline.run --phases data_prep sft

# With custom config
python -m pipeline.run --gpu-profile a100_80gb --config configs/a100_80gb.yaml

# Dry run (print config only)
python -m pipeline.run --gpu-profile p100_16gb --dry-run
```

## Key Technical Decisions

1. **Multilingual not translated**: Use FinQA English directly alongside ViNumQA Vietnamese (HUSTUET approach). LLMs learn math in a shared representation space.
2. **Program-centric reward**: Weight 0.7 for valid syntax >> 0.2 for correct answer. Financial auditability requires correct reasoning, not just correct numbers.
3. **LoRA over full fine-tune on constrained GPUs**: Qwen3-0.6B with LoRA r=64 uses only 1.4GB VRAM. Scale by switching profile.
4. **P100 compatibility**: No BF16, no flash-attention-2. Always use FP16 + standard attention on Pascal GPUs.

## Program DSL

The financial reasoning DSL supports these operations:
- Binary: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`
- Table: `table_sum`, `table_average`, `table_max`, `table_min`
- Step references: `#0`, `#1`, etc.
- Constants: `const_100`, `const_1000`, `const_m1`, etc.

Example: `subtract(1500, 1000), divide(#0, 1000), multiply(#1, const_100)` → 50.0

## Troubleshooting

- **Gradient error with LoRA + checkpointing**: Call `model.enable_input_require_grads()` before training
- **OOM on P100**: Reduce `max_seq_length` to 2048 or use 4-bit quantization for student
- **TRL not installed**: Pipeline automatically falls back to manual GRPO implementation
- **Low PA after GRPO**: Increase `alpha` weight (validity) relative to `beta` (execution)
