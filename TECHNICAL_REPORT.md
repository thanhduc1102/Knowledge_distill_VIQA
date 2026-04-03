# Technical Report: Knowledge Distillation for Vietnamese Financial Numerical Reasoning

## VLSP 2025 Challenge - Hybrid Pipeline Architecture

---

## 1. Overview

This system implements a **Knowledge Distillation (KD) pipeline** for the VLSP 2025 Financial Numerical Reasoning challenge. The goal is to train a compact student model (Qwen3.5-4B) that can accurately generate structured reasoning programs from financial data (tables + text) in Vietnamese, by learning from a larger teacher model (Qwen3.5-27B).

### 1.1 Problem Definition

**Input**: Financial context (pre_text, table, post_text) + Question in Vietnamese  
**Output**: 
- Computation program: sequence of DSL operations (e.g., `divide(914, 391), multiply(#0, const_100)`)
- Numerical answer: final computed result

**Evaluation Metrics**:
- **EA (Execution Accuracy)**: Is the numerical answer correct?
- **PA (Program Accuracy)**: Is the reasoning program symbolically equivalent to the gold program? (Primary metric)

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISTILLATION PIPELINE              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: DATA PREPARATION                                     │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ ViNumQA (Vi) │  │ FinQA (En)   │  ← Multilingual Data       │
│  │ 2993 train   │  │ 8281 samples │                             │
│  └──────┬───────┘  └──────┬───────┘                             │
│         └──────┬──────────┘                                     │
│                ▼                                                │
│  ┌─────────────────────────────┐                                │
│  │ Merge + program_re augment  │  → 14,661 training samples    │
│  │ Table → Markdown conversion │                                │
│  └──────────┬──────────────────┘                                │
│             ▼                                                   │
│  Phase 2: TEACHER DISTILLATION                                  │
│  ┌─────────────────────────────┐                                │
│  │ Qwen3.5-27B (Teacher)      │                                │
│  │ Chain of Numerical Reason.  │  → Structured reasoning traces │
│  │ Multi-retry validation      │                                │
│  └──────────┬──────────────────┘                                │
│             ▼                                                   │
│  Phase 3: SUPERVISED FINE-TUNING (SFT)                          │
│  ┌─────────────────────────────┐                                │
│  │ Qwen3.5-4B (Student)       │                                │
│  │ LoRA r=128, α=256          │                                │
│  │ Train on teacher traces     │  → SFT checkpoint             │
│  └──────────┬──────────────────┘                                │
│             ▼                                                   │
│  Phase 4: GRPO + PCPO REWARD                                   │
│  ┌─────────────────────────────┐                                │
│  │ Group Relative Policy Opt.  │                                │
│  │ R = R_valid(α + β·R_exec   │  → GRPO checkpoint            │
│  │           + γ·R_bonus)      │                                │
│  └──────────┬──────────────────┘                                │
│             ▼                                                   │
│  Phase 5: MULTI-PATH INFERENCE                                  │
│  ┌─────────────────────────────┐                                │
│  │ N candidate programs        │                                │
│  │ DSL execution               │  → Majority voting → Answer   │
│  │ Majority voting             │                                │
│  └──────────┬──────────────────┘                                │
│             ▼                                                   │
│  Phase 6: EVALUATION                                            │
│  ┌─────────────────────────────┐                                │
│  │ EA (Execution Accuracy)     │                                │
│  │ PA (Program Accuracy)       │  → Symbolic comparison (sympy) │
│  └─────────────────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Methodology

### 2.1 Phase 1: Multilingual Data Integration

**Rationale**: Following the HUSTUET winning approach, we use the original English FinQA data directly (without translation) mixed with Vietnamese ViNumQA data. This exploits the multilingual representation space of LLMs to learn "mathematical logic" independent of language.

**Data Augmentation with `program_re`**: FinQA includes alternative programs (`program_re`) that produce the same answer through different computation paths. We duplicate training samples using these alternatives, teaching the model mathematical equivalence:

$$
\text{If } f(x) = g(x) \text{ for program } f \text{ and } g, \text{ then both } (x, f) \text{ and } (x, g) \text{ are valid training pairs.}
$$

**Table → Markdown Conversion**: Raw 2D table arrays are converted to Markdown format for better LLM comprehension:

```
| Header1 | Header2 | Header3 |
| ------- | ------- | ------- |
| Data1   | Data2   | Data3   |
```

**Output**:
- 14,661 SFT training samples (2,993 Vi + 8,281 En + 3,387 program_re augmented)
- 584 validation samples
- 11,274 teacher distillation inputs

### 2.2 Phase 2: Knowledge Distillation via Reasoning Traces

**Teacher Model**: Qwen3.5-27B (or larger: 35B-A3B, 122B-A10B)

The teacher generates **structured reasoning traces** following the Chain of Numerical Reasoning (CoNR) format:

```
**Phan tich lap luan:**
1. Hieu cau hoi: [Goal identification]
2. Xac dinh du lieu: [Data extraction from table/text]
3. Lap luan logic: [Step-by-step logic]
4. Lien ket ham: [Function mapping]

**Chuong trinh tinh toan:**
divide(914, 391), multiply(#0, const_100)

**Dap an cuoi cung:**
233.75832
```

**Multi-level Validation Strategy**:
1. **Exact match** (highest quality): Predicted program exactly matches gold program
2. **Answer match**: Program is syntactically valid and produces correct numerical answer
3. **Invalid**: Program fails validation

This graduated validation is more flexible than strict exact-match-only approaches, capturing cases where the teacher finds valid alternative solutions.

### 2.3 Phase 3: Supervised Fine-Tuning (SFT)

**Student Model**: Qwen3.5-4B with LoRA

**LoRA Configuration**:
- Rank $r = 128$, Alpha $\alpha = 256$
- Target: all linear layers
- Trainable parameters: ~5-8% of total
- Dropout: 0.05

**Training Details**:
- Loss computed only on assistant response tokens (user prompt tokens masked with label=-100)
- Gradient checkpointing enabled for memory efficiency
- Cosine learning rate schedule: $\eta_t = \eta_{max} \cdot \frac{1 + \cos(\pi t / T)}{2}$

$$
\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)
$$

where $x$ is the input prompt and $y$ is the teacher's reasoning trace.

### 2.4 Phase 4: GRPO with PCPO Reward

**Algorithm**: Group Relative Policy Optimization (GRPO)

GRPO generates $N$ completions per prompt, computes rewards, normalizes within the group, and updates the policy using REINFORCE:

$$
\nabla \mathcal{L}_{GRPO} = -\mathbb{E}_{x \sim D} \left[ \frac{1}{N} \sum_{i=1}^{N} A_i \cdot \nabla \log P_\theta(y_i | x) \right]
$$

where the advantage $A_i$ is computed by normalizing rewards within each group:

$$
A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}
$$

**PCPO Reward Function** (Program-Centric Policy Optimization):

$$
R(p, x) = R_{valid} \cdot (\alpha + \beta \cdot R_{exec}(p, x) + \gamma \cdot R_{bonus})
$$

| Component | Value | Description |
|-----------|-------|-------------|
| $R_{valid}$ | $\in \{0, 1\}$ | 1 if program is syntactically valid, 0 otherwise (gating) |
| $R_{exec}$ | $\in \{0, 1\}$ | 1 if executed answer matches gold answer |
| $R_{bonus}$ | $\in \{0.1, 0.5, 1.0\}$ | Conciseness: 1.0 if shorter, 0.5 if equal, 0.1 if longer than gold |
| $\alpha$ | 0.7 | Base reward weight for valid programs |
| $\beta$ | 0.2 | Execution accuracy weight |
| $\gamma$ | 0.1 | Conciseness bonus weight |

**Design Rationale**: The reward function prioritizes:
1. **Program validity** ($\alpha = 0.7$): A syntactically invalid program gets R=0 regardless of other factors
2. **Execution correctness** ($\beta = 0.2$): Correct numerical answer is rewarded but not dominant
3. **Code conciseness** ($\gamma = 0.1$): Shorter programs are preferred for auditability

This aligns with the financial domain requirement where **auditability > correct result**: a correct answer with incorrect reasoning has zero value in finance.

### 2.5 Phase 5: Multi-Path Inference with Majority Voting

Generate $N$ candidate programs per question (N=15 for RTX 6000):

1. **Sampling**: Temperature=0.7, top_p=0.95 for diverse candidates
2. **Extraction**: Regex-based program and answer extraction
3. **Validation**: Syntax check via DSL validator
4. **Execution**: Run valid programs against table data
5. **Voting**: Count answer frequencies, select most common

$$
\hat{a} = \arg\max_{a} \sum_{i=1}^{N} \mathbb{1}[\text{exec}(p_i) = a]
$$

**Confidence Score**: $\text{conf} = \frac{\text{count}(\hat{a})}{N}$

### 2.6 Phase 6: Evaluation

**EA (Execution Accuracy)**:
$$
EA = \frac{|\{i : |\hat{a}_i - a_i^*| / \max(|a_i^*|, 10^{-10}) < 10^{-4}\}|}{|D_{test}|}
$$

For zero gold answer: absolute tolerance $< 10^{-5}$.

**PA (Program Accuracy)** - Using symbolic comparison from FinQA:

1. Tokenize both programs into structured token lists
2. Build symbolic expression trees using symbol substitution
3. Simplify both expressions with **sympy**
4. Compare simplified symbolic forms

Example of symbolic equivalence:
```
Gold: add(A, B), multiply(#0, C)  →  (a0 + a1) * a2
Pred: add(B, A), multiply(#0, C)  →  (a1 + a0) * a2  
sympy.simplify: (a0 + a1) * a2 == (a1 + a0) * a2  →  TRUE (commutative)
```

---

## 3. Financial DSL (Domain-Specific Language)

### 3.1 Operation Set

| Operation | Arguments | Output | Formula |
|-----------|-----------|--------|---------|
| `add` | num1, num2 | number | $num1 + num2$ |
| `subtract` | num1, num2 | number | $num1 - num2$ |
| `multiply` | num1, num2 | number | $num1 \times num2$ |
| `divide` | num1, num2 | number | $num1 / num2$ |
| `exp` | num1, num2 | number | $num1^{num2}$ |
| `greater` | num1, num2 | bool | $num1 > num2$ → yes/no |
| `table_sum` | row_header, none | number | $\sum_{j} T[row][j]$ |
| `table_average` | row_header, none | number | $\frac{1}{n}\sum_{j} T[row][j]$ |
| `table_max` | row_header, none | number | $\max_j T[row][j]$ |
| `table_min` | row_header, none | number | $\min_j T[row][j]$ |

### 3.2 Constants and References

- `const_100`, `const_1000`, `const_1000000`: Numeric constants
- `const_1`, `const_2`, ..., `const_12`: Small integers
- `const_m1`: Negative one (-1)
- `#0`, `#1`, ...: Reference to result of step 0, 1, etc.

### 3.3 Example Program

Question: "What is the profit margin percentage?"  
Table: Revenue=914, Profit=391

```
divide(391, 914), multiply(#0, const_100)
```

Step 0: divide(391, 914) = 0.42779...  
Step 1: multiply(#0, const_100) = multiply(0.42779, 100) = 42.77897

---

## 4. Hardware Configurations

| Profile | GPU | VRAM | Teacher | Student | Quant |
|---------|-----|------|---------|---------|-------|
| p100_16gb | Tesla P100 | 16 GB | Qwen3.5-4B (4bit) | Qwen3.5-0.8B | FP16 |
| rtx6000_96gb | RTX 6000 Pro | 96 GB | Qwen3.5-27B | Qwen3.5-4B | BF16 |
| rtx6000_96gb_35b | RTX 6000 Pro | 96 GB | Qwen3.5-35B-A3B | Qwen3.5-4B | BF16 |
| rtx6000_96gb_122b | RTX 6000 Pro | 96 GB | Qwen3.5-122B-A10B (4bit) | Qwen3.5-9B | BF16 |
| a100_80gb | A100 | 80 GB | Qwen3.5-27B | Qwen3.5-9B | BF16 |

### VRAM Estimation

| Model | FP16 | 4bit NF4 | LoRA r=128 overhead |
|-------|------|----------|---------------------|
| Qwen3.5-0.8B | ~1.6 GB | ~0.5 GB | +0.2 GB |
| Qwen3.5-4B | ~8 GB | ~2.5 GB | +0.8 GB |
| Qwen3.5-9B | ~18 GB | ~6 GB | +1.5 GB |
| Qwen3.5-27B | ~54 GB | ~16 GB | +4 GB |
| Qwen3.5-35B-A3B | ~70 GB | ~22 GB | +5 GB |
| Qwen3.5-122B-A10B | ~244 GB | ~70 GB | +15 GB |

---

## 5. Dataset Statistics

### ViNumQA (Vietnamese Financial QA)
| Split | Samples | Usage |
|-------|---------|-------|
| Train | 2,993 | KD training |
| Valid | 584 | Validation |
| Test | 497 | Public evaluation |
| Private Test | 1,625 | Final ranking |

### FinQA (English Financial QA)
| Split | Samples | Usage |
|-------|---------|-------|
| Train | 6,251 | Multilingual augmentation |
| Dev | 883 | Multilingual augmentation |
| Test | 1,147 | Multilingual augmentation |

### Combined Training Data
| Component | Samples |
|-----------|---------|
| ViNumQA train | 2,993 |
| FinQA all splits | 8,281 |
| program_re augmented | +3,387 |
| **Total SFT train** | **14,661** |

---

## 6. Key Technical Innovations

### 6.1 Cross-lingual Knowledge Transfer Without Translation

Instead of translating English FinQA to Vietnamese (risking semantic drift), we mix both languages directly. LLMs can learn mathematical reasoning in a shared multilingual representation space. This approach:
- Eliminates translation noise
- Doubles available training data
- Proved to increase PA from 66.23% to 73.84% (HUSTUET ablation study)

### 6.2 Program-Centric Reward Design

The PCPO reward function fundamentally differs from standard RL in NLP:
- Standard RL: rewards correct final answer
- PCPO: rewards valid, correct, and concise **programs**

The gating mechanism ($R_{valid}$) ensures that any syntactically invalid program receives zero reward, regardless of whether it accidentally produces a correct answer. This enforces auditability.

### 6.3 Multi-Level Teacher Validation

Unlike strict exact-match validation, our system accepts teacher outputs at multiple quality levels:
- Exact program match (highest quality traces)
- Correct answer with valid alternative program (still valuable for learning)

This significantly increases the teacher distillation yield, especially for complex multi-step problems.

### 6.4 Symbolic Program Equivalence

The PA metric uses sympy-based symbolic comparison rather than string matching. This correctly identifies mathematically equivalent programs:
- `add(A, B)` ≡ `add(B, A)` (commutativity)
- `multiply(A, add(B, C))` ≡ `multiply(A, B), add(#0, multiply(A, C))` (distributivity)

---

## 7. Deployment Guide

### 7.1 Prerequisites on Kaggle

1. Upload datasets to Kaggle:
   - `thanhduc1108/vinumericalqa-private` (ViNumQA dataset)
   - `thanhduc1108/finqa-en` (FinQA dataset)
   
2. Upload models to Kaggle:
   - `thanhduc1108/qwen_35_27b` (Teacher model)
   - `thanhduc1108/qwen_35_4b` (Student model)

3. Upload pipeline code:
   - `thanhduc1108/vlsp2025-kd-pipeline` (Run `python scripts/kaggle_upload.py --upload-code`)

4. (Optional) Upload wheels for offline:
   - `thanhduc1108/vlsp2025-kd-wheels`

### 7.2 Kaggle Notebook Setup

1. Create new notebook on Kaggle
2. Select **GPU RTX 6000 Pro** accelerator
3. Add all input datasets listed above
4. Set **Internet OFF** for competition compliance
5. Copy contents of `kaggle/kaggle_kd_notebook.py` into notebook cells
6. Run all cells

### 7.3 Expected Timeline (RTX 6000 Pro 96GB)

| Phase | Estimated Time |
|-------|---------------|
| Data Preparation | ~2 minutes |
| Teacher Distillation (11K samples) | ~3-4 hours |
| SFT Training (3 epochs) | ~1-2 hours |
| GRPO Training (1 epoch) | ~1-2 hours |
| Inference (15 candidates, 497 test) | ~1 hour |
| Evaluation | ~1 minute |
| **Total** | **~6-9 hours** |

---

## 8. Troubleshooting

### Common Issues

1. **`qwen3_5` model type not recognized**: Upgrade transformers to >= 5.0
2. **HybridCache import error**: Upgrade peft to >= 0.18
3. **CUDA OOM on P100**: Reduce `max_seq_length`, use 4bit quantization
4. **Flash Attention not available**: Install `flash-attn` or set `use_flash_attention: false`
5. **grad_norm: nan**: Normal for FP16 on P100 with small batches
6. **Low teacher match rate**: Increase `max_retries`, check prompt template format

### Memory Optimization

- Enable gradient checkpointing: trades compute for memory
- Use LoRA: only 5-8% parameters are trainable
- Use 4bit quantization for teacher on limited GPUs
- Reduce batch size and increase gradient accumulation

---

## 9. References

1. HUSTUET (2025). "Program-Centric Policy Optimization for Financial Numerical Reasoning." VLSP 2025.
2. Chen et al. (2021). "FinQA: A Dataset of Numerical Reasoning over Financial Data." EMNLP.
3. Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models."
4. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
5. Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
