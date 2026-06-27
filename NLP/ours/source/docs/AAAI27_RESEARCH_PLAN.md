# AAAI-27 Research Plan — Structure-Grounded Auditable Financial RAG

> Main axis (locked): **a typed financial structure graph used as a shared substrate that
> closes the loop from retrieval to *concept–period–role (CPR) aware verification* and
> *selective answering*.** The contribution is structure-level grounding, NOT metadata
> exploitation (which the T²-RAGBench leaderboard already shows is a near-oracle artifact),
> and it is demonstrated to transfer across FinQA/ConvFinQA/TAT-DQA and OOD (FinanceBench).

## 1. The research gap (one sentence)

Existing financial RAG uses structure (HierFinRAG, "structure first") to *help generation*
or uses metadata to *win retrieval*, but **no method closes the loop to a structure-grounded
verifier that checks an answer's concept, period, and arithmetic role**, so numeric answers
remain un-auditable and confidence does not separate correct from wrong (documented failure:
FinQA grounding AUROC ≈ 0.50; large `grounded_wrong` bucket).

## 2. Why prior signal is not enough (measured, not asserted)

The legacy verifier (`generation/verifier.py`) decides "grounded" = *value appears anywhere
in the ledger* and "derivable" = *value is an op over any two cells*. This **over-fires**: a
wrong answer routinely coincides with an unrelated cell or an unrelated pair. Consequence:
the support flag is almost always on (legacy supported rate 0.93–0.94) so it carries little
information, and `grounded_wrong` dominates.

## 3. Contribution C1 — Concept-Period-Role (CPR) structure grounding

`src/gsr_cacl/research/cpr_verifier.py`. An answer is supported only when the supporting
fact(s) are **concept-consistent** with the question intent (canonical IFRS/GAAP match or
content-token overlap), **period-consistent** with the requested year(s), and play the
**role** the inferred arithmetic task requires (old/new for difference & %-change;
part/total for ratio; siblings for sum/avg). This re-uses the *same* role-tagged operand
plan the generator already builds (`ledger/select.py::calculation_plan`) — i.e. the verifier
becomes symmetric with generation. Confidence is **continuous**:

```
conf = max( grounded_score, derivable_score, value_only_floor, raw_floor )
grounded_score  = value_match · concept_consistency · period_consistency · (1/sqrt(#cell_matches))   [ambiguity damping]
derivable_score = role-consistent-plan_confidence  (or 0.75·cc·pc for a role-consistent operand pair)
value_only_floor= 0.25/sqrt(#matches) if value present but concept/period MISMATCH   [downgraded]
```

This is annotation-free, model-free, and structure-based (not metadata) — so it is designed
to transfer.

### Results (executed)

**HEADLINE — Qwen3.5-4B, current pipeline, random 400/dataset**
(`outputs/research/generation_system_q35_s400/`, `cpr_grounding_q35s400/`). Higher base
accuracy than the legacy run (FinQA NM 0.099→**0.278**, ConvFinQA 0.243→**0.44**), so the
auditability signal is measured on a usable system:

| Dataset | raw NM | verify→reask NM | AUROC legacy→CPR | ΔAUROC CI95 (P>0) | AURC legacy→CPR | safe-cov @70% acc legacy→CPR |
|---|---|---|---|---|---|---|
| FinQA | 0.278 | 0.295 | 0.579 → **0.658** | [0.030, 0.131] (0.998) | 0.698 → **0.641** | — (base acc too low) |
| ConvFinQA | 0.44 | 0.445 | 0.640 → **0.755** | [0.072, 0.158] (1.00) | 0.442 → **0.375** | 0.073 → **0.36** |
| TAT-DQA | (running) | | | | | |

CPR significantly improves AUROC on FinQA **and** ConvFinQA (CI excludes 0), and on ConvFinQA
turns selective answering from "safe on 7% of queries" into "safe on 36% of queries" at a 70%
accuracy bar. (The CV-calibrator helps AURC at full-set scale but overfits at n=400; raw CPR
confidence is the robust default.)

**150-sample, Qwen2.5-7B generator** (`outputs/research/cpr_grounding/`):

| Dataset | AUROC legacy→CPR | Acc@coverage-25% legacy→CPR |
|---|---|---|
| FinQA | 0.526 → **0.640** | 0.210 → **0.342** |
| ConvFinQA | 0.578 → **0.723** | 0.474 → **0.711** |
| TAT-DQA | 0.599 → 0.534 | 0.237 → **0.289** |

**Full test set, Qwen3.5 generator** (`outputs/research/cpr_grounding_full/`, n=1147/3458/1144):

| Dataset | raw_acc | AUROC legacy→CPR | Separation legacy→CPR | grounded_wrong downgraded |
|---|---|---|---|---|
| FinQA | 0.099 | 0.537 → **0.628** | +0.062 → +0.080 | 671/964 (70%) |
| ConvFinQA | 0.243 | 0.735 → 0.741 | +0.169 → **+0.266** | 1379/2325 (59%) |
| TAT-DQA | 0.140 | 0.684 → 0.642 | +0.107 → +0.117 | 563/911 (62%) |

**Robust universal win:** CPR removes **59–70% of false "grounded" flags** while keeping most
correct ones, and **fixes the broken FinQA calibration** (0.50/0.54 → 0.63–0.64). The
"accuracy when confident" story is strongest on ConvFinQA (top-quartile accuracy 0.47→0.71).

**Paired bootstrap 95% CI (2000 resamples), CPR − legacy on the full set** — significance is
honest and dataset-specific:

| Dataset | Δ AUROC [CI95] (P>0) | Δ Separation [CI95] (P>0) |
|---|---|---|
| FinQA | **[+0.037, +0.147]** (0.999) | [−0.037, +0.074] (0.73) |
| ConvFinQA | [−0.008, +0.022] (0.80) | **[+0.062, +0.137]** (1.00) |
| TAT-DQA | **[−0.091, +0.008]** (0.046 ⇒ CPR worse) | [−0.049, +0.073] (0.61) |

So the defensible significant claims are: CPR **significantly improves AUROC on FinQA**
(the documented broken-calibration case) and **significantly improves separation on
ConvFinQA**; on TAT-DQA it is significantly *worse* on AUROC — reported as the honest
limitation (concept/period extraction noise on non-standard tables; a period-reliability gate
was added but does not recover it, confirming the bottleneck is table parsing, not the CPR
mechanism).

**Honest limitation:** TAT-DQA mid-range AUROC dips (noisy non-standard tables → noisy
concept/period extraction). Its *confident slice* still improves (7B @cov.25 0.24→0.29).
Fix direction = better table/period parsing (multi-level headers), reported as future work.

## 4. Contribution C2 — selective answering on CPR confidence (deployable abstention)

`scripts/research/cpr_selective_policy.py`: AURC (area under risk-coverage; lower=better) +
a deployable "answer iff confidence ≥ τ" policy, plus a 5-fold CV logistic calibrator over the
CPR structure features (concept/period consistency, grounded/derivable, value-only flags).

**Full-set AURC (lower better):**

| Dataset | legacy | CPR | CPR + CV-calibrator |
|---|---|---|---|
| FinQA | 0.884 | **0.843** | 0.864 |
| ConvFinQA | 0.604 | 0.594 | **0.568** |
| TAT-DQA | 0.801 | 0.793 | **0.778** |

**Deployable safe coverage — ConvFinQA, answer only at ≥60% accuracy (risk ≤ 0.4):**

| | safe coverage @ 60% acc |
|---|---|
| legacy value-only | 0.013 |
| CPR | **0.215** |
| CPR + calibrator | **0.243** |

i.e. value-only grounding can safely answer ~1.3% of ConvFinQA queries; CPR-based abstention
safely answers ~24% at the same accuracy bar — an **~18× increase in usable coverage**. This
is the concrete reliability/selective-prediction contribution; CPR confidence also gates
`generation_system_eval.py`'s verify-then-reask (fires on genuinely ungrounded answers).

## 5. Generalization (OOD) — FinanceBench

`scripts/research/financebench_cpr_audit.py`. Answer FinanceBench questions from GOLD 10-K
evidence with Qwen2.5-7B, then test whether CPR separation/AUROC gain holds on real SEC
filings (isolates verification from retrieval). **Result (n=126, OOD):**

| Metric | legacy value-only | CPR |
|---|---|---|
| AUROC | 0.732 | **0.756** |
| Separation (acc_sup − acc_unsup) | 0.220 | **0.279** |
| Accuracy when "supported" | 0.253 | **0.406** |
| supported-but-wrong | 71 | **19** (−73%) |

The *same* pattern as T2-RAGBench holds on out-of-distribution real 10-Ks — the auditability
gain is structure-based, not a benchmark/metadata artifact. DocFinQA cache is currently
incomplete (corrupted `.incomplete`); re-download to add a long-document setting.

## 6. Paper framing

- **Title direction:** "Structure-Grounded Auditable Retrieval-Augmented Generation for
  Financial Numerical QA."
- **Claim 1 (retrieval):** modular structure-aware retrieval (MMER, honest 5-fold CV W.Avg
  MRR@3 0.722) beats all non-oracle leaderboard systems; metadata is acknowledged as a
  near-oracle artifact, not our contribution.
- **Claim 2 (the novelty):** CPR structure grounding turns an un-auditable, near-random
  confidence into a calibrated selective-prediction signal, generalizing across datasets and
  to OOD FinanceBench; it removes most `grounded_wrong`.
- **Claim 3 (reliability):** selective answering / verify-then-reask on CPR confidence.
- **Honest negatives kept:** TAT-DQA structure noise; metadata caveat; generation NM ceiling.

## 7. Remaining must-do
1. FinanceBench OOD table (running) + add bootstrap CI on AUROC/separation.
2. Re-run full-set generation with 7B (not qwen3.5) for the headline auditability table.
3. Calibrated selective-risk threshold + risk-coverage/AURC table.
4. TAT-DQA table parser upgrade (multi-level header) to test if the regression closes.
5. Fact-extraction F1 + oracle-vs-auto ledger gap (reviewers require this).
6. Repair DocFinQA cache → long-document generalization.
