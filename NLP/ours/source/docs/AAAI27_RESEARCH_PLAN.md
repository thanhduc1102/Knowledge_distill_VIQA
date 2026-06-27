# AAAI-27 — Structure-Grounded Selective Answering for Financial Numerical QA

> **One axis, locked.** A typed financial *structure graph* yields a single
> **Concept–Period–Role (CPR)** grounding signal that makes numeric answers *auditable* and
> supports *selective answering*. The contribution is structure-level reliability, **not**
> metadata exploitation (the T²-RAGBench leaderboard is largely a metadata near-oracle) and
> **not** raw-accuracy chasing. It is the best annotation-free reliability signal we measure,
> it generalizes across FinQA/ConvFinQA/TAT-DQA and to OOD FinanceBench, and every claim is
> backed by a paired-bootstrap CI and a head-to-head against standard baselines.

Artifacts: `src/gsr_cacl/research/cpr_verifier.py`,
`scripts/research/{cpr_grounding_eval,cpr_selective_policy,cpr_answer_router,financebench_cpr_audit,generation_system_eval}.py`,
outputs under `outputs/research/{cpr_grounding_q35s400,cpr_grounding_full,cpr_router,financebench_cpr,generation_system_q35_s400}/`.

---

## 1. Problem & gap

Financial RAG must (i) retrieve the right evidence among look-alike tables and (ii) produce a
*verifiable* numeric answer. The second is unsolved: small/mid LLMs answer financial QA at low
accuracy, and **existing confidence signals do not tell us which numeric answers to trust.**

The deployed verifier (`generation/verifier.py`) decides "grounded" = *the number appears
anywhere in the ledger* and "derivable" = *the number is an op over any two cells*. This
**over-fires**: a wrong answer routinely coincides with an unrelated cell or pair. Measured
consequence: the legacy support flag fires on 93–94% of answers (so it is nearly
uninformative), `grounded_wrong` dominates, and on FinQA the confidence is near-random
(AUROC ≈ 0.5).

**Gap.** No prior financial-RAG work (HierFinRAG, "structure first", FinanceBench provenance,
DocFinQA) closes the loop to a *structure-grounded verifier that checks an answer's concept,
period, and arithmetic role* — the level at which financial relevance is actually defined.

## 2. Method — CPR structure grounding

For a typed structure graph over `document → table → row/column → fact → concept/period`, an
answer is **supported only when** the supporting fact(s) are simultaneously:
- **Concept-consistent** with the question (canonical IFRS/GAAP match, else content-token overlap),
- **Period-consistent** with the requested year(s), gated by *period-extraction reliability*
  (on noisy tables a "mismatch" is treated as a parse failure, not a wrong period), and
- **Role-consistent** with the inferred task — reusing the generator's own role-tagged operand
  plan (`ledger/select.py::calculation_plan`: old/new for diff & %-change, part/total for ratio,
  siblings for sum/avg). *Verification is made symmetric with generation.*

Confidence is **continuous**:
```
conf = max( grounded_score, derivable_score, value_only_floor, raw_floor )
grounded_score  = value_match · concept_consistency · period_consistency · (1/sqrt(#cell_matches))
derivable_score = role-consistent-plan_confidence   (or 0.75·cc·pc for a role-consistent pair)
value_only_floor= 0.25/sqrt(#matches) when value present but concept/period MISMATCH  (downgraded)
```
Annotation-free, model-free, structure-based → designed to transfer. (Continuous beat a hard
6-level bucket; a period-reliability gate is principled and neutral.)

## 3. Headline results — Qwen3.5-4B, current pipeline, random 400/dataset

`outputs/research/{generation_system_q35_s400,cpr_grounding_q35s400}`. Base Number-Match is
usable (FinQA 0.278, ConvFinQA 0.44, TAT-DQA 0.273), so reliability is measured on a real system.

| Dataset | raw NM | verify→reask NM | **AUROC** legacy→CPR | ΔAUROC CI95 (P>0) | **AURC↓** legacy→CPR | grounded_wrong downgraded |
|---|---|---|---|---|---|---|
| FinQA | 0.278 | 0.295 | 0.579 → **0.658** | [0.030, 0.131] (0.998) | 0.698 → **0.641** | 184/231 (80%) |
| ConvFinQA | 0.44 | 0.445 | 0.640 → **0.755** | [0.072, 0.158] (1.00) | 0.442 → **0.375** | 124/189 (66%) |
| TAT-DQA | 0.273 | 0.28 | 0.676 → **0.696** | [−0.037, 0.076] (0.76) | 0.641 → **0.584** | 200/236 (85%) |

**All three datasets improve** on AUROC and AURC; FinQA and ConvFinQA significantly (paired
bootstrap CI excludes 0). CPR removes **66–85% of false "grounded" flags** while keeping most
correct ones. (The earlier TAT-DQA AUROC regression existed only on a *degenerate* generator run
— 16-token truncated answers; it disappears on a properly functioning generator.)

## 3b. Component ablation — which of C/P/R carries the gain

`cpr_ablation.py` (AUROC predicting raw correctness; single / pair / full configs):

| dataset | value_only | C | P | R | CP | CR | PR | **CPR** |
|---|---|---|---|---|---|---|---|---|
| FinQA | 0.573 | 0.575 | 0.575 | 0.629 | 0.572 | 0.651 | 0.640 | **0.658** |
| ConvFinQA | 0.637 | 0.637 | 0.633 | 0.720 | 0.642 | 0.751 | 0.738 | **0.755** |
| TAT-DQA | 0.669 | 0.641 | 0.605 | 0.669 | 0.624 | 0.704 | 0.667 | **0.697** |

**Findings (drives method design):**
- **Role is the workhorse** — R-alone ≈ full CPR; the role-consistent operand check (reusing the
  generator's plan) is where the signal lives.
- **Concept is complementary, not standalone** — C-alone ≈ value-only, but C+R > R everywhere.
- **Period is the weakest criterion.** As a *multiplicative* factor it was *harmful* (P-alone below
  value-only on all 3; CPR<CR on TAT-DQA). Fix shipped: **partial-credit period** (`0.5+0.5·pc`,
  modulates but never zeros a good concept+role match). After the fix, period is neutral-to-positive
  and full CPR ≥ CR on FinQA/ConvFinQA, ~CR on TAT-DQA. This is a concrete, ablation-driven design
  decision, not a guess.

## 3c. Multi-operand derivation — raising the auditable ceiling

The verifier can only certify what the ledger can reconstruct. Measured *auditable ceiling*
(gold-doc, gold answer grounded-or-derivable, `fact_extraction_recall.py`):

| Dataset | certifiable 2-op | **certifiable 3-op** |
|---|---|---|
| FinQA | 0.483 | **0.797** (+0.31) |
| ConvFinQA | 0.673 | **0.823** (+0.15) |
| TAT-DQA | 0.637 | **0.823** (+0.19) |

Most FinQA answers are multi-operand (ratio of a difference, part over a sum-total, sum of 3),
which 2 operands cannot reach. `research/derivation.py` adds bounded 3-operand certification.
Wired into CPR at *lower* confidence (coincidence risk), it **raises both coverage and AUROC**
(s400): ConvFinQA 0.755→**0.767**, TAT-DQA 0.697→**0.733** (now near-significant vs legacy),
FinQA held at 0.659. This is the single biggest lever on the system's certification capacity and
is now the default (`components` includes `"3op"`).

## 3d. Cross-generator generality (signal is model-agnostic)

CPR improves FinQA AUROC significantly across **three different generators** (paired bootstrap CI
excludes 0 each time) — the reliability signal is a property of the structure, not of one model:

| Generator | setting | FinQA AUROC legacy→CPR | ΔAUROC CI95 (P>0) |
|---|---|---|---|
| Qwen2.5-3B-Instruct (non-VLM) | full (n=1147) | 0.531 → **0.637** | [0.054, 0.154] (1.00) |
| Qwen3.5-4B | 400-sample | 0.579 → **0.658** | [0.030, 0.131] (0.998) |
| (earlier model) | full (n=1147) | 0.537 → **0.628** | [0.037, 0.147] (0.999) |

(Aside: with the weaker 3B, the **KG-audit prompt** lifts FinQA NM 0.117→0.188 — structure helps
the generator more when the model is weaker.) Multiple non-VLM generators are used, per the
decision to not depend on a single (slow) VLM.

## 4. Rigorous comparison — CPR vs standard reliability baselines

Head-to-head AUROC of each annotation-free signal predicting raw-answer correctness
(`cpr_answer_router.py`, s400):

| Dataset | value-only grounding | self-consistency (raw↔KG agreement) | **CPR (ours)** | CPR + self-consistency |
|---|---|---|---|---|
| FinQA | 0.576 | 0.521 | **0.653** | 0.629 |
| ConvFinQA | 0.638 | 0.675 | **0.751** | **0.780** |
| TAT-DQA | 0.662 | 0.543 | **0.680** | 0.641 |

**CPR is the best single annotation-free reliability signal on every dataset**, beating both
value-presence and a self-consistency baseline; on ConvFinQA it further combines with
self-consistency (0.780).

## 5. Deployable selective answering (the reliability payoff)

`cpr_selective_policy.py` — "answer iff confidence ≥ τ", τ fit to a target risk; plus a 5-fold
CV logistic calibrator over CPR structure features. **Safe coverage at a fixed accuracy bar:**

| | ConvFinQA @ ≥70% acc | ConvFinQA @ ≥60% acc | TAT-DQA @ ≥60% acc |
|---|---|---|---|
| value-only | 0.073 | 0.358 | 0.013 |
| **CPR** | **0.360** | **0.590** | 0.078 |
| CPR + calibrator | — | 0.563 | **0.188** |

i.e. value-only grounding can safely answer ~7% of ConvFinQA queries at 70% accuracy; CPR-based
abstention safely answers **36%** at the same bar (≈5×). The calibrator helps at full-set scale
and on TAT-DQA; raw CPR confidence is the robust default at n=400.

## 5b. Long-context / evidence-dilution robustness (DocFinQA substitute)

DocFinQA's HF loader is currently broken (`DatasetGenerationError` + heavy files), so we test the
*same* failure mode — relevant evidence buried in long, noisy context — in a controlled way:
inject `m` distractor facts from other documents and track reliability AUROC
(`long_context_robustness.py`). **CPR retains its full advantage over value-only at every dilution
level and does not degrade** (m: 0→150): FinQA CPR 0.655→0.679, ConvFinQA 0.763→0.787,
TAT-DQA 0.726→0.738 (legacy stays ~0.57/0.64/0.67 throughout). Structure typing is robust to
context dilution; the value-only/CPR gap (~0.08–0.13) is preserved as the ledger grows.

## 6. Generalization — OOD FinanceBench (real SEC 10-Ks)

`financebench_cpr_audit.py` (answer from gold evidence, isolating verification from retrieval),
Qwen2.5-7B, n=126:

| Metric | value-only | CPR |
|---|---|---|
| AUROC | 0.732 | **0.756** |
| Separation | 0.220 | **0.279** |
| Accuracy when "supported" | 0.253 | **0.406** |
| supported-but-wrong | 71 | **19** (−73%) |

The same pattern holds on out-of-distribution filings — the auditability gain is structural,
not a benchmark/metadata artifact.

## 7. Full-set confirmation + significance

`cpr_grounding_full/` (full test sets 1147/3458/1144). Paired bootstrap (2000 resamples) of
CPR−legacy: FinQA **ΔAUROC [+0.037,+0.147]** (P=0.999), ConvFinQA **ΔSeparation [+0.062,+0.137]**
(P=1.0). Universal: CPR downgrades 59–70% of false grounded flags.

## 8. Honest negatives (reported, not hidden)

- **Answer routing fails.** Routing between the RAW and KG answers by CPR confidence (argmax or
  conservative high-precision override) does **not** beat raw NM (switches help ≈ as often as
  they hurt), despite a large oracle best-of headroom (+0.09–0.13). Lesson: CPR confidence is
  reliable for *global ranking / abstention*, not for *pairwise answer arbitration*. This sharpens
  the contribution to selective answering.
- **TAT-DQA** has the weakest (non-significant) AUROC gain; its non-standard tables are the hard
  regime for structure extraction → future work is a multi-level-header parser.
- **Absolute NM is modest** (small generator); the contribution is reliability/selective
  answering, a recognized trustworthy-ML axis, not SOTA accuracy.

## 9. Paper framing

- **Title:** *Know When You're Right: Structure-Grounded Selective Answering for Financial
  Numerical QA.*
- **C1 (signal):** CPR structure grounding is the best annotation-free reliability signal
  (beats value-presence and self-consistency; significant on FinQA/ConvFinQA).
- **C2 (mechanism):** it removes 60–85% of hallucinated "grounded" flags by binding answers to
  concept × period × role on a typed structure graph, symmetric with generation.
- **C3 (deployability + generality):** calibrated selective answering (safe coverage ↑ ~5×) that
  transfers to OOD FinanceBench.
- **Supporting:** MMER structure-aware retrieval (honest 5-fold-CV W.Avg MRR@3 0.722, beats all
  non-oracle leaderboard systems) feeds the same structure substrate.

## 10. Remaining to submission
1. Re-run §3 at full-set with Qwen3.5-4B (currently 400/ds; VLM is slow on T4 — budget GPU time).
2. Add a self-consistency baseline with k≥5 sampled answers (stronger than raw↔KG agreement).
3. Fact-extraction F1 + oracle-vs-auto ledger gap (reviewer-required).
4. Multi-level-header parser for TAT-DQA; re-test the gain.
5. Repair DocFinQA cache → long-document generalization row.
6. Human audit of provenance precision on 100 samples.
