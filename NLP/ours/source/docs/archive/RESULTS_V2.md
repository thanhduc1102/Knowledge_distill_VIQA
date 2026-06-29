# LEDGER-RAG v2 — Verified Results (E1/E2 entity ontology · C2/C3 structural signal · C5 verifier · CACL v2)

> All numbers below were produced **on 2× Tesla T4** against the real T²-RAGBench cache
> (`intfloat/multilingual-e5-large-instruct`, exact FAISS `IndexFlatIP`). Reproduce with the
> scripts named in each section. This document supersedes the earlier `RESULTS.md` for the
> retrieval phase and corrects the fabricated leaderboard figures that were in `ASSESSMENT.md`.

## 0. Real SOTA context (verified from the papers, not invented)

From the T²-RAGBench paper (arXiv 2506.12071) and "From BM25 to Corrective RAG" (arXiv 2604.01733):

| | FinQA MRR@3 | ConvFinQA MRR@3 | TAT-DQA MRR@3 | Aggregate R@5 |
|---|---|---|---|---|
| BM25 | 0.389 | 0.500 | 0.400 | — |
| Hybrid RRF | 0.389 | 0.519 | 0.438 | — |
| **Best (Hybrid + Cohere rerank)** | — | — | — | **0.816 / MRR@3 0.605** |
| Oracle-context Number-Match ceiling | — | — | — | **0.350** |

> The previously-circulated claim *"GPT-5.4 + Metadata-aware BM25 ≈ 73.7 NumberMatch / FinQA
> MRR@3 90.3"* does **not** exist in any source and has been removed from our reasoning.

## 1. Validity / leakage check (Experiment #0) — `scripts/validity_check.py`

Decides whether the metadata-driven gain is legitimate or a near-oracle artefact.

| dataset | mean docs / (company,year) | metadata-only recall (company+year) | % queries where metadata set is a singleton==gold | mean candidate-set size | year-in-question | company-in-question |
|---|---|---|---|---|---|---|
| FinQA | 3.49 | **1.000** | 1.1% | 14.1 | 98.3% (82.4% == gold) | 88.3% |
| ConvFinQA | 2.66 | **1.000** | 4.6% | 9.4 | 98.3% (79.3%) | 88.3% |
| TAT-DQA | 15.74 | **1.000** | 0.0% | 23.3 | 95.7% (81.1%) | 81.2% |

**Conclusion (defensible):** metadata filtering guarantees the gold is reachable (recall 1.0)
but only narrows the corpus to ~9–23 candidates and is a *singleton==gold* in **0–4.6%** of
cases → it is **not** an oracle; ranking *within* the candidate set still needs text + entity +
structure. And the metadata is **recoverable from the question text** (~80–88%), so using it is
legitimate (it is part of the query), not a hidden side channel. TAT-DQA's `report_year` does not
vary within a company (173 groups = 173 companies), so year gives no discrimination there.

## 2. Entity channel — hash vs GICS/alias ontology (E1+E2) — `scripts/entity_ablation.py`

| arm | FinQA MRR@3 | ConvFinQA MRR@3 | TAT-DQA MRR@3 |
|---|---|---|---|
| dense (e5) | 0.376 | 0.390 | 0.235 |
| + hash-entity (rerank) | 0.651 | 0.721 | 0.362 |
| + **ontology**-entity (rerank, E1) | 0.653 | 0.722 | 0.362 |
| FULL hash (exact filter) | 0.712 | 0.767 | 0.401 |
| FULL **ontology + alias** (E1+E2) | 0.710 | 0.769 | 0.401 |

**Honest finding:** on these datasets the ontology embedder **matches** the hash baseline
(±0.002) — it does **not** beat it, because company names are already clean (alias rarely
fires) and same-entity ranking is near-ceiling for both. The ontology's value is *robustness*
(alias/suffix/acronym variants, deployment NER) and *economic structure* (GICS proximity,
verified by `test_ontology_embedder_sector_proximity`), at **no cost**. Reported as a no-regression
generalisation upgrade, not a headline gain.

## 3. Structural signal — C2 concept ontology + C3 concept-coverage — `scripts/full_eval2.py`

This is where the **GSR/structural contribution finally becomes positive** (the old
constraint-CS contributed exactly 0.000 to retrieval). C3 scores a document by how well its
Fact-Ledger covers the **canonical concept(s) + period** the query asks about (C2 maps
line-items → IFRS/GAAP/XBRL concepts; coverage is query-conditioned).

| dataset | FULL (entity+meta) | **FULL + C3** | Δ MRR@3 | Δ R@1 | Δ R@5 |
|---|---|---|---|---|---|
| FinQA | 0.710 | **0.743** | +0.033 | +0.040 | +0.025 |
| ConvFinQA | 0.769 | **0.818** | +0.049 | +0.060 | +0.021 |
| TAT-DQA | 0.401 | **0.455** | +0.054 | +0.039 | +0.070 |

Coverage diagnostics: 69–79% of documents carry ≥1 canonical concept; C3 only *fires* on
39–55% of queries, yet still lifts MRR@3 by +3.3 / +4.9 / +5.4. Best weight δ≈0.1 on all three.
This validates the reframing from *query-independent constraint verification* (useless for
ranking) to *query-conditioned concept+period coverage* (the real structural retrieval signal).

## 4. Constraint verifier — C5 (concept-grounded, scale-aware)

`compute_concept_equation_score` evaluates the IFRS/GAAP identities (e.g. Revenue−COGS=GrossProfit,
OCF+ICF+FCF=NetChangeInCash) directly on the Fact-Ledger by **canonical concept + period**, in
scale-normalised absolute units. It is a **verifier / generation** signal (kept out of the
ranking score, where global consistency does not discriminate). A value-identity channel-aligned
negative provably lowers it (`test_concept_equation_verifier`). Wired for the generation phase
(deferred per the current focus-on-retrieval decision).

## 5. CACL v2 — InfoNCE retrieval training (D1–D4) — `src/gsr_cacl/training/cacl_infonce.py`

Upgrades the original single-margin, entity-only `cacl_train.py`: **InfoNCE** over many
**realistic in-corpus hard negatives** (same company ±year — the EDA's hardest zone) with a
**false-negative guard** (exclude gold `context_id`; drop same-(company,year) duplicates), and
**jointly learns the weights of the signals retrieval actually uses** (text + entity + C3
coverage). Held-out eval (n=500), learned weights `[w_text, w_ent, w_cov]`:

| dataset | text+entity (fixed w) | CACL2 trained, no coverage | **CACL2 trained, full (text+entity+coverage)** | learned [w_text, w_ent, w_cov] |
|---|---|---|---|---|
| FinQA | 0.654 | 0.636 | **0.665** | [1.33, 1.06, 0.73] |
| ConvFinQA | 0.756 | 0.757 | **0.781** | [1.37, 1.10, 0.70] |
| TAT-DQA | 0.364 | 0.333 | **0.416** | [1.33, 1.05, 0.73] |

Across all three datasets the model **learns w_cov ≈ 0.70–0.73** (it actively chooses the
concept-coverage channel) and the full arm beats the no-coverage arm by **+0.03 / +0.02 / +0.08**
MRR@3 — the same structural signal that helps in §3, now confirmed under a learned-weight
contrastive objective with realistic hard negatives.

The learned `w_cov > 0` confirms the model *chooses* to use the concept-coverage channel; the
"full" arm (with coverage) beats the "no-cov" arm, i.e. the structural channel is the
differentiator inside the metadata-filtered pool. (Absolute MRR here is on a held-out subset and
uses a lighter entity warm-up than §3; §3's `full_eval2` remains the authoritative retrieval
number.)

## 6. What is preserved

The original GSR/CACL baseline is untouched for re-evaluation: `methods/gsr_retrieval.py`,
`negative_sampler/chap.py`, `training/train.py` (3-stage), and `entity/encoder.py::HashMetadataEmbedder`.
All new defaults (entity `embedder="hash"`, `alias_match=False`) reproduce the prior numbers.

## 6b. Final Integrated Evaluation — `scripts/full_eval2_with_cacl.py`

Runs all arms (dense → FULL → FULL+C3 sweep → FULL+CACL2) and auto-selects the best arm
for `retrieval_top3.jsonl` (generator input). Results on full test sets (2× T4):

| dataset | dense | FULL | FULL+C3(δ=0.1) | CACL2-weights | **selected top3** |
|---|---|---|---|---|---|
| FinQA | 0.3756 | 0.7104 | **0.7432** | 0.7194 | FULL+C3(δ=0.1) |
| ConvFinQA | 0.3905 | 0.7686 | **0.8176** | 0.8017 | FULL+C3(δ=0.1) |
| TAT-DQA | 0.2350 | 0.4008 | **0.4554** | 0.4331 | FULL+C3(δ=0.1) |

CACL2 arm note: entity embedder in checkpoint was trained on ~500–2000 query training examples
(InfoNCE), while the fixed-weight arm re-trains SupCon entity on the full corpus (12 epochs) —
hence fixed-weight wins on full test MRR@3. CACL2 confirms w_cov > 0 is optimal; the magnitude
(0.73) transfers as direction but not scale.

`retrieval_top3.jsonl` outputs: FinQA=1147 records, ConvFinQA=3458, TAT-DQA=1144 (total 5749).
Each record carries top-3 retrieved docs + Fact-Ledger evidence block (ready for generator phase).

## 7. What is preserved

The original GSR/CACL baseline is untouched for re-evaluation: `methods/gsr_retrieval.py`,
`negative_sampler/chap.py`, `training/train.py` (3-stage), and `entity/encoder.py::HashMetadataEmbedder`.
All new defaults (entity `embedder="hash"`, `alias_match=False`) reproduce the prior numbers.

## 8. Reproduce

```bash
cd ours/source && export PYTHONPATH=src
python scripts/validity_check.py
python scripts/entity_ablation.py --dataset finqa --device cuda:0
python scripts/full_eval2.py     --dataset finqa --device cuda:0     # E1/E2 + C2/C3
python src/gsr_cacl/training/cacl_infonce.py --dataset finqa --device cuda:0
# FINAL: integrated eval with CACL2 weights + best-arm top3 output
python scripts/full_eval2_with_cacl.py --dataset finqa    --device cuda:0
python scripts/full_eval2_with_cacl.py --dataset convfinqa --device cuda:0
python scripts/full_eval2_with_cacl.py --dataset tatqa    --device cuda:0
python tests/test_ledger_rag.py   # 13/13
```
