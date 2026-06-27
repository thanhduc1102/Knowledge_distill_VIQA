# AAAI-27 Final Research Direction

This document supersedes the older "KG improves answer accuracy" narrative.  The current
evidence supports a narrower and stronger paper:

**Auditable Conditional-Salience Retrieval for Financial RAG**

## Thesis

Financial RAG fails in two places that document-level retrievers and answer-only LLM
metrics hide:

1. selecting the right evidence inside an entity/company cluster, where global semantic
   similarity is weak;
2. knowing whether a generated numeric answer is grounded in a verifiable financial fact.

The system should therefore be positioned around:

- **conditional salience retrieval**: BM25/IDF estimated inside the company or filing pool;
- **Fact Ledger verification**: cell-level grounding, arithmetic/provenance checks, and
  risk-coverage diagnostics;
- **verify-then-reask**: raw table context first, KG evidence only when the raw answer is
  ungrounded.

## What changed

We explicitly drop the headline claim that KG evidence universally increases Number Match.
The experiments show the opposite on stronger generators: raw table context can beat filtered
KG evidence.  KG is valuable as an audit and calibration layer, not as a hard answer override.

## New implemented experiments

- `scripts/research/external_financebench_eval.py`
  - external FinanceBench open-source evidence retrieval;
  - compares BM25, company-local loclex, company-year loclex, and cross-encoder rerank.
- `scripts/research/faithfulness_risk_eval.py`
  - grounded vs ungrounded correctness;
  - hallucination-catch proxy;
  - provenance coverage/precision proxy;
  - risk-coverage curve.
- `scripts/research/verify_then_reask.py`
  - raw answer is kept when grounded;
  - ungrounded answers are re-asked with KG evidence/provenance.
- `scripts/research/learned_coordinate_eval.py`
  - weakly learned row/column matcher from answer supervision;
  - negative/limited result is reported honestly.
- `scripts/research/paper_ablation_report.py`
  - single JSON artifact collecting retrieval ablations, external benchmark, faithfulness,
    verify-then-reask, and coordinate results.

## Key results after the update

### External benchmark: FinanceBench evidence retrieval

| Method | MRR@3 | R@1 | R@3 | R@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.3211 | 0.2533 | 0.4133 | 0.4867 |
| BM25 + cross-encoder rerank | 0.4556 | 0.3667 | 0.5733 | 0.6667 |
| company loclex | 0.6867 | 0.5533 | 0.8533 | 0.9467 |
| company-year loclex | 0.8144 | 0.7267 | 0.9200 | 0.9933 |

This is not T2-RAGBench and therefore reduces the artifact-only concern.  It is an
evidence-retrieval benchmark rather than full-PDF retrieval, so the paper must label it
accurately.

### Faithfulness

| Dataset | grounded acc | ungrounded acc | separation | catch proxy |
|---|---:|---:|---:|---:|
| FinQA | 0.0966 | 0.0999 | -0.0033 | 0.8461 |
| ConvFinQA | 0.4504 | 0.1073 | 0.3431 | 0.7123 |
| TAT-DQA | 0.2696 | 0.0952 | 0.1744 | 0.7825 |

Faithfulness is strong on ConvFinQA and TAT-DQA, but not on FinQA for the current Qwen3.5
run.  This must be reported as a limitation.

### Verify-then-reask

| Dataset | raw NM | verify-then-reask NM |
|---|---:|---:|
| FinQA | 0.0994 | 0.1526 |
| ConvFinQA | 0.2432 | 0.2840 |
| TAT-DQA | 0.1399 | 0.1407 |

The new policy improves FinQA/ConvFinQA and is neutral on TAT-DQA using deterministic
extractive re-ask.  It is the correct direction because it does not discard raw table
context when the model already produced a grounded answer.

### Learned coordinate matcher

| Dataset | heuristic | coord | learned | any-path upper signal |
|---|---:|---:|---:|---:|
| FinQA | 0.3459 | 0.2579 | 0.2642 | 0.3836 |
| ConvFinQA | 0.4189 | 0.3470 | 0.3790 | 0.4817 |
| TAT-DQA | 0.0975 | 0.1191 | 0.0939 | 0.1480 |

The learned matcher is not a standalone win.  The useful result is that independent
grounding paths are complementary; the next step is a calibrated selector/agreement model,
not another single-path heuristic.

## Paper claims that are safe

- Conditional salience inside entity clusters is a robust retrieval signal.
- Sparse local lexical evidence can beat dense and reranker baselines in financial
  evidence retrieval.
- Fact Ledger should be used as verifier/provenance, not as a universal answer generator.
- Verify-then-reask is a safer inference policy than symbolic override.

## Claims to avoid

- "KG improves end-to-end Number Match on all datasets."
- "Accounting identities are the main verifier."
- "Fact-level neural retrieval beats loclex."
- "Metadata filtering alone is the contribution."

## Remaining must-do before submission

- Add full-PDF or long-document retrieval if time permits.
- Run Qwen re-ask with HF generator, not only deterministic extractive re-ask.
- Add confidence intervals / bootstrap significance.
- Add human or semi-automatic audit for provenance precision on a 100-sample subset.
- Consolidate old docs so reviewers and collaborators do not see conflicting claims.
