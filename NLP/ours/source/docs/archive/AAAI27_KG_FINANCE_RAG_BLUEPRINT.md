# AAAI-27 Blueprint: Financial Fact Graph for End-to-End Retrieval and Generation

Date: 2026-06-19

This note is the current canonical research direction after reading the repository, checking the
existing outputs, and re-checking recent financial RAG / KG-RAG literature. It deliberately treats
"metadata beats SOTA" as a benchmark observation, not as the paper's main scientific contribution.

## 1. Verdict

The repo has reached a useful but dangerous point:

- Metadata/company scoping is legitimate on T2-RAGBench because the benchmark is context-independent
  and company information is part of the question. The live leaderboard also contains a May 13, 2026
  GPT-5.4 + Metadata-aware BM25 row with strong retrieval and Number-Match.
- But metadata is not a deep research contribution. It mostly turns corpus-level retrieval into a
  smaller within-company / within-filing disambiguation problem.
- The original GSR-style KG/GAT direction is not enough: column-template accounting edges were mostly
  dead on row-major financial tables, and the current fact-level neural retriever is still weak
  without targeted training.
- The strongest AAAI story is not "KG improves retrieval"; it is:

> A typed Financial Fact Graph is a single symbolic substrate that supports retrieval, hard-negative
> learning, generator evidence selection, deterministic inference-time calculation/verification, and
> audit provenance.

That is the system-level contribution that HierFinRAG, Structure-First KG reasoning, metadata RAG,
and generic GraphRAG systems do not fully cover in one pipeline.

## 2. Repository Audit

### What Is Solid

- `ledger/` extracts row-major facts `(concept, period, value, unit, scale, provenance)` from tables.
- `ontology/concepts.py` canonicalizes important US-GAAP/IFRS-like concepts and defines accounting
  identities.
- `kg/fact_graph.py` builds a typed graph over facts with identity and temporal edges.
- `experts/local_lexical.py` and `scripts/research/difficulty_decomposition.py` expose the real
  within-company retrieval regime.
- `negative_sampler/channel_aligned.py` has the right principle: each negative breaks exactly one
  channel: metric, period, entity, scale, or identity.
- `generation/retrieval_bridge.py` can package KG evidence/provenance for the generator.
- Updated in this pass: `generation/verifier.py` now produces step-level grounding and arithmetic
  fractions. These are used now for deterministic KG-side checking; preference/RL utilities remain
  future optional work, not the current generator direction.

Smoke test:

```bash
cd ours/source
PYTHONPATH=src python tests/test_ledger_rag.py
# 16/16 tests passed
```

### What Is Still Not Enough

- KG evidence selection is still heuristic; it is not a trained fact-level retriever.
- Current verifier is deterministic and useful, but it only checks common arithmetic forms. It is not
  yet a full program-trace verifier.
- Existing Qwen generation outputs are weak:
  - FinQA NM 0.099
  - ConvFinQA NM 0.243
  - TAT-DQA NM 0.140
- The fact-level semantic stream was a negative result in zero-shot mode; it needs channel-aligned
  contrastive training to justify itself.
- There is no external benchmark validation yet on DocFinQA / LOFin / FinanceBench-style corpora.

## 3. Literature Positioning

### Financial RAG and Long Financial QA

- T2-RAGBench contains 23,088 QA pairs over 7,318 financial reports and explicitly evaluates
  retrieval before numerical reasoning. Its EACL 2026 paper says 91.3% of transformed questions were
  validated as context-independent, and Hybrid BM25 was the strongest baseline in the original study:
  https://aclanthology.org/2026.eacl-long.8/
- A 2026 retrieval benchmark over T2-RAGBench reports Hybrid + Cohere rerank as the strongest
  retrieval pipeline: Recall@5 0.816, MRR@3 0.605. It also reports BM25 beating dense retrieval on
  financial documents, and HyDE underperforming dense retrieval:
  https://arxiv.org/html/2604.01733v1
- The live T2-RAGBench leaderboard currently lists GPT-5.4 + Metadata-aware BM25 at W.Avg 73.7,
  with FinQA/ConvFinQA/TAT-DQA MRR@3 = 90.3 / 84.5 / 67.9:
  https://t2ragbench.demo.hcds.uni-hamburg.de/
- Metadata-Driven RAG for Financial QA studies FinanceBench-style long filings and finds that
  chunk metadata/contextual chunks are a major gain source:
  https://arxiv.org/abs/2510.24402
- FinanceBench has 10,231 questions and evidence strings; GPT-4-Turbo with retrieval answered or
  refused incorrectly on 81% of the sampled manual evaluation:
  https://arxiv.org/abs/2311.11944
- DocFinQA extends FinQA to full-document contexts averaging 123K words, making evidence localization
  much harder:
  https://arxiv.org/html/2401.06915v2
- LOFin uses about 145,000 SEC filings and 1,595 open-domain QA instances, explicitly targeting
  near-duplicate standardized financial documents:
  https://arxiv.org/html/2505.20368v1

### KG and Graph RAG

- HierFinRAG uses a Table-Text GNN and Symbolic-Neural Fusion, reporting FinQA EM 82.5 and strong
  FinanceBench gains. This already occupies "hierarchical table-text graph + symbolic calculator":
  https://www.mdpi.com/2227-9709/13/2/30
- Structure First, Reason Next constructs financial KG triplets with metric/company/period/value/unit
  and reports about 12% relative execution-accuracy gain over vanilla Llama-3.1-8B on FinQA:
  https://arxiv.org/abs/2601.07754
- GraphRAG, HippoRAG, KAG, and KG2RAG show graph value for global sensemaking, multi-hop entity
  retrieval, or professional-domain logical reasoning:
  https://arxiv.org/abs/2404.16130
  https://arxiv.org/abs/2405.14831
  https://arxiv.org/abs/2409.13731
  https://arxiv.org/abs/2502.06864
- But generic LLM-generated KGs are noisy. "Less is More" shows denoising LLM-generated KGs improves
  GraphRAG variants; "When to use Graphs in RAG" explicitly warns GraphRAG can underperform vanilla
  RAG depending on task:
  https://arxiv.org/abs/2510.14271
  https://arxiv.org/abs/2506.05690

### Numeric Retrieval and Process Rewards

- Quantity-aware retrieval shows magnitudes and units need explicit treatment instead of being just
  tokens:
  https://aclanthology.org/2024.findings-emnlp.707/
- Dense retrievers suffer a granularity dilemma: one embedding struggles to preserve both global
  semantics and fine-grained discriminators:
  https://aclanthology.org/2025.findings-emnlp.1051/
- Process supervision outperforms outcome-only supervision in multi-step reasoning, but classic PRMs
  require labels or learned verifiers:
  https://arxiv.org/abs/2305.20050
  https://aclanthology.org/2024.acl-long.510/

The gap: finance has deterministic values, units, periods, and accounting identities. We can build
process rewards from the ledger itself, without human step labels and without a learned PRM.

## 4. Data Diagnosis From This Repo

Difficulty decomposition is the most important honest diagnostic:

| Regime | FinQA | ConvFinQA | TAT-DQA |
|---|---:|---:|---:|
| Corpus BM25 | 0.666 | 0.641 | 0.418 |
| Corpus BM25, company masked | 0.600 | 0.590 | 0.394 |
| Company pool + global BM25 | 0.719 | 0.692 | 0.565 |
| Company pool + local IDF | 0.819 | 0.825 | 0.681 |
| Company-year pool + local IDF | 0.915 | 0.919 | 0.681 |

Interpretation:

- Company/year metadata is powerful, but it is mostly a regime switch, not a full solution.
- Local IDF inside the company cluster is the current strongest simple signal.
- TAT-DQA is the hard canary: year gives no additional discrimination, so residual ranking remains
  inside a same-company set.
- This supports a general research framing: entity-clustered numeric corpora require local,
  fact-level disambiguation.

Final retrieval outputs also show a structural signal helps but is not enough:

| Dataset | Dense | Full entity+meta | Full + concept coverage |
|---|---:|---:|---:|
| FinQA | 0.376 | 0.710 | 0.743 |
| ConvFinQA | 0.391 | 0.769 | 0.818 |
| TAT-DQA | 0.235 | 0.401 | 0.455 |

The generation side is the larger bottleneck now. Retrieval can put the right document in top-k,
but Qwen still fails to pick the correct facts and arithmetic without stronger evidence control.

## 5. Proposed Contributions

### C1. Artifact-Controlled Residual Benchmarking

Contribution type: evaluation method and benchmark diagnosis.

Define retrieval difficulty by regimes:

- corpus-level retrieval
- entity-scoped retrieval
- entity-period-scoped retrieval
- within-table / within-section fact retrieval

Report MRR@3 and Number-Match in each regime, with company-masked and year-masked controls. This
turns the metadata concern into a scientific contribution: we quantify exactly what metadata solves
and what remains unsolved.

Why it matters: reviewers can no longer dismiss gains as metadata leakage if the paper's main
claims are on the residual regime after metadata.

### C2. Typed Financial Fact Graph, Not Generic GraphRAG

Contribution type: representation.

Build a graph with nodes:

```text
Fact = (concept, entity, period, value, unit, scale, source, cell provenance)
```

Edges:

- temporal: same concept across periods
- accounting identity: operands to target at a period
- section/table/row/cell hierarchy
- text-to-cell mention links
- unit/scale normalization links

Accounting identities should be used primarily as verifiers and provenance, not as a global ranking
score. Important rules:

- Revenue - CostOfRevenue = GrossProfit
- GrossProfit - OperatingExpenses = OperatingIncome
- PretaxIncome - IncomeTaxExpense = NetIncome
- OperatingCashFlow + InvestingCashFlow + FinancingCashFlow = NetChangeInCash
- TotalAssets = TotalLiabilities + TotalEquity
- TotalDebt = ShortTermDebt + LongTermDebt
- percent change = `(new - old) / abs(old)`
- ratio/margin = `part / total`

Use caveats: identities are conditional on statement type, sign convention, non-operating items, and
whether all operands are present. The graph should verify when conditions are satisfied, not force
every table into a template.

### C3. Channel-Aligned Fact Retrieval Training

Contribution type: retriever learning.

The current zero-shot fact-level semantic stream is not enough. Train a fact-level scorer with
negative samples that break exactly one channel:

- concept/metric swap -> trains concept matching
- period swap -> trains temporal gate
- entity swap -> trains entity gate
- scale break -> trains magnitude/unit channel
- identity break -> trains accounting verifier

The scoring form should be multiplicative, not only additive:

```text
S(q,d) = max_f sigma_concept(q,f) * gamma_entity(q,f) * gamma_period(q,f) * rho_unit_value(q,f)
```

Then combine with local lexical salience rather than replacing it. The paper's strongest retrieval
claim should be improvement over company-pool local IDF, not over dense retrieval.

### C4. Ledger-Grounded Inference-Time Generation Support

Contribution type: generator support and verifiability, without training the LLM.

For each retrieved top-k set, the KG should do as much deterministic work as possible before the
LLM is called:

- arbitrate the top-3 retrieved documents by query-conditioned fact support
- select exact operands from the chosen document(s)
- compute a symbolic answer when the task is lookup/difference/percent-change/ratio/sum/average
- pass the calculation trace and source cells to the LLM
- verify the final answer and any generated intermediate arithmetic

Current inference-time KG output:

```text
KG_SELECTED_DOC
KG_SELECTION_RATIONALE
KG_SYMBOLIC_ANSWER
KG_CALCULATION_TRACE
KG_OPERAND_PROVENANCE
KG_CONFLICTS_IN_TOPK
```

The LLM's job becomes narrow: copy or lightly verbalize the KG-supported result, not parse all raw
markdown tables and invent arithmetic under distractor pressure.

Implementation status:

- `generation/retrieval_bridge.py` now re-ranks/focuses the noisy top-k with KG evidence scores.
- `ledger/select.py::calculation_plan()` computes symbolic answers and operand provenance.
- `generation/verifier.py` returns `grounding_fraction`, `arithmetic_fraction`, and step checks for
  post-hoc audit/evaluation.
- Future optional work: use these deterministic checks as DPO/ORPO/GRPO reward, but this is deferred.

### C5. Selective Evidence Arbitration Over Noisy Top-3

Contribution type: end-to-end KG-for-generator.

MRR@3 means the generator sees three documents, often with at least two plausible distractors. KG
should decide how the generator uses top-k:

- high confidence: give only top-1 facts and provenance
- medium confidence: give top-1 plus conflict checks
- low confidence: give top-2/top-3 with explicit per-doc fact conflicts

Evidence block should include:

- selected facts
- operands and formula
- source cells
- unit/scale
- matched period/entity/concept
- why other top-k docs are weaker

This directly addresses the finance transparency requirement: the answer is not just "LLM said so";
it is grounded in auditable cell-level evidence.

## 6. Experimental Plan

### Retrieval

Baselines:

- BM25, dense, hybrid RRF, hybrid + reranker
- metadata filter + BM25
- company-pool local IDF
- ColBERT/late interaction
- current MMER fusion
- trained fact graph scorer

Required ablations:

- remove entity gate
- remove period gate
- remove magnitude/unit channel
- remove local IDF
- remove concept coverage
- each negative type individually
- oracle fact ledger vs automatic fact ledger

Metrics:

- MRR@3, R@1/3/5, nDCG@3
- pool recall
- fact evidence F1
- latency and token cost

### Generation

Baselines:

- raw top-3 Qwen
- evidence-block Qwen
- calculator/tool prompting
- oracle context
- KG-symbolic answer + LLM copy/check prompt

Metrics:

- Number-Match
- grounded-number rate
- arithmetic consistency rate
- identity-violation rate
- answer-correct but reasoning-wrong rate
- citation/provenance accuracy

### Benchmarks

Primary:

- T2-RAGBench: FinQA, ConvFinQA, TAT-DQA

External:

- DocFinQA for full-document localization
- LOFin for large-scale standardized filings
- FinanceBench for evidence-string finance QA
- MultiHiertt or TAT-QA for table reasoning generality

AAAI-27 needs at least one external benchmark beyond T2-RAGBench. DocFinQA is the best first choice
because it stresses the same within-filing evidence localization problem at full-document scale.

## 7. Immediate Engineering Checklist

1. Implement fact evidence F1: compare selected facts against answer-bearing cells or oracle program
   operands where available.
2. Extend `FinancialFactGraph` with section/table/row/cell nodes and text-to-cell mention edges.
3. Add a trained fact scorer:
   - query encoder + fact concept encoder
   - explicit entity/period/unit gates
   - local lexical salience feature
   - channel-aligned InfoNCE
4. Add generator experiments:
   - Qwen3.5-4B smoke
   - Qwen3.5-9B if memory permits
   - evidence-block vs raw top-3
   - KG-selected doc vs original top-1
   - KG-symbolic answer prompt vs fact-only prompt
5. Create an external benchmark loader for DocFinQA first, then FinanceBench/LOFin.

## 8. Paper Framing

Recommended title direction:

```text
Fact-Ledger RAG: Verifiable Fact-Level Retrieval and Process-Supervised Generation for Financial QA
```

Core claim:

> In financial RAG, the atomic object of relevance is not a document or a generic entity triple,
> but an auditable financial fact `(concept, entity, period, value, unit, provenance)`. A single
> typed fact graph can therefore drive retrieval, training negatives, generator rewards, and
> explanations.

What not to claim:

- Do not claim metadata itself is novel.
- Do not claim generic KG/GNN is the breakthrough.
- Do not claim SOTA solely from company scoping.
- Do not treat accounting identities as universally valid ranking signals.

What to claim if experiments confirm:

- Metadata-controlled residual difficulty is the right evaluation regime.
- Local + fact-level salience beats global document embeddings in entity-clustered financial QA.
- Ledger-derived negatives train clean retrieval channels.
- Deterministic KG-side evidence selection/calculation improves Qwen generation faithfulness and
  Number-Match without training the generator.
- Cell-level provenance yields auditability that standard RAG and black-box LLM reasoning lack.

## 9. Implementation Update: KG-Generator Bridge v4

Date: 2026-06-19

This update supersedes any plan that treats generator training or RL as the active next step.
The current generator direction is inference-time KG support only: the KG selects/focuses the
retrieved top-k, extracts operands, computes when safe, exposes provenance, and lets a frozen
Qwen model copy/check the result.

Implemented rules now include:

- safe `page_content` ledger extraction for TAT-style chunks with text plus multiple markdown tables
- table/text quota so narrative facts supplement tables instead of crowding out cells
- explicit `% CHANGE` / narrative percent facts, but only when a visible percent cue exists
- question-literal numerator ratios, e.g. `$6.9 million / EMEA total`
- temporal comparison answers such as "which year exceeded the previous year"
- query-supported confidence gating for temporal arithmetic and lookup
- extractive generator now copies high-confidence `KG_SYMBOLIC_ANSWER`

### KG Bridge Metrics

| Dataset | original top1 | KG top1 | best policy | symbolic cov. | symbolic NM all | NM when available |
|---|---:|---:|---:|---:|---:|---:|
| FinQA | 0.6417 | 0.6469 | 0.6600 (`margin>=0.15`) | 0.5937 | 0.1395 | 0.2349 |
| ConvFinQA | 0.7279 | 0.7287 | 0.7420 (`margin>=0.25`) | 0.5604 | 0.2120 | 0.3782 |
| TAT-DQA | 0.3260 | 0.3628 | 0.3820 (`rankprior=2.0`) | 0.6154 | 0.0918 | 0.1491 |

### Generator Metrics

Qwen3-4B, frozen, 100-query samples:

| Dataset | rules2 | rules3 | rules4 |
|---|---:|---:|---:|
| FinQA NM | 0.21 | 0.21 | 0.22 |
| TAT-DQA NM | 0.18 | 0.19 | 0.20 |
| TAT-DQA grounded rate | 0.32 | 0.36 | 0.43 |

Full extractive-symbolic baseline:

| Dataset | rules2 NM | rules4 NM | comment |
|---|---:|---:|---|
| FinQA | 0.1229 | 0.1526 | symbolic copy clearly helps |
| ConvFinQA | 0.2296 | 0.2282 | essentially flat, guardrails trade coverage for precision |
| TAT-DQA | 0.1206 | 0.1180 | NM flat/slightly down, but grounded rate rises to 0.455 |

Interpretation: rules4 is the best current inference-time generator bridge for Qwen samples. It is
not "metadata SOTA"; it is a confidence-calibrated typed fact graph that acts as router, calculator,
verifier, and provenance layer. The remaining weakness is not whether KG is useful, but calibration:
lookup/count questions and ambiguous same-company chunks need stronger query-to-fact alignment before
the KG is allowed to emit a symbolic answer.
