# Structure-Level KG Method for AAAI-27

## Motivation from related work

The literature around financial RAG and long-document QA points to the same gap:

- **HierFinRAG** uses a heterogeneous/hierarchical view of financial evidence and a
  symbolic-neural fusion idea: raw dense retrieval is not enough for table-heavy financial
  QA.
- **DocFinQA** stresses the long-document setting: relevant evidence can be buried in long
  10-K-style filings, so the system must reason over document sections/tables rather than
  isolated chunks.
- **FinanceBench** evaluates open-book financial QA over SEC filings with evidence, making
  provenance and auditable evidence central rather than optional.
- Recent KG-for-finance work such as "Structure First, Reason Next" argues that financial
  reasoning benefits from explicit structure before generation.

The method in this repository follows the same direction but makes the graph more specific:
we do **not** build a generic entity-relation KG.  We build a typed **Financial Structure
Graph** whose nodes and edges correspond to the actual objects a financial QA system must
audit: document, table, row, column, cell/fact, concept, period, temporal relation, and
accounting-support relation.

Reference links used in project notes:

- HierFinRAG: https://www.mdpi.com/2227-9709/13/2/30
- DocFinQA: https://arxiv.org/html/2401.06915v3
- FinanceBench: https://arxiv.org/abs/2311.11944
- T2-RAGBench: https://arxiv.org/pdf/2506.12071

## Proposed full-stack method

### 1. Structure graph construction

For each retrieved financial document/snippet, build:

- `document` node
- `table` node
- `row` nodes from line items
- `column` nodes from period/value headers
- `fact`/cell nodes containing value, raw cell text, scale, unit, row index, column index
- `concept` nodes from canonical financial ontology
- `period` nodes from year/period headers

Edges:

- `document -> table`
- `table -> row`
- `table -> column`
- `row -> fact`
- `column -> fact`
- `fact -> concept`
- `fact -> period`
- `fact -> fact` temporal same-concept edge
- `fact -> fact` accounting-identity support edge when available

This graph is implemented in:

- `src/gsr_cacl/kg/structure_graph.py`

### 2. Structure-aware retrieval arbitration

The retriever still provides top-k candidates.  The KG layer does not replace retrieval; it
performs **evidence arbitration**:

`score(doc, query) = retrieval-rank prior + fact support + structure support`

Structure support measures:

- concept coverage;
- period coverage;
- row-column alignment;
- temporal affordance for difference / percent-change / comparison;
- arithmetic affordance for ratio / sum / average.

This directly targets the actual financial QA failure: two snippets can share company and
lexical terms, but only one has the correct row-column-period structure needed to answer.

### 3. Graph evidence paths for LLM support

Instead of passing only selected facts, the bridge now emits paths like:

`document -> table -> row[net revenue] -> col[2019] -> value[$5829]`

These paths make the context auditable and reduce ambiguity for the LLM.  They are now wired
into `generation/retrieval_bridge.py` under `STRUCTURE_EVIDENCE_PATHS`.

### 4. Verification and re-ask

Generation policy remains conservative:

1. Let the LLM answer from raw table/context.
2. Verify answer against the structure/fact graph.
3. If ungrounded, re-ask with graph evidence paths and selected facts.

This avoids the earlier failure mode where filtered KG evidence removed information that the
LLM could have used from the raw table.

## Direct evaluation

### Structure graph arbitration only

| Dataset | Original top1 | Structure-only top1 | Best structure policy |
|---|---:|---:|---:|
| FinQA | 0.6417 | 0.5902 | 0.6504 |
| ConvFinQA | 0.7279 | 0.6486 | 0.7377 |
| TAT-DQA | 0.3260 | 0.3663 | 0.3767 |

Structure-only is not a universal replacement for retrieval.  But when combined with a rank
prior, it improves top1 on all three datasets.  TAT-DQA benefits most because its raw top1 is
weak and table structure matters more.

### KG bridge after structure integration

| Dataset | Original top1 | Best gated KG/structure top1 | Delta |
|---|---:|---:|---:|
| FinQA | 0.6417 | 0.6617 | +0.0201 |
| ConvFinQA | 0.7279 | 0.7420 | +0.0142 |
| TAT-DQA | 0.3260 | 0.3829 | +0.0568 |

This is the strongest evidence for making KG/structure-level the main contribution: the
graph is not merely a post-hoc verifier; it can focus the right document among noisy top-k,
especially on the structure-heavy TAT-DQA setting.

## What this contributes beyond loclex

`loclex` solves conditional lexical salience inside a company/document pool.  The structure
graph adds an orthogonal signal: whether the candidate has the **right table shape and
reasoning affordance** for the question.

Examples:

- percent-change questions need same concept across two periods;
- ratio questions need part/total operands in the same local table context;
- lookup questions need row-concept and column-period alignment;
- comparison questions need temporal or cross-row structure.

## Claims that are now defensible

1. Financial QA benefits from a typed structure graph, not a generic entity KG.
2. Structure graph arbitration improves top-k focus when used with confidence/rank gating.
3. Structure evidence paths make LLM context more auditable than raw snippets alone.
4. The graph should be used across the full pipeline: retrieval arbitration, evidence
   planning, verification, and re-ask.

## Remaining research risk

- Structure score still has hand-designed weights.  A future version should learn the
  scorer from query/document labels while preserving interpretability.
- TAT-DQA still has low top3 recall, so graph arbitration cannot rescue documents absent
  from top-k.
- Full-document evaluation is still needed for a stronger long-document claim.
