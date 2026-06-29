# Related Work & Literature Store (financial RAG / structure / auditability)

Curated for the AAAI-27 line of work. Two buckets: **local PDFs** (in `hypothesis/`,
`research_paper/`) and **key cited works** (URL-only in project notes). Each entry notes the
*gap it leaves* that our structure-grounded auditable RAG targets.

## A. Local PDFs — directly relevant (`hypothesis/`)

| File | Work | One-line | Gap it leaves for us |
|---|---|---|---|
| 2407.12883v4 | **BRIGHT** (ICLR'25) — realistic reasoning-intensive retrieval benchmark | retrieval where relevance needs reasoning, not surface match | motivates "relevance is reasoning/structure-indexed"; no financial table structure |
| 2503.05037v2 | **Collapse of Dense Retrievers** — short/early/literal biases outrank factual evidence | dense retrievers are biased away from the deciding evidence | direct empirical support for our granularity-dilemma motivation (entity/period/magnitude drowned) |
| 2512.10787v2 | **Replace, Don't Expand** — fixed-budget evidence assembly to fight context dilution in multi-hop RAG | adding context dilutes; curate a fixed evidence budget | supports fact-level (not doc-level) evidence packing; no typed financial graph / verification |
| 2602.03647v1 | **Search-R2** — search-integrated reasoning via actor-refiner | iterative search+reason loop | a generation-time loop; no auditable symbolic substrate or selective answering |

## B. Local PDFs — background (`research_paper/`)

| File | Work | Relevance |
|---|---|---|
| 2104.08663v4 | **BEIR** — heterogeneous zero-shot IR benchmark | evaluation-protocol reference for cross-benchmark generalization |
| 2022.emnlp-main.187 | **CodeRetriever** — contrastive pre-training for code search | contrastive retrieval design reference (negatives) |
| 2024.emnlp-main.402 | Legal case retrieval via synthetic query-candidate pairs | synthetic hard-negative reference (cf. our ledger-perturbation negatives) |
| 2105.00377v1 | **MathBERT** — pretrained model for math formula understanding | numeracy/representation-of-numbers reference |
| 1-s2.0-S1532046421003129 | **CODER** — knowledge-infused cross-lingual medical term embedding | ontology/concept-canonicalization reference (cf. our IFRS/GAAP concept map) |
| 2022.acl-long.297 | **LexGLUE** — legal language understanding benchmark | domain-specialization benchmark reference |

## C. Key cited works (URL-only; to fetch/store full PDFs later)

- **T²-RAGBench** (EACL'26) — retrieval-first financial QA over FinQA/ConvFinQA/TAT-DQA; metric MRR@3 + downstream NumberMatch. Leaderboard: https://t2ragbench.demo.hcds.uni-hamburg.de/ . *Confirmed: directly using gold company/year metadata beats SOTA — i.e. the leaderboard is largely a metadata-exploitation game, not a research contribution. We must NOT hang our contribution on metadata.*
- **HierFinRAG** (MDPI Informatics 13(2):30) — heterogeneous/hierarchical financial evidence + symbolic-neural fusion. https://www.mdpi.com/2227-9709/13/2/30 . *Gap: stops at hierarchical retrieval; does not close the loop to role-aware verification or selective answering.*
- **DocFinQA** (arXiv 2401.06915) — long-document (full 10-K) financial QA. *Gap: long-context retrieval+reasoning; no typed structure-grounded verifier. Dataset cached locally (kensho/DocFinQA).*
- **FinanceBench** (arXiv 2311.11944) — open-book QA over SEC filings with evidence/provenance. *Gap: provenance is LLM/human-judged, not a typed structure graph with concept-period-role checks. Dataset cached locally (PatronusAI/financebench).*
- **"Structure First, Reason Next"** — explicit structure before generation for financial reasoning. *Gap: structure for generation, not for auditable verification + risk-coverage.*
- **Numbers Matter!** (Findings EMNLP'24) — quantity-aware retrieval requiring explicit numeric conditions in the query. *Gap: number-in-query conditioning; not magnitude-disentangled fact grounding.*
- **Granularity Dilemma** (Findings EMNLP'25) — relevance is fact-indexed, drowned in a single doc vector. *Motivation anchor.*
- **Numeracy Gap** (Findings EACL'26) — tokenization destroys numeric magnitude. *Motivation anchor.*
- **ColBERT / ColBERTv2** — late interaction (token MaxSim). *Baseline; not fact/structure typed.*

## E. Reliability / calibration / selective prediction baselines (now central to our study)

These are the signals we compare CPR against on a strong generator (see [../RESULTS.md §4](../RESULTS.md)).

- **Self-consistency** (Wang et al., ICLR'23) — sample k reasoning paths, majority-vote; agreement as a
  confidence proxy. *Cost k×. We use k=5. Our finding: structure (CPR) at 2× Pareto-dominates it at 6×.*
- **Verbalized confidence / P(True)** (Kadavath et al. 2022; Tian et al., EMNLP'23) — ask the model to state
  its probability of being correct. *Strong on a strong model, but misses confident-but-ungrounded errors.*
- **Selective prediction / risk-coverage** (El-Yaniv & Wiener; Geifman & El-Yaniv) — abstain to trade coverage
  for accuracy; AURC metric. *Our deployment framing; CPR feeds the abstention threshold.*
- **Conformal prediction** (Vovk; Angelopoulos & Bates) — distribution-free coverage guarantees. *Planned upgrade
  for the fusion selective-answering head (C4).*
- **Calibration of LLMs for QA** (Si et al.; Mielke et al.) — confidence calibration reference. *CPR is a
  structure-grounded, annotation-free complement to model-internal calibration.*

## F. Findings log — what the 2026-06-29 strong-generator study changed (store, don't re-derive)

1. **Metadata is legitimate, not just an exploit.** company/year are recoverable from the question 87–98%
   ([../RESULTS.md §2](../RESULTS.md)), so metadata-aware BM25 (leaderboard #1's recipe) is a *valid* setting.
   We report BOTH: metadata-aware BM25 **W.Avg 0.747** (SOTA-competitive, simple, reproducible) AND honest
   content-only MMER **0.736**. The earlier note "must not hang the contribution on metadata" still holds for the
   *novelty* (reliability), but the metadata-aware result is a legitimate, clearly-labeled retrieval contribution.
2. **The "CPR is the best reliability signal" claim was a weak-generator artifact.** On Gemini 2.5 Flash,
   self-consistency (0.749) and verbalized confidence (0.802) beat standalone CPR (0.651). Reframed contribution =
   **cost-efficiency** (CPR+verbalized 2× ≥ self-consistency 6×) + **orthogonal error capture** (CPR catches
   +9.5–16.5% confident hallucinations model-internal misses) + **generator-strength spectrum** (CPR essential
   when weak, complementary when strong).
3. **Verify-then-reask is net-negative on strong models** → CPR is for abstention, not answer override.
4. **System ceilings quantified:** extraction certifiable 0.45–0.80 (2op→3op); role-operand selection F1 ~0.5 vs
   a Gemini oracle → next lever is *learned operand attribution*, not re-weighting (re-weighting CPR did not help).

## G. Findings log — 2026-06-29 (round 2: retrieval boost, operand learning, ontology, DocFinQA)

5. **Retrieval pushed to W.Avg 0.798 (honest).** Adding a question-derived `meta` expert as an 8th fusion column
   (MMER 8-expert) → FinQA 0.846 / ConvFinQA 0.862 / TAT 0.554. **Beats T²-RAGBench #1 on ConvFinQA** (0.862 vs
   0.845) and approaches #1 overall (~0.82) with NO frontier LLM. (`scripts/modular_retrieval.py … ,meta`.)
6. **Learned operand attribution works (the #1 lever).** Distant supervision from the gold answer (operand set
   reconstructing it) + bge-small features + logistic → **derivation hit-rate 2–3× the heuristic** (FinQA .50 vs
   .16, ConvFinQA .85 vs .56, TAT .58 vs .25). Cracks the "ambiguous multi-operand attribution" open problem.
   (`scripts/research/learned_operand_attribution.py`.)
7. **Financial ontology source = XBRL US-GAAP (FASB).** Machine-readable taxonomy at
   http://xbrl.fasb.org/us-gaap/2025/elts/us-gaap-2025.xsd (ASC/SEC EDGAR-derived). We curated 42→80 concepts;
   line-item coverage 14%→22–31%. Deeper annotation-free direction: a learned concept encoder over line-item strings.
8. **DocFinQA loader fixed by streaming.** kensho/DocFinQA full-load times out (123K words/doc); `datasets`
   `streaming=True` + islice caches a slice (`fetch_docfinqa.py`). Long-document end-to-end (chunk→BM25→Gemini→CPR):
   same pattern as T²-RAGBench (CPR 0.528 > value-only 0.427; verbalized 0.645 > CPR) → generator-strength /
   cost-efficiency finding **generalizes to long documents**.

### Financial data sources discovered (for ontology + future benchmarks)
- **XBRL US-GAAP taxonomy** (FASB/SEC) — canonical concept list, machine-readable .xsd. https://xbrl.us/ , https://www.fasb.org/xbrl
- **DocFinQA** (kensho/DocFinQA, ACL'24) — long-document FinQA; acquired via streaming.
- Other domain-finance QA worth adding for breadth: **MultiHiertt** (multi-table hierarchical), **TAT-QA/TAT-DQA**,
  **FinQA/ConvFinQA**, **FinanceBench**, **BizBench**, **FinDER** — candidates for cross-benchmark generalization.

## H. Findings log — 2026-06-29 (round 3: metadata SOTA, rerank, soft-CPR, concept encoder, linkage)

9. **Metadata-aware (provided) ⊕ MMER = NEW SOTA.** MMER 8-expert with provided company+year (sector via entity/GICS)
   → FinQA 0.914 / ConvFinQA 0.932 / TAT 0.653, **W.Avg 0.873 > leaderboard #1 (~0.82)** with no frontier LLM. Reproduces
   (and exceeds) the prior unsaved SOTA. Honest setting (meta-from-question) = 0.798. (`--meta-provided`.)
10. **`company_sector` is redundant once company is exact** (3field ≡ company+year) but useful standalone (sector_only
    > BM25, esp. TAT +0.10 where year doesn't discriminate). "Use all 3" is realised: company+year in `meta`, sector in entity.
11. **Retrieval is causal for the output.** Gold doc at rank-1 vs absent → Number-Match **+0.34..+0.54** (`retrieval_nm_linkage`).
    Justifies the retrieval push; explains low TAT NM (109/300 queries have gold outside top-3).
12. **Generic cross-encoder rerank FAILS on financial tables.** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reimers & Gurevych;
    Nogueira & Cho monoBERT-style) trained on MS-MARCO web passages **destroys** ranking (bm25 0.65→0.28; meta 0.88→0.37) —
    table/number text is out-of-distribution. ⇒ rerankers must be **fine-tuned in-domain**; MMER's *learned in-domain*
    fusion is the right design. (Cite: Reimers & Gurevych EMNLP'19 sentence-transformers; Nogueira & Cho 2019 passage rerank.)
13. **Learned operand attribution → soft-weight CPR is neutral.** Down-weighting (not dropping) fixes the hard-restrict
    catastrophe (back to ≈ full CPR) but doesn't exceed it: the attribution signal is *redundant* with CPR's concept/period/role
    typing for reliability ranking. Its real payoff is the derivation/ceiling regime (deriv-hit 2–3×).
14. **Learned concept encoder** (bge-small anchors, annotation-free): intrinsic alias→concept top-1 **0.688** (80 classes);
    semantic coverage extends 22–31% (exact) → ~98% (τ=0.55) at ~0.69 precision. Generalises to unseen line-item phrasings;
    fine-tuning a contrastive head is the precision lever. (Cite: CODER medical concept embedding — §B — as ontology-encoder analogue.)

## D. The gap we fill (one paragraph)

Existing financial RAG either (i) retrieves documents — coarse, metadata-driven, does not
generalize and does not help reasoning; or (ii) builds generic GraphRAG entity graphs — not
suited to numeric tables; or (iii) uses structure only to *help generation* (HierFinRAG,
"structure first"). **No prior work uses a typed financial structure graph as a shared
substrate that closes the loop to *concept–period–role (CPR) aware verification* and
*selective answering*, and demonstrates that this structure-level grounding — not document
retrieval or metadata — is what makes numeric answers reliably auditable and *transfers*
across heterogeneous datasets (FinQA/ConvFinQA/TAT-DQA) and to OOD settings (FinanceBench,
DocFinQA).** That is our contribution axis.
