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
