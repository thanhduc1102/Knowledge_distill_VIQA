# Integrated retrieval ablation — FinQA (n=1147, corpus=2789)
_coverage: {'pct_docs_with_canonical_concept': 69.5, 'pct_queries_with_canonical_concept': 39.4}_

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.376 | 0.296 | 0.476 | 0.548 | 0.401 |
| +entity (rerank) | 0.653 | 0.572 | 0.754 | 0.784 | 0.679 |
| FULL (entity+meta-filter) | 0.710 | 0.602 | 0.846 | 0.918 | 0.745 |
| FULL + C3 (δ=0.1) | 0.743 | 0.642 | 0.867 | 0.943 | 0.775 |
| FULL + C3 (δ=0.2) | 0.731 | 0.629 | 0.858 | 0.932 | 0.763 |
| FULL + C3 (δ=0.3) | 0.725 | 0.622 | 0.854 | 0.929 | 0.758 |
| FULL + C3 (δ=0.5) | 0.704 | 0.602 | 0.830 | 0.913 | 0.737 |