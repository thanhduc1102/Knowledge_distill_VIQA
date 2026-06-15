# Integrated retrieval ablation — TAT-DQA (n=1144, corpus=2723)
_coverage: {'pct_docs_with_canonical_concept': 78.7, 'pct_queries_with_canonical_concept': 55.0}_

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.235 | 0.183 | 0.302 | 0.348 | 0.252 |
| +entity (rerank) | 0.362 | 0.270 | 0.476 | 0.532 | 0.392 |
| FULL (entity+meta-filter) | 0.401 | 0.287 | 0.546 | 0.663 | 0.438 |
| FULL + C3 (δ=0.1) | 0.455 | 0.326 | 0.620 | 0.733 | 0.498 |
| FULL + C3 (δ=0.2) | 0.447 | 0.316 | 0.611 | 0.727 | 0.489 |
| FULL + C3 (δ=0.3) | 0.443 | 0.316 | 0.603 | 0.719 | 0.484 |
| FULL + C3 (δ=0.5) | 0.437 | 0.313 | 0.594 | 0.704 | 0.477 |