# Entity-channel ablation — TAT-DQA (n=1144, corpus=2723)

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.235 | 0.183 | 0.302 | 0.348 | 0.252 |
| dense + hash-entity (rerank) | 0.362 | 0.270 | 0.476 | 0.532 | 0.392 |
| dense + ontology-entity (rerank) | 0.362 | 0.270 | 0.476 | 0.532 | 0.392 |
| FULL hash (exact filter) | 0.401 | 0.287 | 0.546 | 0.663 | 0.438 |
| FULL ontology (alias filter, E1+E2) | 0.401 | 0.287 | 0.546 | 0.663 | 0.438 |