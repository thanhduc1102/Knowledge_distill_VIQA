# Entity-channel ablation — FinQA (n=1147, corpus=2789)

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.376 | 0.296 | 0.476 | 0.548 | 0.401 |
| dense + hash-entity (rerank) | 0.651 | 0.570 | 0.749 | 0.783 | 0.676 |
| dense + ontology-entity (rerank) | 0.653 | 0.572 | 0.754 | 0.784 | 0.679 |
| FULL hash (exact filter) | 0.712 | 0.605 | 0.843 | 0.918 | 0.745 |
| FULL ontology (alias filter, E1+E2) | 0.710 | 0.602 | 0.846 | 0.918 | 0.745 |