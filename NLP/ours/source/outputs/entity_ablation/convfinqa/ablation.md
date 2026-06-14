# Entity-channel ablation — ConvFinQA (n=3458, corpus=1806)

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.390 | 0.301 | 0.505 | 0.595 | 0.420 |
| dense + hash-entity (rerank) | 0.721 | 0.642 | 0.815 | 0.844 | 0.745 |
| dense + ontology-entity (rerank) | 0.722 | 0.642 | 0.816 | 0.843 | 0.746 |
| FULL hash (exact filter) | 0.767 | 0.667 | 0.891 | 0.945 | 0.799 |
| FULL ontology (alias filter, E1+E2) | 0.769 | 0.668 | 0.892 | 0.944 | 0.800 |