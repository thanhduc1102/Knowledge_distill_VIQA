# Retrieval ablation — FinQA (n=300, corpus=2789)

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense_only | 0.381 | 0.300 | 0.483 | 0.547 | 0.407 |
| dense+equationCS | 0.381 | 0.300 | 0.483 | 0.547 | 0.407 |
| dense+entity_rerank | 0.669 | 0.573 | 0.787 | 0.813 | 0.699 |
| FULL (entity+metadata_filter) | 0.732 | 0.617 | 0.873 | 0.933 | 0.768 |