# Integrated retrieval ablation — ConvFinQA (n=3458, corpus=1806)
_coverage: {'pct_docs_with_canonical_concept': 72.9, 'pct_queries_with_canonical_concept': 43.0}_

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.390 | 0.301 | 0.505 | 0.595 | 0.420 |
| +entity (rerank) | 0.722 | 0.642 | 0.816 | 0.843 | 0.746 |
| FULL (entity+meta-filter) | 0.769 | 0.668 | 0.892 | 0.944 | 0.800 |
| FULL + C3 (δ=0.1) | 0.818 | 0.728 | 0.927 | 0.965 | 0.846 |
| FULL + C3 (δ=0.2) | 0.810 | 0.720 | 0.921 | 0.962 | 0.839 |
| FULL + C3 (δ=0.3) | 0.806 | 0.715 | 0.916 | 0.958 | 0.834 |
| FULL + C3 (δ=0.5) | 0.787 | 0.700 | 0.895 | 0.942 | 0.815 |