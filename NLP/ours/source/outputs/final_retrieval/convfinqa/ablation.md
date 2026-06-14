# Final Retrieval Ablation — ConvFinQA  (n=3458, corpus=1806)
_Coverage: {'pct_docs_with_canonical_concept': 72.9, 'pct_queries_with_canonical_concept': 43.0}_
_CACL2 loaded: True_
_CACL2 weights: w_text=1.373, w_ent=1.104, w_cov=0.703_

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.3905 | 0.3013 | 0.5055 | 0.5946 | 0.4200 |
| FULL (entity+meta, β=0.6) | 0.7686 | 0.6680 | 0.8918 | 0.9439 | 0.8003 |
| FULL + C3 δ=0.1 (fixed w) ← **BEST** | 0.8176 | 0.7279 | 0.9274 | 0.9647 | 0.8459 |
| FULL + C3 δ=0.2 (fixed w) | 0.8105 | 0.7201 | 0.9213 | 0.9618 | 0.8390 |
| FULL + C3 δ=0.3 (fixed w) | 0.8057 | 0.7154 | 0.9164 | 0.9584 | 0.8342 |
| FULL + C3 δ=0.5 (fixed w) | 0.7868 | 0.6998 | 0.8947 | 0.9422 | 0.8145 |
| FULL + C3 CACL2-weights | 0.8017 | 0.7221 | 0.8973 | 0.9479 | 0.8264 |