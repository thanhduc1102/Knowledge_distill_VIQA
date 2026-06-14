# Final Retrieval Ablation — TAT-DQA  (n=1144, corpus=2723)
_Coverage: {'pct_docs_with_canonical_concept': 78.7, 'pct_queries_with_canonical_concept': 55.0}_
_CACL2 loaded: True_
_CACL2 weights: w_text=1.329, w_ent=1.046, w_cov=0.734_

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.2350 | 0.1827 | 0.3024 | 0.3479 | 0.2523 |
| FULL (entity+meta, β=0.6) | 0.4008 | 0.2867 | 0.5463 | 0.6635 | 0.4382 |
| FULL + C3 δ=0.1 (fixed w) ← **BEST** | 0.4554 | 0.3260 | 0.6198 | 0.7334 | 0.4976 |
| FULL + C3 δ=0.2 (fixed w) | 0.4467 | 0.3164 | 0.6110 | 0.7273 | 0.4889 |
| FULL + C3 δ=0.3 (fixed w) | 0.4432 | 0.3164 | 0.6031 | 0.7194 | 0.4843 |
| FULL + C3 δ=0.5 (fixed w) | 0.4368 | 0.3129 | 0.5944 | 0.7037 | 0.4772 |
| FULL + C3 CACL2-weights | 0.4331 | 0.3103 | 0.5900 | 0.7037 | 0.4734 |