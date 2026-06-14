# Final Retrieval Ablation — FinQA  (n=1147, corpus=2789)
_Coverage: {'pct_docs_with_canonical_concept': 69.5, 'pct_queries_with_canonical_concept': 39.4}_
_CACL2 loaded: True_
_CACL2 weights: w_text=1.334, w_ent=1.058, w_cov=0.727_

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| dense | 0.3756 | 0.2964 | 0.4760 | 0.5475 | 0.4014 |
| FULL (entity+meta, β=0.6) | 0.7104 | 0.6024 | 0.8457 | 0.9180 | 0.7452 |
| FULL + C3 δ=0.1 (fixed w) ← **BEST** | 0.7432 | 0.6417 | 0.8675 | 0.9433 | 0.7752 |
| FULL + C3 δ=0.2 (fixed w) | 0.7306 | 0.6286 | 0.8579 | 0.9320 | 0.7633 |
| FULL + C3 δ=0.3 (fixed w) | 0.7254 | 0.6225 | 0.8535 | 0.9285 | 0.7583 |
| FULL + C3 δ=0.5 (fixed w) | 0.7043 | 0.6016 | 0.8300 | 0.9128 | 0.7367 |
| FULL + C3 CACL2-weights | 0.7194 | 0.6199 | 0.8413 | 0.9215 | 0.7508 |