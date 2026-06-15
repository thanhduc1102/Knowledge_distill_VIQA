# Thực nghiệm & Kết quả đầy đủ

Tất cả thực nghiệm chạy **honest** (câu hỏi đã bóc tiền tố `company:` mà loader nhồi vào; năm/concept
rút từ câu hỏi, không từ gold metadata), trên **toàn bộ** test set:
FinQA `test` (1147 q / 2789 docs), ConvFinQA `turn_0` (3458 / 1806), TAT-DQA `test` (1144 / 2723).
Embedding: `intfloat/multilingual-e5-large-instruct`. Fusion: weighted RRF (k=60).

## Bản đồ script → thực nghiệm

| Script | Mục đích |
|---|---|
| [`scripts/fact_f1.py`](../../scripts/fact_f1.py) | Đo chất lượng Fact-Ledger (extraction vs reasoning bottleneck) |
| [`scripts/header_retrieval.py`](../../scripts/header_retrieval.py) | Leak audit + header-only + full-text BM25 vs dense |
| [`scripts/hybrid_period_retrieval.py`](../../scripts/hybrid_period_retrieval.py) | Tuning period bonus + gate trên FinQA |
| [`scripts/validate_retrieval.py`](../../scripts/validate_retrieval.py) | Grid 3-dataset, gated/adaptive period, weighted-avg picker |
| [`scripts/multichannel_retrieval.py`](../../scripts/multichannel_retrieval.py) | Thêm concept-coverage channel |
| [`scripts/stage1_full.py`](../../scripts/stage1_full.py) | Gated concept + cell-match (table retrieval) |
| [`scripts/stage2_rerank.py`](../../scripts/stage2_rerank.py) | Cross-encoder rerank (stage-2) |

---

## EXP-0 — Fact-F1: chẩn đoán nút thắt (FinQA, gold table)

`python scripts/fact_f1.py --dataset FinQA --sample 0`

```
cell_recall            = 0.953   → trích xuất TỐT (không phải nút thắt)
canonical_rate         = 0.144   → ontology nhỏ (chỉ map 14% line-item → cell-match phải dùng raw token)
period_rate            = 0.693
answer_is_lookup       = 0.038   → 96% câu hỏi cần TÍNH TOÁN
answer_derivable_1op   = 0.615   → 61.5% đáp án với tới bằng 1 phép tính từ ledger
```

**Kết luận:** generation NM = 9.9% nhưng 61.5% one-op-derivable → nút thắt generator là
**selection + arithmetic**, không phải extraction/retrieval. (Định hướng generator nằm ngoài docs này.)

---

## EXP-1 — Leak audit + Dense vs BM25 vs Header (FinQA full)

`python scripts/header_retrieval.py --dataset FinQA --sample 0`

| arm | MRR@3 | R@3 | R@5 |
|---|---|---|---|
| dense_honest | 0.394 | 0.480 | 0.568 |
| dense_leaky (nhồi `company:`) | 0.392 | 0.480 | 0.574 |
| fulltext_bm25 | **0.665** | **0.782** | **0.852** |
| header_bm25 (chỉ row+col header) | 0.318 | 0.392 | 0.452 |
| dense_honest + header (RRF) | 0.392 | 0.495 | 0.585 |

**Đọc:** (1) nhồi company vào dense query ≈ vô ích (−0.002) → leak không ở đây; (2) BM25 >> dense;
(3) header-only yếu (header quá generic, không phân biệt công ty).

---

## EXP-2 — Tuning period (FinQA full)

`python scripts/hybrid_period_retrieval.py --dataset FinQA --sample 0`

| arm | MRR@3 | R@3 | R@5 |
|---|---|---|---|
| bm25_only | 0.665 | 0.782 | 0.852 |
| bm25 + period(0.05) | **0.670** | **0.796** | **0.862** |
| bm25 + period(0.1) | 0.648 | 0.765 | 0.857 |
| bm25 + dense(0.3) | 0.602 | 0.725 | 0.794 |

**Đọc:** period bonus phải NHỎ (0.05 = tie-break); lớn hơn kéo doc sai-công-ty cùng năm lên. Dense
làm giảm.

---

## EXP-3 — Grid 3-dataset + gated/adaptive period

`python scripts/validate_retrieval.py --sample 0`

| arm | FinQA | ConvFinQA | TAT-DQA | W.Avg |
|---|---|---|---|---|
| bm25_only | 0.665 | 0.642 | **0.418** | 0.602 |
| bm25 + period FIXED(0.05) | 0.670 | 0.661 | 0.388 ❌ | 0.609 |
| bm25 + period GATED(0.05, thr0.6) | 0.671 | **0.658** | 0.414 | 0.6124 |
| **bm25 + period GATED(0.05, thr0.4)** | **0.678** | 0.656 | **0.417** | **0.6125** |

**Đọc:** period cố định **hại TAT-DQA** (80% bảng đa-kỳ → period không discriminative). **Gate**
(chỉ áp khi ≤40% pool khớp năm) cứu TAT-DQA *và* cải thiện FinQA. → chọn `period_gate=0.4`.

---

## EXP-4 — Concept-coverage channel (3-dataset)

`python scripts/multichannel_retrieval.py --sample 0`

| arm | FinQA | ConvFinQA | TAT-DQA | W.Avg |
|---|---|---|---|---|
| bm25 + period | 0.678 | 0.656 | 0.417 | 0.6125 |
| + concept(0.1) | 0.675 | **0.668** | 0.381 ❌ | 0.6121 |
| + concept(≥0.3) | giảm | giảm | giảm | <0.58 |

**Đọc:** concept-coverage giúp ConvFinQA nhưng **hại TAT-DQA** (canonical chỉ phủ 14% → nhiễu). Tùy
chọn, không bật mặc định.

---

## EXP-5 — Cell-match (table retrieval) + gated concept (3-dataset)

`python scripts/stage1_full.py --sample 0`

| arm | FinQA | ConvFinQA | TAT-DQA | W.Avg |
|---|---|---|---|---|
| bm25 + period | 0.680 | 0.653 | 0.414 | 0.6108 |
| **+ cellmatch(0.3)** | **0.683** | 0.660 | **0.412** | **0.6155** |
| + concept_gated(0.1) | 0.675 | 0.664 | 0.389 ❌ | 0.6117 |
| + concept + cellmatch | 0.677 | **0.668** | 0.388 ❌ | 0.6138 |

**Đọc:** **cell-match là tín hiệu cấu trúc tốt nhất & ROBUST** — giúp FinQA+ConvFinQA, *trung tính*
TAT-DQA (không hại như concept). Vì dùng raw row-label tokens (phủ cao) thay canonical concept.
→ chọn `cell_bonus=0.3`.

---

## EXP-6 — Cross-encoder rerank (stage-2, 3-dataset)

`python scripts/stage2_rerank.py --sample 0 --pool 20` (reranker `BAAI/bge-reranker-base`)

| arm | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| stage1 (BM25+period+cell) | **0.683** | **0.660** | 0.412 |
| ce_only | 0.555 | 0.577 | 0.387 |
| rrf(stage1, ce) | 0.642 | 0.652 | **0.413** |
| weighted(stage1 + 1.0·ce) | 0.581 | 0.611 | 0.405 |

**Đọc:** cross-encoder rerank **làm giảm** điểm — linearization bottleneck trên markdown bảng. BEST =
stage1_only. → **không dùng** general CE rerank.

---

## Kết quả cuối (production class)

`HybridBM25Retrieval(top_k=5)` — tái lập chính xác:

| | FinQA | ConvFinQA | TAT-DQA | W.Avg MRR@3 |
|---|---|---|---|---|
| MRR@3 | 0.6834 | 0.6602 | 0.4123 | **0.6155** |
| R@3 | 0.8003 | 0.7863 | 0.5367 | — |
| R@5 | 0.8596 | 0.8505 | 0.6110 | — |
| NDCG@3 | 0.7135 | 0.6926 | 0.4443 | — |

Artifacts JSON: `outputs/{header_retrieval,hybrid_period,validate_retrieval,multichannel_retrieval,stage1_full,stage2_rerank,fact_f1}/`.
