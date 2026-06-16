# MMER — kết quả: experts độc lập + fusion học được

Honest (bóc prefix `company:`; year/company/concept rút từ câu hỏi). Pool = BM25 top-50
(không nhồi gold). Eval trên **held-out test split** (50% query); fusion huấn luyện trên
50% còn lại. Script: [`scripts/modular_retrieval.py`](../../scripts/modular_retrieval.py).

## Mỗi method đo ĐỘC LẬP, rồi fusion học được

### FinQA (test split = 574 q | pool recall 0.981)

| method | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| lexical (BM25+abbr) | 0.660 | 0.551 | 0.788 | 0.861 | 0.693 |
| entity (ontology+SupCon) | 0.299 | 0.183 | 0.458 | 0.594 | 0.339 |
| concept (C2/C3 coverage) | 0.049 | 0.031 | 0.075 | 0.115 | 0.055 |
| cell (Fact-Ledger) | 0.289 | 0.200 | 0.411 | 0.511 | 0.320 |
| **fusion: linear** | **0.774** | 0.667 | 0.901 | 0.936 | 0.807 |
| fusion: mlp | 0.773 | 0.666 | 0.899 | 0.939 | 0.805 |
| fusion: gate | 0.764 | 0.660 | 0.889 | 0.925 | 0.796 |

`linear weights`: lexical 0.649 · entity 0.579 · cell 0.187 · concept 0.096.

**Đọc:** không expert đơn lẻ nào > 0.66, nhưng **fusion học được đạt 0.774** (+0.114 so với
BM25 đơn, honest, không leak). Entity (0.30) và cell (0.29) đơn lẻ yếu nhưng *bổ sung* cho
lexical — fusion khai thác đúng phần bù đó. Trọng số linear khớp trực giác: lexical & entity
là hai trụ, cell/concept là gia vị. (Đây đúng mô hình `JointScorer` thế hệ 1, tổng quát hóa.)

### ConvFinQA (test split = 1729 q | pool recall 0.972)

| method | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| lexical (BM25+abbr) | 0.647 | 0.547 | 0.773 | 0.840 | 0.679 |
| entity (ontology+SupCon) | 0.426 | 0.303 | 0.586 | 0.709 | 0.467 |
| concept | 0.156 | 0.096 | 0.235 | 0.329 | 0.176 |
| cell | 0.339 | 0.263 | 0.434 | 0.521 | 0.363 |
| fusion: linear | 0.766 | 0.666 | 0.883 | 0.932 | 0.796 |
| fusion: mlp | 0.766 | 0.667 | 0.883 | 0.931 | 0.796 |
| **fusion: gate** | **0.770** | 0.678 | 0.880 | 0.931 | 0.799 |

`linear weights`: **entity 0.767** · lexical 0.567 · cell 0.196 · concept 0.098.
Entity là trụ MẠNH NHẤT ở ConvFinQA (multi-turn cùng công ty → disambiguation thực thể quan
trọng nhất) — fusion tự học điều này, khác hẳn FinQA nơi lexical mạnh hơn.

### TAT-DQA (test split = 572 q | pool recall 0.886)

| method | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| lexical (BM25+abbr) | 0.429 | 0.327 | 0.559 | 0.642 | 0.463 |
| entity | 0.137 | 0.079 | 0.217 | 0.323 | 0.157 |
| concept | 0.008 | 0.002 | 0.018 | 0.032 | 0.010 |
| cell | 0.143 | 0.093 | 0.213 | 0.285 | 0.161 |
| fusion: linear | 0.472 | 0.344 | 0.635 | 0.734 | 0.514 |
| fusion: mlp | 0.484 | 0.358 | 0.643 | 0.733 | 0.525 |
| **fusion: gate** | **0.494** | 0.365 | 0.656 | 0.734 | 0.535 |

⚠️ **Pool recall chỉ 0.886** — BM25 top-50 đã bỏ sót ~11% gold → đây là TRẦN của TAT-DQA.
Để vượt cần cải thiện *pool* (recall), không phải fusion: union dense / late-interaction.

## Tổng hợp — best fusion vs từng method (test split, honest)

| | FinQA | ConvFinQA | TAT-DQA | **W.Avg** |
|---|---|---|---|---|
| BM25+abbr (best standalone) | 0.660 | 0.647 | 0.429 | 0.612 |
| **MMER fusion (best head)** | **0.774** | **0.770** | **0.494** | **0.716** |
| Δ | +0.114 | +0.123 | +0.065 | **+0.104** |

W.Avg theo số query test-split (574/1729/572). So với Phase A full-set 0.6176, MMER nâng
honest W.Avg lên **~0.716** (+0.10). `gate` thắng 2/3 dataset (ConvFinQA, TAT-DQA); `linear`
ngang `mlp` ở FinQA. Gate = mixture-of-experts có điều kiện query, *học* được quy tắc
discriminative-gating thay vì đặt tay.


## Vì sao đây là cách làm đúng (so với "xếp chồng" Phase A)
- **Độc lập:** mỗi expert là một class riêng, biểu diễn riêng, đo MRR@3 standalone — nhìn rõ
  đóng góp thực của từng phương pháp (không bị che bởi cộng dồn bonus).
- **Kết hợp học được:** trọng số do InfoNCE listwise học, không phải hệ số tay → không over-fit
  một dataset; tự cân bằng tín hiệu yếu/mạnh.
- **Mở rộng (continual):** thêm expert (vd LateInteraction ColBERT fact-level) = thêm 1 cột vào
  ma trận + huấn luyện lại đầu fusion nhỏ; experts cũ giữ nguyên.
