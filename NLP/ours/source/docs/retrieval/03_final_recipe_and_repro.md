# Công thức cuối, lý do hệ số & cách tái lập

## 1. Công thức retrieval tối ưu

```
Cho mỗi query q (honest — chỉ dùng text câu hỏi):
  1. BM25 → lấy top-50 candidate pool, gán điểm RRF:  s(d) = 1/(k + rank_bm25(d) + 1),  k=60
  2. METADATA (period), gated:
       q_years = các năm regex từ q
       matchers = candidate có kỳ báo cáo ∩ q_years
       NẾU |matchers|/|pool| ≤ period_gate(0.4):   s(d) += period_bonus(0.05)/k  ∀ d∈matchers
  3. TABLE (cell-level), gated:
       q_content = token nội dung của q (bỏ stopword/year)
       cm(d) = max over cells của d  [ jaccard(q_content, tokens(row_label)) × period_factor ]
               period_factor = 1.0 nếu kỳ cell ∈ q_years; 0.6 nếu doc có q_year; 0.3 còn lại
       hits = candidate có cm(d) ≥ cell_min(0.34)
       NẾU |hits|/|pool| ≤ gate(0.4):   s(d) += cell_bonus(0.3) × cm(d)/k  ∀ d∈hits
  4. Trả top-k theo s(d)
```

Triển khai: [`gsr_cacl.methods.HybridBM25Retrieval`](../../src/gsr_cacl/methods/hybrid_bm25_retrieval.py).

## 2. Lý do từng hệ số (đều từ ablation, không phải đoán)

| Hệ số | Giá trị | Lý do |
|---|---|---|
| `candidate_pool` | 50 | đủ phủ gold (R@50 cao); pool nhỏ hơn cắt mất recall, lớn hơn không thêm |
| `rrf_k` | 60 | chuẩn document-level RRF (văn liệu); single-channel BM25 nên k chỉ là hằng tỉ lệ |
| `period_bonus` | 0.05 | EXP-2: 0.05 là tie-break tối ưu; ≥0.1 kéo doc sai-công-ty cùng năm lên (giảm điểm) |
| `period_gate` | 0.4 | EXP-3: gate cứu TAT-DQA (period không discriminative khi pool đa-kỳ) & cải thiện FinQA |
| `cell_bonus` | 0.3 | EXP-5: 0.3 tốt nhất; cao hơn bắt đầu lấn át BM25 |
| `cell_min` | 0.34 | ngưỡng coi là "hit" (≈ jaccard 1/3 — overlap đáng kể giữa câu hỏi và row-label) |
| `gate` (cell) | 0.4 | dùng chung cơ chế gate discriminative như period |

**Nguyên tắc chung — DISCRIMINATIVE GATING.** Mọi tín hiệu cấu trúc (period, cell) chỉ được áp khi nó
*phân biệt* trong pool (≤ gate tỉ lệ candidate khớp). Đây là phát hiện then chốt: cùng một bonus cố
định giúp FinQA/ConvFinQA nhưng hại TAT-DQA; gate hóa giải mâu thuẫn này, cho một cấu hình **robust
chung cho cả 3 dataset**.

## 3. Honesty contract (chống leak)

- Năm & nội dung dùng cho period/cell-match **rút từ câu hỏi**, không từ field `report_year`/
  `company_name` vàng.
- `query_meta` được nhận trong API cho tương thích interface nhưng **không tham gia ranking**.
- Khi đo, phải **bóc tiền tố `company:`** mà `wrappers._build_queries` nhồi vào (xem §5).

## 4. Cách dùng

```python
from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.methods import HybridBM25Retrieval   # hoặc GSR_REGISTRY["hybrid_bm25"]

data = load_t2ragbench_split("FinQA", split="test")
retr = HybridBM25Retrieval(data.corpus, top_k=3)
docs = retr.retrieve("What was the gross profit in 2019?")          # honest query
# docs = retr.retrieve_with_scores(...)  / retr.retrieve_batch(...)
```

Tham số có thể override: `period_bonus, period_gate, cell_bonus, cell_min, gate, candidate_pool`.

## 5. Tái lập toàn bộ bảng kết quả

```bash
cd ours/source
# Chẩn đoán & từng họ tín hiệu
python scripts/fact_f1.py            --dataset FinQA --sample 0
python scripts/header_retrieval.py   --dataset FinQA --sample 0
python scripts/hybrid_period_retrieval.py --dataset FinQA --sample 0
# Validate 3-dataset & chốt hệ số
python scripts/validate_retrieval.py   --sample 0    # gated/adaptive period grid
python scripts/multichannel_retrieval.py --sample 0  # concept-coverage channel
python scripts/stage1_full.py          --sample 0    # cell-match (winner)
python scripts/stage2_rerank.py        --sample 0 --pool 20   # cross-encoder (negative result)
```

Mỗi script ghi JSON vào `outputs/<tên>/`. `--sample N` để chạy nhanh N query khi debug.

> **Lưu ý môi trường:** cần `faiss-cpu`, `rank_bm25`, `sentence-transformers`; model e5-large &
> bge-reranker-base tải tự động từ HF (lần đầu ~2GB + ~1GB).

## 6. Hạn chế đã biết & hướng mở rộng

- **TAT-DQA vẫn thấp (0.41)** — bảng đa dạng, ít cấu trúc năm; ontology phủ thấp. Hướng: index-time
  contextual enrichment (làm giàu doc bằng concept/period rút từ bảng — honest), hoặc fine-tune embedder.
- **Concept-coverage chưa bật mặc định** — cần ontology lớn hơn để hết hại TAT-DQA; nếu mở rộng alias
  thì có thể gộp cùng cell-match.
- **Reranker structure-aware** — general cross-encoder thất bại; reranker đọc được cấu trúc bảng
  (table-aware) là hướng mở, nhưng cell-match ở fusion đã đảm nhận phần lớn vai trò này.
- **Đòn bẩy lớn kế tiếp ở GENERATOR**, không phải retrieval (retrieval đã chạm trần ~0.62 W.Avg).
  Xem chẩn đoán Fact-F1 trong [02_experiments_and_results.md](02_experiments_and_results.md) §EXP-0.
