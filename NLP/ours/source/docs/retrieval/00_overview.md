# Retrieval Optimization — Overview & Executive Summary

> Bộ tài liệu này mô tả **toàn bộ quá trình phân tích, lựa chọn phương pháp, và thực thi** việc tối
> ưu pha *retrieval* cho hệ GSR-CACL trên T²-RAGBench (FinQA, ConvFinQA, TAT-DQA). Tất cả số liệu
> đo **honest** (không nhồi metadata vàng vào query) trên **toàn bộ** test set.

## Tài liệu trong thư mục này

| File | Nội dung |
|---|---|
| [00_overview.md](00_overview.md) | Tóm tắt điều hành, phương án cuối, số liệu headline (file này) |
| [01_method_analysis.md](01_method_analysis.md) | Phân tích từng họ tín hiệu: text/dense, metadata, table-granularity, graph/KG, reranking — kèm phán quyết & bằng chứng |
| [02_experiments_and_results.md](02_experiments_and_results.md) | Toàn bộ thực nghiệm, script, bảng kết quả đầy đủ, ablation |
| [03_final_recipe_and_repro.md](03_final_recipe_and_repro.md) | Công thức production, lý do chọn hệ số, cách tái lập & mở rộng |

## TL;DR — phương án tối ưu

**Retriever = BM25 (backbone) + gated period tie-break + gated cell-level match.**
Không dùng dense, không dùng đồ thị kế toán (GSR gốc), không dùng cross-encoder rerank — cả ba đã
được đo là **không giúp hoặc làm giảm** điểm trên dữ liệu này.

Productionized: [`gsr_cacl.methods.HybridBM25Retrieval`](../../src/gsr_cacl/methods/hybrid_bm25_retrieval.py)
(đăng ký `hybrid_bm25` trong `GSR_REGISTRY`).

## Số liệu headline (honest, full test set)

| Phương pháp | FinQA | ConvFinQA | TAT-DQA | W.Avg MRR@3 |
|---|---|---|---|---|
| dense e5-large (baseline) | 0.394 | 0.426 | 0.245 | 0.383 |
| full-text BM25 | 0.665 | 0.642 | 0.418 | 0.602 |
| BM25 + gated period | 0.680 | 0.653 | 0.414 | 0.611 |
| **BM25 + gated period + gated cell-match** | **0.683** | **0.660** | **0.412** | **0.6155** |
| _(tham chiếu) hệ leaky dùng gold-metadata filter_ | _0.710_ | — | — | — |

`n` = 1147 / 3458 / 1144 (tổng 5749 query). Metric = MRR@3.

## Bốn kết luận cốt lõi

1. **BM25 là xương sống phổ quát.** Dense (e5) yếu hơn hẳn trên cả 3 dataset; mọi hybrid có dense
   đều làm giảm điểm. Khớp với văn liệu text-and-table mới nhất (BM25 > dense trên tài chính).
2. **Đồ thị tri thức kế toán (GSR gốc) đóng góp = 0.** Đã chứng minh bằng toán học + thực nghiệm:
   template khớp 0% trên bảng row-major, GAT xuất vector 0, constraint score là hằng số. Phần "+15%"
   của GSR gốc thực chất đến từ tín hiệu *entity*, không phải đồ thị.
3. **Tín hiệu cấu trúc (period, cell-match) chỉ giúp khi *discriminative* → bắt buộc phải GATE.**
   Cùng một bonus cố định giúp FinQA/ConvFinQA nhưng *làm hại* TAT-DQA (80% bảng đa-kỳ). Gate theo
   tỉ lệ pool khớp là cơ chế chung sửa triệt để vấn đề này.
4. **Cross-encoder rerank (general) làm giảm điểm** trên cả 3 — nó chịu đúng *linearization
   bottleneck* khi đọc markdown bảng. "Reranking là đòn bẩy lớn nhất" của benchmark khác KHÔNG
   chuyển giao được sang reranker yếu/đại-trà trên dữ liệu bảng.

## Vị trí trong pipeline & bước tiếp theo

Pha retrieval đã chạm trần fusion tầng-1 (~0.61–0.62 W.Avg). Đòn bẩy lớn tiếp theo **không** nằm ở
retrieval mà ở **generator** (đã đo: nút thắt là operand-selection + arithmetic, xem
[02_experiments_and_results.md](02_experiments_and_results.md) §Fact-F1). Định hướng generator
(Symbolic-Neural Fusion) nằm ngoài phạm vi bộ tài liệu retrieval này.
