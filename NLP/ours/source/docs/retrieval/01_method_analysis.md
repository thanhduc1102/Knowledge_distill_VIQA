# Phân tích từng họ phương pháp Retrieval

Tài liệu này giải thích **tại sao** mỗi họ tín hiệu được giữ lại hoặc loại bỏ, kèm cơ chế, bằng
chứng đo được, và tham chiếu văn liệu. Số liệu chi tiết: [02_experiments_and_results.md](02_experiments_and_results.md).

---

## 1. Text retrieval — Dense vs Sparse (BM25)

**Cơ chế.** Dense = embed câu hỏi & tài liệu bằng `intfloat/multilingual-e5-large-instruct` (1024-d),
cosine. Sparse = BM25 trên token toàn văn.

**Đo được.** Trên cả 3 dataset, **BM25 vượt dense rõ rệt**:

| | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| dense e5 | 0.394 | 0.426 | 0.245 |
| BM25 | 0.665 | 0.642 | 0.418 |

**Phân tích.** Câu hỏi T²-RAGBench được reformulate để *chứa* tên công ty + thuật ngữ tài chính cụ
thể ("net cash from operating activities", "2019"). Đây là *từ vựng kiểm soát* → BM25 khớp chính
xác. Dense bị *granularity dilemma*: một vector tổng nhấn chìm khóa mịn (entity/period), nhất là khi
các báo cáo cùng công ty qua nhiều năm gần như trùng từ vựng.

**Văn liệu xác nhận.** [BM25→Corrective RAG benchmark (text-and-table)](https://arxiv.org/abs/2604.01733):
BM25 R@5 0.644 > dense 0.587; "precise financial terminology provided strong lexical signals".

**Phán quyết: GIỮ BM25 làm backbone. LOẠI dense** — fusion có dense (RRF w=0.3 hoặc 1.0) đều làm
giảm điểm vì kênh yếu kéo kênh mạnh xuống.

---

## 2. Metadata filtering

**Các biến thể & phán quyết.**

| Biến thể | Cơ chế | Phán quyết |
|---|---|---|
| Hard filter (gold field) | Lọc corpus theo `company_name`/`report_year` field vàng | ❌ **LEAK** — dùng nhãn mà test-time không có |
| Self-query (infer từ câu hỏi) | NER/regex rút entity+year *từ câu hỏi* → lọc/boost | ✅ chính danh |
| Soft gated boost | Boost candidate khớp, không veto | ✅ **đang dùng** (period) |
| Index-time enrichment | LLM thêm summary company/period/metric lúc index | ⏳ chưa thử (+2-3pp theo văn liệu) |

**Định lượng leak.** Hệ cũ nhồi `company: question` vào query và lọc theo gold metadata, đạt FinQA
0.710. Honest BM25 (không gold) = 0.665; nhồi company vào *dense query* thêm ≈ 0.000. → **leak thật
chỉ ~0.04**, vì entity đã nằm sẵn trong câu hỏi → BM25 bắt được honest.

**Period là facet metadata chính danh.** Rút *năm từ câu hỏi* (regex), boost doc có kỳ báo cáo khớp.

> ⚠️ **Bài học gating (quan trọng).** Period dạng *kênh RRF độc lập* hoặc *bonus cố định* làm GIẢM
> điểm TAT-DQA: 80% bảng TAT-DQA đa-kỳ → gần như mọi candidate khớp năm → period không phân biệt,
> chỉ thêm nhiễu. **Fix: GATE** — chỉ áp period khi ≤ `period_gate` (0.4) tỉ lệ pool khớp năm (period
> phải *discriminative*). Gate vừa cứu TAT-DQA (0.388→0.417) vừa cải thiện FinQA (0.670→0.678).

**Văn liệu.** [Self-query retriever](https://medium.com/@lorevanoudenhove/enhancing-rag-performance-with-metadata-the-power-of-self-query-retrievers-e29d4eecdb73),
[metadata filtering + hybrid search](https://zilliz.com/blog/metadata-filtering-hybrid-search-or-agent-in-rag-applications).

**Phán quyết: GIỮ period dạng gated soft-boost. LOẠI hard filter gold (leak).**

---

## 3. Table retrieval — granularity & serialization

Nguyên lý nền: [Dense X Retrieval (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.845/) —
**granularity càng mịn càng tốt** (proposition > passage > document) vì đơn vị mịn cho điểm chính xác.

### 3.1 Bốn mức granularity

| Mức | Cơ chế | Bằng chứng | Trên dữ liệu ta |
|---|---|---|---|
| **Table-level** | embed cả bảng markdown → 1 vector | **tệ nhất**: over-compress, "vector dilution", mất quan hệ row-col | ❌ chính là cách dense/GSR gốc làm |
| **Row-level** | mỗi hàng (line-item + kỳ) = 1 đơn vị | top-3 recall cao nhất ([FinQA fine-grained](https://arxiv.org/abs/2206.08506)) | ✅ hợp bảng row-major |
| **Cell-level** | mỗi (row,col) = 1 fact | top-5 recall cao hơn row; [TableRAG](https://arxiv.org/abs/2410.04739), [FT-RAG](https://arxiv.org/abs/2605.01495) | ✅ = Fact Ledger của ta |
| **Schema-level** | index riêng header row/col | bổ trợ; header-only đứng riêng **yếu** (ta đo 0.318) | ⚠️ chỉ tốt khi kết hợp |

**Tại sao table-level chết:** [BM25→CRAG benchmark](https://arxiv.org/abs/2604.01733) đo **73% lỗi
retrieval = table-structure mismatch** — markdown bảng không embed tốt cho câu hỏi về cell cụ thể.

### 3.2 Serialization

Định dạng serialization ảnh hưởng tới ±20% hiệu năng ([Table Meets LLM](https://arxiv.org/abs/2305.13062)).
**Hai mục đích, hai serialization:**
- *Generator* (LLM đọc): giữ **markdown** (LLM hiểu tốt nhất).
- *Retrieval* (matching): serialize thành **fact tuple** `"concept [period] = value"` — tránh
  linearization bottleneck. (`Fact.render()` đã có.)

### 3.3 Hiện thực ở hệ ta: cell-match channel

`cell_match(q, d) = max_f [ jaccard(q_content_tokens, tokens(f.concept)) × period_factor(f) ]`,
với `period_factor = 1.0` nếu kỳ của fact khớp năm câu hỏi, `0.6` nếu doc có năm đó, `0.3` còn lại.
Dùng **raw row-label tokens** (phủ ~100% fact) thay vì canonical concept (chỉ 14% phủ).

**Đo được:** cell-match(0.3) gated là tín hiệu cấu trúc tốt nhất — **giúp FinQA+ConvFinQA, trung
tính TAT-DQA** (không hại như concept-coverage). Đây là hiện thân của row/cell-level retrieval, đúng
nguyên lý granularity.

**Phán quyết: GIỮ cell-match (gated). LOẠI table-level (markdown thô). Concept-coverage: tùy chọn**
(giúp ConvFinQA nhưng hại TAT-DQA do ontology chỉ phủ 14%).

---

## 4. Graph retrieval — phân biệt "graph nào sống"

| Loại graph | Cơ chế | Phán quyết |
|---|---|---|
| **Accounting-identity KG** (GSR gốc) | node=cell, edge=ω±1 ràng buộc kế toán → GAT | ❌ **CHẾT** (xem dưới) |
| **Fact graph** (concept–period–value) | Ledger như graph nông; neighbor expansion | ✅ cell-match dùng nó |
| **Entity-relation KG** ([HybridRAG](https://arxiv.org/abs/2408.04948)) | KG entity + vector, fuse | ⚠️ ROI nghi ngờ (entity đã ở BM25); chi phí xây KG cao |
| **Structural neighbor expansion** ([FT-RAG](https://arxiv.org/abs/2605.01495)) | từ fact kéo node năm liền kề | ✅ rất hợp **generator** (kéo operand %change) |

**Bằng chứng KG kế toán chết (quan trọng):**
- Template khớp **0%** trên bảng row-major FinQA/ConvFinQA (header là năm, không phải line-item).
- Edge fallback là *positional* với `omega=0` → message = `w·α·0 = 0` → GAT xuất **vector 0** với mọi
  input, mọi seed.
- Constraint score = 1.0 hằng số → `dense_only` và `dense+equationCS` cho MRR@3 **giống hệt đến 17
  chữ số** (0.38111111111111096).
- Paper GSR gốc tuyên bố "bỏ KG → 51.2" và "phủ template 87%" **không tái lập được** bằng code đã ship.

**Phán quyết: LOẠI KG kế toán khỏi retrieval. GIỮ fact-graph** cho neighbor expansion (chủ yếu ở
generator).

---

## 5. Reranking (stage-2)

**Giả thuyết (từ văn liệu):** [BM25→CRAG benchmark](https://arxiv.org/abs/2604.01733) báo cáo
cross-encoder rerank là đòn bẩy lớn nhất (+12pp R@5, dùng Cohere Rerank v4.0).

**Đo được (BAAI/bge-reranker-base, rerank top-20):** **LÀM GIẢM điểm trên cả 3 dataset.**

| | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| stage1 (BM25+period+cell) | 0.683 | 0.660 | 0.412 |
| ce_only | 0.555 | 0.577 | 0.387 |
| rrf(stage1, ce) | 0.642 | 0.652 | 0.413 |

**Phân tích.** General cross-encoder chấm tương đồng *ngữ nghĩa* trên markdown bảng bị truncate →
chịu đúng *linearization bottleneck*, không đọc được cấu trúc bảng, và làm hỏng tín hiệu lexical mạnh
của BM25. Đòn bẩy +12pp của benchmark dùng reranker *mạnh, đại-trà-tốt* (Cohere) — không chuyển giao
sang reranker yếu trên dữ liệu bảng.

**Phán quyết: LOẠI general cross-encoder rerank.** Reranker *có thể* giúp phải là **structure/fact-
aware** — mà cell-match (ở tầng fusion) đã đảm nhận vai trò đó.

---

## Bảng tổng phán quyết

| Họ phương pháp | Giữ/Loại | Lý do một dòng |
|---|---|---|
| BM25 full-text | ✅ GIỮ (backbone) | mạnh nhất, honest |
| Dense e5 | ❌ LOẠI | yếu hơn BM25, fusion làm giảm |
| Period (gated soft-boost) | ✅ GIỮ | metadata chính danh, gate để discriminative |
| Hard metadata filter (gold) | ❌ LOẠI | leak |
| Cell-level match (gated) | ✅ GIỮ | row/cell granularity, robust 3 dataset |
| Concept-coverage | ⚠️ TÙY CHỌN | giúp ConvFinQA, hại TAT-DQA |
| Table-level (markdown thô) | ❌ LOẠI | 73% lỗi structure-mismatch |
| Accounting KG + GAT | ❌ LOẠI | đóng góp toán học = 0 |
| Cross-encoder rerank | ❌ LOẠI | linearization bottleneck, làm giảm điểm |
| Fact-graph neighbor expansion | ✅ GIỮ (generator) | kéo operand đúng kỳ |
