# Modular Multi-Expert Retrieval (MMER) — thiết kế

> Trả lời phê bình "xếp chồng": thay vì cộng dồn bonus thủ công lên BM25, mỗi tín hiệu là một
> **Expert độc lập** cho ra điểm `s_i(Q,D)`; một **đầu fusion HỌC ĐƯỢC (MLP)** kết hợp chúng —
> đúng tinh thần `JointScorer` thế hệ 1 (`s = α·s_text + β·s_entity + γ·s_constraint`) nhưng
> tổng quát hơn và mỗi expert **đánh giá riêng được**.

## Nguyên tắc
1. **Độc lập:** mỗi expert có biểu diễn riêng, scoring riêng, đo MRR@3 standalone riêng.
2. **Kết hợp học được:** fusion là MLP/gate, không phải hệ số tay.
3. **Continual-learning friendly:** experts huấn luyện/đóng băng độc lập; thêm expert mới chỉ
   cần huấn luyện lại *đầu fusion nhỏ* (experts giữ nguyên → không nhiễu xuyên-mô-đun).

## Bảy Expert (mỗi cái một kỹ thuật nghiên cứu khác nhau)

| Expert | Kỹ thuật | Biểu diễn | Score | retriever? |
|---|---|---|---|---|
| **Lexical** | Sparse IR (BM25 + abbr sentinel) | túi từ + sentinel khái niệm | BM25(q,d) | ✅ |
| **Dense** | Bi-encoder semantic | e5-large 1024-d | cos(q,d) | ✅ |
| **LateInt** | **Late interaction (ColBERT-style) cấp FACT** | mỗi fact 1 vector (bge-small) | `max_f cos(q, fact)` | ✅ |
| **Entity** | Ontology + metric learning (E1 GICS / E2 alias, SupCon) | OntologyEmb 128-d; query meta **self-query honest** | cos(eq,ed) | — |
| **Concept** | Ontology kế toán (C2: 42 IFRS/GAAP concept + 7 identity) | tập concept ⊕ period | coverage(q,d) | — |
| **Cell** | Trích xuất bảng fine-grained (Fact Ledger) | cell `(row-label, period)` | max row-label⊗period | — |
| **Graph** | **Đồ thị cấu trúc fact (HierFinRAG-style)** | đồ thị concept↔period + identity edges | structural satisfaction theo *intent* (temporal/ratio/lookup) | — |

`retriever?` = expert có thể **gieo pool** (chấm toàn corpus → tăng recall). Mỗi expert chuẩn
hóa điểm **theo từng query trong pool** (min-max) để cùng thang trước khi fusion.

### Hai expert đáng chú ý (đáp ứng định hướng KG/ontology)
- **LateInt** giải *context-sharing* (1 doc trả 3+ câu hỏi): thay vì 1 vector cho cả bảng, giữ
  1 vector cho mỗi fact và chấm theo fact liên quan nhất (`max_f`). Là analogue cấp-câu của
  ColBERT token-MaxSim; chấm toàn corpus nên **phá trần recall** của BM25.
- **Graph** dựng đồ thị `concept↔period` + cạnh đồng-nhất-thức kế toán, rồi chấm theo *cấu trúc*
  mà query cần (intent): câu "change/growth" cần concept tồn tại ở ≥2 kỳ (đường đi
  concept–periodA–periodB); câu "ratio/margin" cần ≥2 concept đồng hiện trong cùng kỳ; identity
  cho phép concept *suy ra được* hưởng tín dụng một phần. **Khác với gen-1**: operand khớp trên
  HÀNG (đúng trục), không phải cột-năm (lý do gen-1 chết). Đây là hiện thân định hướng
  HierFinRAG: xây đồ thị cấu trúc như tín hiệu truy hồi.

## Đầu Fusion (học được)

Pool ứng viên = BM25 top-50 (honest backbone, đảm bảo recall). Với mỗi query, mỗi expert cho
vector điểm trên pool → ma trận `[n_pool, n_expert]`.

- **(F1) Global-weight:** `s = Σ softplus(w_i)·s_i` — tổng quát α/β/γ thế hệ 1 cho k expert.
- **(F2) Query-conditioned gate (nâng cấp):** `w_i(Q) = softmax(MLP(φ(Q)))_i`, với `φ(Q)` =
  đặc trưng *độ phân biệt* của tín hiệu cho chính query này
  `[has_year, n_concepts, has_company, |q|, frac_pool_year_match, frac_pool_concept_hit]`.
  → **HỌC** cơ chế discriminative-gating (mà ta phát hiện thủ công ở Phase A) thay vì gate tay.

Huấn luyện: **InfoNCE listwise** trên pool (gold vs distractor cùng pool = hard negative tự
nhiên). Đánh giá **k-fold CV (5)**: mỗi query được chấm bởi model KHÔNG huấn luyện trên fold của
nó → số headline trên *toàn bộ* test set, không leak, không phụ thuộc một split may rủi.

**Pool gieo từ các retriever expert** (lexical ∪ dense ∪ lateint top-50). Pool KHÔNG nhồi gold
→ recall pool là trần thật; thêm dense/lateint vào pool là cách nâng trần (không phải fusion).

## Vì sao thiết kế này đúng hướng SOTA
- Mỗi failure-mode EDA có một expert chuyên trị (same-company→Entity/Cell, abbreviation→Lexical
  sentinel, context-sharing→Cell/Concept fine-grained, semantic→Dense).
- Gate query-conditioned giải mâu thuẫn period-trên-TAT-DQA một cách *học được*, không tay.
- Khung mở: cắm thêm **LateInteractionExpert (ColBERT fact-level, Phase C)** chỉ là thêm 1 cột
  vào ma trận expert + huấn luyện lại gate → đúng tinh thần continual learning.

Hiện thực: `gsr_cacl/experts/` + `scripts/modular_retrieval.py`.
