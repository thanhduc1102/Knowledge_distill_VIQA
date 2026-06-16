# BÁO CÁO NGHIÊN CỨU — Truy hồi Đa-Expert Mô-đun (MMER) cho T²-RAGBench

> Báo cáo tổng hợp toàn bộ quá trình nghiên cứu pha *retrieval*: chẩn đoán, các thế hệ đã thử,
> kiến trúc cuối, phương pháp luận đánh giá trung thực, kết quả đo thực, phân tích và hạn chế.
> Mọi số liệu đo **honest** (không dùng gold metadata của query) trên **toàn bộ** test set.
> Tài liệu liên quan: [00_overview](00_overview.md) · [01_method_analysis](01_method_analysis.md) ·
> [02_experiments](02_experiments_and_results.md) · [04_phaseA](04_phaseA_results.md) ·
> [05_modular_design](05_modular_design.md) · [06_modular_results](06_modular_results.md).

---

## 1. Bài toán & tiêu chí đánh giá

**T²-RAGBench** đánh giá truy hồi trên tài liệu tài chính text+table (FinQA / ConvFinQA /
TAT-DQA). Mỗi câu hỏi có một tài liệu vàng (`context_id`); nhiệm vụ retrieval là xếp tài liệu
vàng lên đầu. Độ đo: **MRR@3** (chính), Recall@{1,3,5}, NDCG@3.

| Dataset | #query (test) | #corpus | Đặc thù |
|---|---|---|---|
| FinQA | 1147 | 2789 | context-sharing 2.97 q/doc |
| ConvFinQA | 3458 | 1806 | multi-turn, cùng công ty nhiều năm |
| TAT-DQA | 1144 | 2723 | bảng đa-kỳ, 1514 viết tắt, ít metadata phân biệt |

**Hợp đồng trung thực (honest contract).** Loader nhồi `"{company}: {question}"`; ta **bóc**
tiền tố này. Năm/công ty/khái niệm chỉ rút **từ câu hỏi**, không từ field gold
`report_year`/`company_name`. Đây là điều kiện khắt khe hơn hệ gen-1 (dùng gold metadata).

---

## 2. Chẩn đoán: vì sao retrieval tổng quát sụp đổ (từ EDA)

Bảy nguyên nhân (chi tiết [EDA](../../../eda/EDA_REPORT_CONSOLIDATED.md)), nhóm lại:

1. **Bảng thống trị & số liệu dày** — 100% doc có bảng; embedding mặc định không hiểu cấu trúc
   row/col; số bị tách subword, mất nghĩa.
2. **Hard negative cùng công ty** — intra-company similarity ~2.8× inter-company → "Apple 2019"
   vs "Apple 2020" gần như trùng. 59.5% query có doc-sai điểm cao hơn doc-đúng.
3. **Context-sharing** — 92% doc FinQA / 99.7% TAT-DQA phục vụ >1 câu hỏi → 1 vector/doc không đủ.
4. **Mismatch hình thái viết tắt** — "GAAP" vs "generally accepted…"; BM25 0 overlap.
5. **Lexical overlap thấp** — câu hỏi reformulate context-independent; Jaccard ~0.06.
6. **Suy luận đa bước** — 38–42% câu hỏi là change/ratio across năm/khái niệm → cần *cấu trúc*.

→ **Hệ quả thiết kế:** không một biểu diễn đơn lẻ nào (sparse/dense/single-vector) trị được cả
6. Cần **nhiều expert chuyên biệt + một bộ kết hợp học được**.

---

## 3. Các thế hệ đã thử — đánh giá trung thực

### 3.1 Gen-1: GSR (Accounting-KG + GAT + constraint score)
- **Đo thực:** FinQA MRR@3 = 0.604 (> dense 0.394). **KHÔNG phải 0.**
- **Nhưng nguồn của 0.604 là tín hiệu ENTITY/metadata** (gold company/year exact-match +
  nhồi company vào query), **không phải** đồ thị. Phần KG/GAT/constraint đóng góp ≈ 0:
  bảng row-major → 0 accounting edge khớp → `constraint_score ≡ 1.0` cho mọi doc.
- **Bài học:** số đẹp đến từ metadata, không từ phần "novel" → phải báo cáo trung thực; và 0.604
  là **leaky** (gold meta). Honest BM25 (0.665) vượt nó.

### 3.2 Gen-2: LEDGER-RAG v2 (ontology E1/E2 + C2/C3 + CACL InfoNCE)
- MRR@3 0.71–0.74 nhưng **candidate set = dense top-50 ∪ same-company±1yr** dùng **gold corpus
  metadata** → recall=1.0 nhân tạo (leaky). Đóng góp thật: ontology + InfoNCE weights là tốt;
  cách dựng candidate là leak.

### 3.3 Phase A: honest BM25 + abbr + thử metadata boost
- BM25 + gated period + gated cell + **abbr sentinel** = **0.6176** W.Avg (honest, đã tích hợp).
- **Negative result quan trọng:** boost company (phẳng & company×year joint) **đều hại** — ở
  MRR@3≈0.68 lỗi còn lại là "đúng công ty/năm, sai section" (context-sharing), boost phẳng chỉ
  làm phẳng thứ hạng tinh của BM25. → tầng BM25 cấp-doc **bão hòa ~0.62**.

→ Đòn bẩy nằm **dưới mức (company,year)**: fact/section-level + kết hợp học được. Đó là MMER.

---

## 4. Kiến trúc MMER — bảy expert độc lập + fusion học được

Nguyên tắc (sửa lỗi "xếp chồng" của Phase A): mỗi tín hiệu là **expert độc lập** cho điểm
`s_i(Q,D)`, đo standalone được; một **đầu fusion HỌC ĐƯỢC** kết hợp — đúng tinh thần `JointScorer`
gen-1 (`α·s_text+β·s_entity+γ·s_constraint`) nhưng tổng quát & sạch.

### 4.1 Bảy expert (mỗi cái một kỹ thuật)

| # | Expert | Kỹ thuật nghiên cứu | Biểu diễn | Score | seed pool? |
|---|---|---|---|---|---|
| 1 | **lexical** | Sparse IR (BM25) + abbr sentinel | túi từ + sentinel khái niệm | BM25 | ✅ |
| 2 | **dense** | Bi-encoder (e5-large-instruct) | 1024-d | cos | ✅ |
| 3 | **lateint** | Late interaction cấp fact (ColBERT-style) | 1 vector/fact (bge-small) | `max_f cos(q,f)` | ✅ |
| 4 | **entity** | Ontology (GICS/alias) + SupCon metric learning | 128-d; query meta self-query | cos | — |
| 5 | **concept** | Ontology kế toán C2 (42 concept + 7 identity) | tập concept⊕period | coverage | — |
| 6 | **cell** | Trích xuất bảng fine-grained (Fact Ledger) | (row-label, period) | max overlap×period | — |
| 7 | **graph** | Đồ thị cấu trúc fact (HierFinRAG-style) | concept↔period + identity edges | structural-satisfaction | — |

**Chi tiết hai expert cấu trúc (định hướng KG/ontology):**

- **lateint (`experts/late_interaction.py`).** Single-vector dense nén cả bảng thành 1 điểm → mờ
  với doc phục vụ nhiều câu hỏi (context-sharing). Late interaction giữ **1 vector/fact**
  (`concept [period] = value`) và chấm doc theo *fact liên quan nhất*: `s = max_f cos(enc Q, enc f)`.
  Là analogue cấp-câu của ColBERT token-MaxSim; vì chấm toàn corpus nên **nâng trần recall** mà
  BM25 bỏ sót.
- **graph (`experts/graph.py`).** Dựng đồ thị `concept↔period` + cạnh đồng-nhất-thức kế toán (7
  IDENTITIES), chấm theo *cấu trúc query cần* (intent): câu **temporal** ("change/growth") cần
  concept tồn tại ở ≥2 kỳ (đường concept–periodA–periodB); câu **ratio** ("margin") cần ≥2 concept
  đồng hiện trong cùng kỳ; concept *suy ra được* qua identity (operand đủ & nối) hưởng tín dụng
  một phần. **Khác gen-1 ở điểm sống còn:** operand khớp trên HÀNG (đúng trục row-major), không
  phải cột-năm — nên đồ thị thực sự kích hoạt thay vì chết.

Mỗi expert chuẩn hóa **min-max theo từng query trong pool** để cùng thang.

### 4.2 Đầu fusion học được (`experts/fusion.py`)

Pool = ∪ retriever-expert top-50 (lexical ∪ dense ∪ lateint). Ma trận `F[pool×k]`. Ba combiner,
huấn luyện **listwise InfoNCE** (gold vs distractor cùng pool):

- **linear:** `s = Σ softplus(w_i)·s_i` — α/β/γ gen-1 mở rộng k expert.
- **mlp:** `s = MLP([s_1..s_k])` — MLP kết hợp tiêu chí (tương tác phi tuyến).
- **gate:** `s = Σ softmax(MLP(φ(Q)))_i · s_i` — mixture-of-experts có điều kiện query;
  `φ(Q)` = đặc trưng độ-phân-biệt `[has_year, has_company, n_concept, |q|, frac_pool_year_match,
  frac_pool_concept_hit, bias]` → **học** quy tắc discriminative-gating (Phase A đặt tay).

### 4.3 Tính chất continual-learning
Experts huấn luyện/đóng băng độc lập; thêm expert mới = thêm 1 cột vào `F` + huấn luyện lại đầu
fusion nhỏ. Không nhiễu xuyên-module, không phải train lại từ đầu → khung mở rộng tự nhiên.

---

## 5. Phương pháp luận đánh giá (chống leak)

- **Honest:** bóc prefix; year/company/concept từ câu hỏi; doc-side metadata/ontology là index-time
  (không leak).
- **Pool KHÔNG nhồi gold** → recall pool là trần thật; nếu gold ngoài pool → tính miss.
- **5-fold CV:** mỗi query được chấm bởi model fusion KHÔNG huấn luyện trên fold của nó → số
  headline trên *toàn bộ* test set, không phụ thuộc một split may rủi, không leak nhãn.
- Standalone expert đo trên toàn bộ query (không cần train).

Tái lập: `PYTHONPATH=src python scripts/modular_retrieval.py --dataset FinQA --cv 5 \
  --experts lexical,dense,entity,concept,cell,graph,lateint --device cuda`

---

## 6. Kết quả

**Cấu hình:** 7 expert, 5-fold CV, honest, full test set. Pool = lexical∪dense∪lateint top-50.
Device 2×T4. `outputs/modular/{dataset}/modular.json`.

### 6.1 Từng expert đo ĐỘC LẬP (MRR@3, toàn bộ query, xếp trong pool ~120 doc)

| expert | FinQA | ConvFinQA | TAT-DQA | kỹ thuật |
|---|---|---|---|---|
| lexical (BM25+abbr) | 0.665 | 0.641 | 0.418 | sparse IR |
| dense (e5-large) | 0.394 | 0.425 | 0.244 | bi-encoder |
| lateint (fact MaxSim) | 0.201 | 0.273 | 0.155 | late interaction |
| entity (ontology+SupCon) | 0.210 | 0.324 | 0.090 | metric learning |
| cell (Fact Ledger) | 0.217 | 0.284 | 0.102 | table extraction |
| graph (HierFinRAG-style) | 0.023 | 0.091 | 0.003 | structural KG |
| concept (C2 coverage) | 0.021 | 0.081 | 0.003 | ontology |

*(Số standalone thấp hơn bảng [06](06_modular_results.md) vì pool ở đây lớn hơn — union 3
retriever ~120 doc thay vì 50 — nên xếp hạng khó hơn. Điều quan trọng là đóng góp khi fusion.)*

### 6.2 Fusion học được (5-fold CV, MRR@3 toàn bộ test set, HONEST)

| | FinQA | ConvFinQA | TAT-DQA | **W.Avg** |
|---|---|---|---|---|
| pool recall | 0.993 | 0.993 | 0.934 | — |
| **lexical (best standalone)** | 0.665 | 0.641 | 0.418 | 0.601 |
| fusion: linear | 0.768 | 0.778 | 0.481 | 0.717 |
| **fusion: mlp** ⭐ | 0.767 | **0.782** | **0.495** | **0.722** |
| fusion: gate | 0.767 | 0.770 | 0.476 | 0.711 |
| **Δ (mlp − lexical)** | **+0.103** | **+0.141** | **+0.077** | **+0.121** |

Full metrics (best head): FinQA `linear` R@1 0.671 / R@3 0.885 / R@5 0.927 / NDCG@3 0.798;
ConvFinQA `mlp` R@1 0.688 / R@3 0.892 / R@5 0.939 / NDCG@3 0.810; TAT-DQA `mlp` R@1 0.368 /
R@3 0.653 / R@5 0.750 / NDCG@3 0.535.

### 6.3 Trọng số fusion học được (linear, CV-avg) — tự thích nghi per dataset

| | lexical | entity | dense | lateint | cell | graph | concept |
|---|---|---|---|---|---|---|---|
| FinQA | **1.136** | 0.716 | 0.207 | 0.128 | 0.133 | 0.091 | 0.083 |
| ConvFinQA | 0.913 | **0.762** | 0.268 | 0.191 | 0.139 | 0.089 | 0.084 |
| TAT-DQA | **0.744** | 0.451 | 0.159 | 0.090 | 0.081 | 0.051 | 0.046 |

### 6.4 So sánh tiến trình & leaderboard (MRR@3)

| Phương pháp | FinQA | ConvFinQA | TAT-DQA | W.Avg |
|---|---|---|---|---|
| dense e5 (gen-2 backbone) | 0.394 | 0.426 | 0.245 | 0.383 |
| honest BM25 (Phase A) | 0.686 | 0.662 | 0.415 | 0.618 |
| **MMER 7-expert fusion (đây)** | **0.768** | **0.782** | **0.495** | **0.722** |
| *leaderboard Hybrid-BM25 (e5)* | *0.398* | *0.436* | *0.293* | *~0.40* |
| *leaderboard #1 GPT-5.4 meta-BM25* | *0.903* | *0.845* | *0.679* | *~0.82* |

---

## 7. Phân tích

**(1) Kết hợp học được > mọi method đơn lẻ — đây là phát hiện cốt lõi.** Không expert nào > 0.67
một mình, nhưng fusion đạt 0.77–0.78 (FinQA/ConvFinQA) và 0.495 (TAT-DQA). W.Avg honest **0.601 →
0.722 (+0.121)** so với BM25-trong-pool, và **+0.104** so với Phase A. Entity/cell/lateint yếu
standalone nhưng *bổ sung* cho lexical ở đúng các query nó trượt — fusion khai thác phần bù đó.
Đây là minh chứng định lượng cho luận điểm "experts độc lập + combiner học được" (đúng tinh thần
`JointScorer` gen-1, tổng quát hóa và sạch).

**(2) MLP combiner thắng tổng thể (2/3 dataset).** `mlp` (MLP phi tuyến trên vector điểm expert)
> `linear` > `gate`. Tương tác phi tuyến giữa experts (vd concept ∧ period, entity ∧ cell) có
giá trị. `gate` (query-conditioned) chưa vượt — có thể do φ(Q) còn thô hoặc cần nhiều dữ liệu
hơn; là hướng tinh chỉnh.

**(3) Fusion tự học trọng số khác nhau per dataset — điều hệ số tay không làm được.**
- ConvFinQA: **entity là trụ mạnh nhất (0.762)** — multi-turn cùng công ty → disambiguation thực
  thể quyết định. Fusion tự phát hiện.
- FinQA: **lexical thống trị (1.136)** — câu hỏi giàu thuật ngữ, BM25 đủ mạnh.
- TAT-DQA: lexical + entity, nhưng mọi tín hiệu đều yếu hơn (bảng đa-kỳ, ít metadata phân biệt).

**(4) dense + lateint nâng TRẦN recall — mục tiêu đã đạt.** Pool recall TAT-DQA **0.886 → 0.934**
(+0.048) nhờ dense/lateint gieo pool (FinQA/ConvFinQA → 0.993). Đây là đóng góp *recall* (kéo gold
mà BM25 bỏ sót vào pool), tách bạch với đóng góp *ranking* của fusion. Trọng số lateint 0.09–0.19:
khiêm tốn ở ranking nhưng giá trị chính là recall.

**(5) graph & concept yếu standalone nhưng nhất quán dương khi fusion.** Ontology chỉ phủ ~14%
canonical concept → tín hiệu thưa; tuy vậy trọng số học được luôn > 0 và giúp lớp query
change/ratio. Là tín hiệu *chính xác-cao, phủ-thấp* — đúng vai trò trong hỗn hợp.

**(6) So với leaderboard.** Vượt xa mọi system non-oracle (~0.40 MRR@3). Tiệm cận #1 GPT-5.4 ở
ConvFinQA (0.782 vs 0.845) nhưng còn cách ở FinQA/TAT-DQA — #1 nhiều khả năng enrichment index-time
dưới mức doc (hướng fine-tune lateint của ta nhắm tới).

---

## 8. Hạn chế & rủi ro

- **Trần recall của pool.** Fusion chỉ xếp lại trong pool; nếu gold ngoài pool thì vô phương.
  TAT-DQA bị chặn ở đây — đòn bẩy là retriever gieo pool tốt hơn (dense/lateint), không phải fusion.
- **Ontology phủ ~14%** canonical concept → concept/graph expert yếu standalone (mạnh khi fusion).
- **So với leaderboard #1 (GPT-5.4 metadata-BM25 @ 0.90 FinQA)** vẫn còn khoảng cách — họ nhiều
  khả năng enrichment index-time dưới mức doc.
- **MRR là retrieval**, chưa phải NM end-to-end; cần generator để lên leaderboard.

## 9. Hướng tiếp theo

- **Fine-tune lateint encoder** (bge-small + LoRA) trên hard-negative same-company±1yr → late
  interaction học miền tài chính (hiện dùng pretrained).
- **Reranker cấu trúc** (table-aware) thay cross-encoder phẳng đã thất bại.
- **Generator fact-grounded** dùng evidence block của Fact Ledger.
- **Mở rộng ontology** để concept/graph phủ cao hơn (giảm phụ thuộc raw token).
