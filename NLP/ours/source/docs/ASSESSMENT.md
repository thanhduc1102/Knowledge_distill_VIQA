# ĐÁNH GIÁ TOÀN DIỆN HỆ THỐNG GSR–CACL & ĐỊNH HƯỚNG NÂNG CẤP

> Tài liệu này đánh giá *thực trạng code* (không phải lời hứa trong paper) của hệ thống
> GSR (Graph-Structured Retrieval) + CACL (Constraint-Aware Contrastive Learning) trên
> benchmark **T²-RAGBench** (FinQA / ConvFinQA / TAT-DQA), đối chiếu với leaderboard &
> SOTA hiện tại, và đề ra kiến trúc nâng cấp **LEDGER-RAG**.
>
> Ngày: 2026-06-13. Phạm vi: toàn bộ `ours/source/src/gsr_cacl/` + `contribution1.pdf` +
> các tài liệu thiết kế (`core_method*.md`, `new_idea.md`, `review_*.md`).

---

## 0. Tóm tắt điều hành (TL;DR)

Hệ thống hiện tại là một **bộ khung retrieval-only được kỹ thuật hóa tốt** (KG builder,
edge-aware GAT, constraint score, joint scorer, 3-stage curriculum) nhưng **kết quả benchmark
đang KHÔNG phản ánh pipeline đã huấn luyện**, và **toàn bộ pha generator chưa tồn tại**.
Có 6 lỗi/khoảng trống nghiêm trọng phải sửa trước khi nói đến chuyện "đánh bại SOTA":

| # | Vấn đề nghiêm trọng | Bằng chứng | Hệ quả |
|---|---------------------|-----------|--------|
| **B1** | Benchmark **không nạp checkpoint** đã train → GAT + JointScorer **khởi tạo ngẫu nhiên** lúc inference | `benchmark_gsr.py:241-252` không truyền `checkpoint_path`; `gsr_retrieval.py:92` chỉ nạp nếu có | Điểm "GSR" thực chất ≈ dense + nhiễu cấu trúc; α/β/γ là mặc định cứng, GAT vô nghĩa |
| **B2** | **Lệch encoder train/eval**: train fine-tune `bge-large` (LoRA) nhưng eval embed bằng `multilingual-e5-large-instruct` | `gsr_default.yaml` (train) vs `benchmark_gsr.py:183,235`; `gsr_retrieval.py` không nạp `text_encoder_state` | Mọi nỗ lực fine-tune text encoder **bị vứt bỏ** ở inference |
| **B3** | **Entity signal là so khớp chuỗi**, không phải embedding học bằng SupCon như paper tuyên bố | `joint_scorer.py:100-107`, `gsr_retrieval.py:189-205` | Đóng góp "metadata embedding" của contribution1 **chưa được hiện thực** |
| **B4** | **CHAP-E là stub**: chỉ chèn header `[COMPANY/YEAR]`, không đổi giá trị bảng | `chap.py:134-150` | Mẫu âm entity vô dụng; không ép model học phân biệt thực thể |
| **B5** | **Constraint score chỉ pairwise**, không trung thực với phương trình nhiều toán hạng (A+B+C=Total) | `constraint_score.py:46-62` + cách build edge `builder.py:130-177` | Phần lớn đẳng thức kế toán bị "băm" thành cặp → tín hiệu yếu |
| **B6** | **Không có generator / verifier / Number-Match**; repo dừng ở `retrieval_top3.jsonl` | toàn repo | Không thể đo end-to-end, không so được với leaderboard (đo Number Match) |

Ngoài ra: **không có SupCon/EntitySupCon thật** (paper §4.3 nói có), **không có DPO/ORPO/RL**,
template matching chỉ là **so chuỗi header nông** (15 template, 14 synonym), và **chưa dùng
trường `table` sạch** của dataset (đang heuristic cắt bảng từ `context`).

**Bối cảnh SOTA (leaderboard T²-RAGBench, 2026-06):** đỉnh bảng là **"GPT-5.4 + Metadata-aware
BM25"** (Weighted-Avg ~73.7 Number Match; FinQA MRR@3 ~90.3). Bài *"From BM25 to Corrective RAG"*
(arXiv 2604.01733) chỉ ra **Hybrid RRF + Cohere Rerank** là pipeline retrieval mạnh nhất
(Recall@5 ~0.816, MRR@3 ~0.605) và **retrieval tốt hơn ⇒ answer tốt hơn**. Hai thông điệp lớn:
(1) **metadata (company/year) là tín hiệu vàng** — đúng trực giác của contribution1; (2) cuộc chơi
thực sự đo ở **Number Match end-to-end**, nên **bắt buộc phải có generator**.

---

## 1. Bản đồ hiện trạng module (cái gì thật, cái gì scaffold)

### 1.1 ĐÃ HOẠT ĐỘNG (engineering tốt, có thể tái sử dụng)
- **KG construction** (`kg/builder.py`, `kg/parser.py`): parse markdown → node = cell
  (`KGNode`: row/col/value/header/header_canonical/is_total), edge accounting (ω=±1) theo template,
  fallback positional (ω=0). Hoàn chỉnh, không stub.
- **Edge-Aware GAT** (`encoders/gat_layer.py`, `gat_encoder.py`): ω được chiếu thành bias attention
  (`edge_proj: Linear(1, n_heads)`) **và** nhân vào message — đúng tinh thần paper. Node feature =
  header-hash embedding (512 bucket) + numeric features `[log|v|, sign, is_zero, bucket]` + sinusoidal
  row/col PE. Mean-pool → graph embedding. Hỗ trợ `external_cell_embeds` (tiêm BGE) — nhưng **chưa
  dùng** ở đường inference.
- **Constraint score** (`scoring/constraint_score.py`): `exp(−|ω·v_u − v_v| / max(|v_v|, ε))`,
  edge rỗng → 1.0. Công thức đúng (5) trong paper, nhưng *pairwise* (xem B5).
- **JointScorer** (`scoring/joint_scorer.py`): α/β/γ learnable qua `softplus(log_param)`. `forward_text_sim`
  = gated cosine + KG-adjustment (±0.2). Hoàn chỉnh về cơ chế, nhưng entity là so khớp (B3).
- **CHAP-A / CHAP-S** (`negative_sampler/chap.py`): biến đổi giá trị/đơn vị thật, dựng lại markdown.
- **3-stage curriculum** (`train.py`): Identity → Structural → Joint; lưu `final_model.pt` đủ state.
- **Benchmark** (`benchmark_gsr.py`): MRR@3, Recall@1/3/5, NDCG@3 + lưu `retrieval_top3.jsonl`
  (đã có format "generator_context" sẵn cho pha sinh).

### 1.2 SCAFFOLD / SAI / THIẾU
- **B1–B6** ở trên.
- `templates/library.py`: 15 template nhưng 2 cái (revenue_segment, yoy_change) header rỗng — stub.
  Matching = `normalize_header` (14 synonym) + tỉ lệ trùng header. **Không semantic**, dễ trượt với
  TAT-DQA (bảng tự do, header đa cấp).
- `losses.py`: chỉ `TripletLoss` + `ConstraintViolationLoss`. **Không** có SupCon/EntitySupCon.
- `datasets/gsr_document.py`: `extract_table` cắt block "|...|" đầu tiên từ `context` — trong khi
  dataset đã có **trường `table` sạch**. Nên dùng trực tiếp `table`.

### 1.3 Số liệu output hiện có (cần đọc đúng bản chất)
`outputs/faiss_run` vs `outputs/manual_run` cho FinQA MRR@3 = **0.60 / 0.73**. Vì B1/B2, đây **không**
phải kết quả của GSR-trained mà gần như **dense retrieval (e5) + nhiễu**. Con số `manual_run` (ConvFinQA
Recall@3 = 0.925) cao bất thường ⇒ cần kiểm tra lại corpus/leakage trước khi tin. **Khuyến nghị: coi mọi
số hiện tại là "chưa hợp lệ" và chạy lại sau khi sửa B1–B2.**

---

## 2. Phân tích thuật toán & đồ thị tri thức (đi sâu theo yêu cầu)

### 2.1 KG hiện tại là "retrieval graph", chưa phải "reasoning graph"
- **Node** chỉ là cell với 1 giá trị scalar; **không** mang đơn vị (USD/%/shares), **không** mang
  scale (thousands/millions), **không** provenance (cell này ở bảng nào, dòng/cột tên gì), **không**
  liên kết text↔table (con số trong narrative ↔ cell).
- **Edge** accounting chỉ pairwise (src→tgt, ω=±1). Đẳng thức `Total = A+B+C` bị tách thành 3 cặp
  `A→Total(+1)`, `B→Total(+1)`, `C→Total(+1)` rồi *điểm trung bình từng cặp* — sai bản chất: một
  đẳng thức đúng/sai phải đánh giá **trên cả phương trình** (residual = |Σω·operand − total|).
- **Hệ quả:** KG không đủ "ngữ nghĩa vững" để (a) tạo tín hiệu cấu trúc mạnh khi retrieve, (b) cung
  cấp fact chính xác cho generator.

### 2.2 Định nghĩa lại KG → **Financial Evidence Graph (FEG)** (đa cấp, mang đủ thông tin)
Node types: `Document`, `Section`, `Table`, `Row(line-item)`, `Cell`, `Metric(concept canonical)`,
`Period`, `Unit/Scale`, `Entity(company)`. Edge types: `has_table`, `has_row`, `has_cell`,
`cell_of_period`, `cell_in_unit`, `mentions(text→cell)`, `equation(operands→total, ω, op)`,
`yoy(period_t → period_{t-1})`, `same_metric_diff_period`. Mỗi **fact** = bộ
`(metric, entity, period, value, unit, scale, provenance)` — đây là **đơn vị liên quan thật sự** của
truy vấn tài chính (tham chiếu HierFinRAG: Section→Para→Table→Cell + TTGNN; và "fact-level relevance"
trong `core_method_update.md`).

### 2.3 CACL: sinh mẫu âm theo **channel-alignment** (mỗi mẫu âm sai đúng 1 kênh)
Thay vì 3 phép CHAP rời rạc (mà CHAP-E lại stub), dùng 5 loại **answer-invalidating, channel-aligned**:
| Loại | Phá | Ép model học kênh |
|------|-----|-------------------|
| period-swap | đổi period (cùng entity) | gate thời gian `γ_t` |
| entity-swap (THẬT) | đổi giá trị bảng sang công ty khác (cùng metric/period) | gate thực thể `γ_e` / entity embedding |
| metric-swap | đổi giá trị giữa 2 metric cùng công ty-năm | concept channel `σ_m` |
| scale-break | nhân cột ×10³ | magnitude channel `ρ` |
| value-identity break | phá `A+B=Total` | verifier bridge / constraint |
Nguyên tắc (review_1/2): mỗi negative khác positive **đúng 1 khía cạnh** ⇒ gradient InfoNCE tập trung
đúng 1 thành phần điểm ⇒ học sạch, không bão hòa. CHAP-E hiện tại vi phạm nguyên tắc này.

---

## 3. Pha Retrieval — thiết kế lại "luồng hoàn chỉnh"

Tín hiệu (kết hợp **nhân tính** thay vì cộng để metadata sai → triệt tiêu, theo `core_method.md §3.2`):

```
s(Q,D) = α·s_text(Q,D)              # cosine của trained encoder (sửa B2)
       + β·s_entity(Q,D)            # cos(e_Q, e_D), e = entity-embedding học bằng SupCon (sửa B3)
       + γ·CS_eq(G_D)               # equation-faithful constraint score (sửa B5)
  với candidate set lọc bằng metadata-aware soft mask (company≈, year∈[t-1,t,t+1]) — tín hiệu vàng SOTA
```
- **Single-stage filtered ANN** (mask mềm + fallback nới lỏng nếu rỗng) — tránh brittle 2-stage.
- **Checkpoint thật** được nạp (sửa B1); encoder train==eval (sửa B2).
- Hybrid với BM25 (RRF) giữ lại làm biến thể mạnh (đối chiếu "Hybrid RRF" của arXiv 2604.01733).

---

## 4. Pha Generator — KG phục vụ sinh (đáp ứng yêu cầu cốt lõi)

Với top-K (vd K=3) tài liệu sau retrieve:
1. **Trích FEG/Fact-Ledger** cho từng doc (dùng trường `table` sạch + parser số trong text).
2. **Fact selection theo query**: chọn các fact `(metric, entity, period, value)` khớp ngữ nghĩa câu hỏi
   (concept-match `σ_m` + gate entity/period) → đưa **đúng cell** cho generator (không nhồi cả markdown).
3. **Generator (Qwen, cấu hình được, mặc định Qwen2.5-3B/Qwen3-4B)**: prompt fact-grounded, sinh
   chuỗi suy luận step (extract-fact / compute) rồi đáp số.
4. **Deterministic Ledger Verifier** (annotation-free): (i) value grounding — số trong đáp án có khớp
   cell trong ledger?, (ii) arithmetic — phép tính đúng?, (iii) accounting identity khi áp dụng được.
5. **Metric Number-Match** (tolerance ε=1e-2) đúng chuẩn leaderboard → so sánh khách quan.
6. **Tùy chọn nâng cao:** tạo cặp ưa thích (chosen/rejected) từ verifier reward để **DPO/ORPO**;
   hoặc **GRPO/RLVR** với reward = answer + grounding + arithmetic (λ_g+λ_a<1, accuracy ưu tiên từ điển).

---

## 5. Kế hoạch phiên bản (mỗi version là 1 thí nghiệm chạy được)

- **v1 — Retrieval đúng đắn:** sửa B1/B2, metadata-aware filter, equation-faithful CS, dùng trường `table`.
  *Đo lại MRR@3/Recall@3/NDCG@3 trên 3 dataset (số liệu thật).*
- **v2 — Entity-embedding + CACL chuẩn:** SupCon entity head, 5 mẫu âm channel-aligned (CHAP-E thật).
- **v3 — KG-for-Generator:** Fact-Ledger + fact selection + Qwen generator + verifier + Number-Match e2e.
- **v4 — Preference/RL:** DPO/ORPO (offline) và GRPO/RLVR (verifier reward) cho generator.

Mỗi version có script chạy + config + thư mục `outputs/<version>/...` lưu metrics + artifacts + log.

---

## 6. Vì sao thiết kế này đủ lớn để "đánh bại" nghiên cứu khác
1. **Đúng tín hiệu vàng của domain** (metadata/entity) — thứ đang vô địch leaderboard — nhưng nâng từ
   so-khớp/BM25 lên **embedding học được + gate nhân tính**, có thể train end-to-end.
2. **KG dùng CHUNG cho cả retrieve & generate** (fact-ledger), giải quyết đúng "context-sharing" (1 doc trả
   nhiều câu) và "mathematical flattening" — hai failure mode đã chứng minh bằng EDA.
3. **Verifier suy luận ký hiệu (symbolic)** annotation-free → reward sạch cho RLVR, đúng hướng SOTA
   (HierFinRAG symbolic-neural fusion) nhưng gắn chặt với KG ràng buộc kế toán.
4. **Mẫu âm answer-invalidating, channel-aligned** — đóng góp huấn luyện mới so với hard-negative cổ điển
   (vốn nguy hiểm vì same-company-other-year là "false negative" nhiều nhất).

> Các điểm reviewer bắt buộc (review_1/2): phải báo cáo **F1 trích fact** + **gap oracle-vs-auto**, **latency
> masked-ANN**, **ablation từng loại negative**, và **so với dense+hard-filter+reranker**. Hệ thống dưới đây
> được thiết kế để đo được tất cả các mục này.
