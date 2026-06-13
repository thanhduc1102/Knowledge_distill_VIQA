# Kế Hoạch Triển Khai Repo Cho Hướng KG Hỗ Trợ Suy Luận Tài Chính

Ngày cập nhật: 2026-06-13

## 1. Mục tiêu của tài liệu này

Tài liệu này ánh xạ ý tưởng nghiên cứu sang công việc kỹ thuật cụ thể trong repo hiện tại:

- phần nào giữ lại,
- phần nào cần refactor,
- phần nào cần xây mới,
- và nên triển khai theo thứ tự nào.

Mục tiêu cuối là chuyển repo từ:

- `prototype retrieval có KG`

sang:

- `hệ evidence graph + retrieval phân cấp + reasoning + executor + verifier`.

---

## 2. Những gì có thể tận dụng từ repo hiện tại

### 2.1 Dataset và benchmark

Có thể giữ và tái sử dụng:

- wrappers cho T2-RAGBench
- loader cho FinQA / ConvFinQA / TAT-DQA
- benchmark retrieval hiện tại

### 2.2 Ý tưởng có thể kế thừa

- chuẩn hóa metadata
- graph-aware retrieval scoring
- hard negatives có cấu trúc
- concept template cho tài chính

### 2.3 Một số module nguồn

- `NLP/ours/source/src/gsr_cacl/datasets/`
- `NLP/ours/source/src/gsr_cacl/templates/`
- `NLP/ours/source/src/gsr_cacl/negative_sampler/`
- một phần `training/`

---

## 3. Những gì nên xem là bản cũ để thay thế

### 3.1 KG builder hiện tại

Các file:

- `kg/builder.py`
- `kg/data_structures.py`
- `scoring/constraint_score.py`

nên được xem là nền tảng cũ cho giai đoạn retrieval, không phải graph cuối cùng của hệ.

Lý do:

- pairwise edge chưa đủ cho reasoning,
- thiếu equation node,
- thiếu provenance đầy đủ,
- thiếu unit/scale normalization object,
- chưa liên kết text-table-footnote.

### 3.2 Logic inference hiện tại

Các hạn chế hiện tại:

- chỉ lấy bảng markdown đầu tiên,
- retrieval mới ở cấp document,
- chưa có evidence atom retrieval,
- chưa có module reasoning thực thi,
- benchmark chưa phản ánh end-to-end reasoning.

---

## 4. Cấu trúc module mới nên xây

## 4.1 `evidence_graph/`

Chức năng:

- định nghĩa schema đồ thị bằng chứng tài chính
- node types
- edge types
- provenance model
- equation representation

Các file gợi ý:

- `schema.py`
- `builder.py`
- `normalizer.py`
- `equations.py`
- `provenance.py`

## 4.2 `metadata_schema/`

Chức năng:

- chuẩn hóa metadata truy xuất
- mapping company alias
- year / quarter / period normalization
- statement type normalization
- unit / currency metadata

## 4.3 `retrieval_hierarchical/`

Chức năng:

- document retrieval
- table / section retrieval
- evidence atom retrieval
- metadata-aware filtering
- hybrid sparse+dense fusion

## 4.4 `reasoning_executor/`

Chức năng:

- định nghĩa DSL
- parser cho DSL
- Python executor
- unit normalization
- tolerance logic

## 4.5 `verifier/`

Chức năng:

- kiểm tra grounding
- kiểm tra company/year consistency
- kiểm tra unit/scale consistency
- kiểm tra equation compatibility
- kiểm tra execution validity

## 4.6 `reasoning_training/`

Chức năng:

- SFT data building
- preference pair generation
- RL reward functions
- offline / online training scripts

---

## 5. Lộ trình triển khai kỹ thuật

### Giai đoạn 1: củng cố retrieval

Mục tiêu:

- làm retrieval mạnh và sạch hơn trước khi nối sang reasoning.

Việc cần làm:

1. hợp nhất train/inference preprocessing
2. mở rộng metadata schema
3. thêm metadata-aware contextual chunk embedding
4. thêm hard negatives:
   - cùng công ty sai năm
   - cùng năm sai công ty
   - cùng company-year sai statement
   - đúng bảng sai row
5. benchmark thêm table/section-level retrieval

### Giai đoạn 2: xây evidence graph

Mục tiêu:

- thay pairwise KG bằng graph trung thành với reasoning.

Việc cần làm:

1. định nghĩa node/edge schema
2. thêm provenance
3. thêm unit/scale object
4. thêm equation node / operator node
5. liên kết text-table-footnote

### Giai đoạn 3: xây module reasoning

Mục tiêu:

- reasoning không còn phụ thuộc vào raw top-k context.

Việc cần làm:

1. query parser
2. operand grounding
3. program generation
4. executor
5. verifier

### Giai đoạn 4: huấn luyện

Mục tiêu:

- huấn luyện reasoning có thể kiểm chứng.

Việc cần làm:

1. SFT trên trace / program
2. Step-DPO hoặc process-aware DPO
3. GRPO / RLVR với reward verifiable

---

## 6. Thứ tự ưu tiên thực tế

Nếu nguồn lực có hạn, nên làm theo thứ tự:

1. metadata schema
2. evidence atom extraction
3. equation-centric graph
4. executor + verifier
5. reasoning SFT
6. preference optimization
7. RL

Thứ tự này quan trọng vì:

- retrieval là đầu vào của reasoning,
- graph là nền của grounding,
- verifier là nền của reward,
- reward chưa ổn thì RL dễ hỏng.

---

## 7. Kết quả mong đợi theo từng chặng

### Chặng 1

- retrieval tốt hơn baseline
- metadata có đóng góp rõ ràng
- có bảng ablation đầu tiên

### Chặng 2

- evidence graph hoạt động được
- có thể truy từ answer về provenance
- có local subgraph cho reasoning

### Chặng 3

- mô hình sinh được program hợp lệ
- executor chạy được
- verifier lọc được reasoning sai

### Chặng 4

- có end-to-end answer accuracy
- có robustness dưới top-3 noisy contexts
- có kết quả đủ mạnh để viết paper

---

## 8. Kết luận

Repo hiện tại đã đủ tốt để làm nền cho nghiên cứu tiếp theo, nhưng chưa thể đi thẳng tới end-to-end reasoning nếu không tái cấu trúc.

Hướng đúng là:

- giữ lại phần benchmark và một phần retrieval,
- thay graph cũ bằng evidence graph mới,
- thêm executor và verifier,
- rồi mới nối sang SFT, DPO và GRPO.

Đó là con đường khả thi nhất để chuyển repo hiện tại thành một hệ thống nghiên cứu mạnh, có chiều sâu và có khả năng công bố.
