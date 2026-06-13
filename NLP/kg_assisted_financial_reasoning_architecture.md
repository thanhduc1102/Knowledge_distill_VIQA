# Kiến Trúc KG Hỗ Trợ Suy Luận Tài Chính Toàn Trình

Ngày cập nhật: 2026-06-13

## 1. Mục tiêu kiến trúc

Xây dựng một hệ thống hỏi-đáp tài chính toàn trình có khả năng:

1. truy xuất đúng tài liệu liên quan;
2. xác định đúng bảng / đoạn / footnote liên quan;
3. định vị đúng toán hạng;
4. thực hiện suy luận số học nhiều bước;
5. kiểm chứng lại kết quả trước khi trả lời.

Trong kiến trúc này, KG không đóng vai trò là “máy tính”.
KG đóng vai trò:

- bộ nhớ cấu trúc,
- bộ neo bằng chứng,
- bộ ràng buộc kiểu dữ liệu,
- bộ kiểm chứng logic tài chính.

Phần tính toán cuối cùng nên đi qua một executor kiểu DSL hoặc Python.

---

## 2. Nguyên tắc thiết kế

### G1. KG phải phục vụ cả retrieval và reasoning

Nếu KG chỉ dùng để:

- match template,
- encode table,
- rerank tài liệu,

thì nó chỉ là graph retrieval.

Muốn hỗ trợ reasoning, KG phải giữ được:

- provenance,
- entity,
- thời gian,
- đơn vị và scale,
- loại statement,
- quan hệ tổng hợp,
- phép toán,
- liên kết text-table-footnote.

### G2. Không dùng pairwise edge để biểu diễn phương trình đầy đủ

Ví dụ:

`Current Assets + Non-Current Assets = Total Assets`

không nên chỉ biểu diễn bằng hai cạnh:

- `Current Assets -> Total Assets`
- `Non-Current Assets -> Total Assets`

Mà cần:

- equation node,
- hoặc operator node,
- hoặc hyperedge.

### G3. Mọi fact phải có provenance

Mỗi giá trị / quan hệ / equation cần truy được về:

- document,
- page,
- section,
- table id,
- row,
- column,
- sentence span,
- footnote span.

---

## 3. Đề xuất đồ thị bằng chứng tài chính

### 3.1 Các loại node

- `Document`
- `Section`
- `Table`
- `Row`
- `Column`
- `Cell`
- `Sentence`
- `Footnote`
- `Company`
- `TimePeriod`
- `MetricConcept`
- `Unit`
- `Equation`
- `Operation`

### 3.2 Các loại edge

- `belongs_to`
- `mentions`
- `alias_of`
- `same_company`
- `same_period`
- `same_metric`
- `has_unit`
- `has_scale`
- `supports`
- `derived_from`
- `part_of_equation`
- `equation_result`
- `referenced_by_footnote`
- `contradicts`

---

## 4. Vai trò của KG trong retrieval

KG nên hỗ trợ retrieval theo 3 tầng:

1. `document-level retrieval`
2. `table/section-level retrieval`
3. `evidence-atom retrieval`

Evidence atom ở đây có thể là:

- cell,
- row aggregate,
- sentence,
- footnote span,
- equation candidate.

Như vậy, thay vì chỉ lấy top-k document rồi đẩy nguyên văn sang LLM, hệ cần:

1. lấy top-k document;
2. xác định top-m table/section;
3. tách các atom;
4. xây local evidence graph;
5. chuyển graph con này sang module reasoning.

---

## 5. Vai trò của KG trong reasoning

Trong reasoning, KG nên làm bốn việc:

### 5.1 Neo toán hạng

KG giúp xác định:

- giá trị nào thuộc đúng công ty,
- đúng năm / quý,
- đúng statement,
- đúng đơn vị / scale,
- đúng metric.

### 5.2 Kiểm tra tính tương thích

Trước khi cộng / trừ / chia, hệ cần kiểm tra:

- cùng đơn vị hay chưa,
- cùng kỳ hay chưa,
- có cùng nghĩa tài chính hay không,
- có phải số gốc hay tỷ lệ không.

### 5.3 Gợi ý phương trình

Equation nodes và operator nodes trong KG cho phép:

- gợi ý phép toán phù hợp,
- giảm search space cho module program generation.

### 5.4 Kiểm chứng hậu nghiệm

Sau khi executor tính xong, verifier có thể dùng KG để hỏi:

- kết quả có phù hợp với constraint graph không?
- các toán hạng có đến từ đúng bằng chứng không?
- có lẫn sai company/year không?

---

## 6. Kiến trúc đề xuất

### 6.1 Pha truy xuất

Đầu vào:

- câu hỏi
- toàn bộ corpus tài chính

Các bước:

1. parser tách:
   - company
   - time scope
   - target metric
   - loại phép toán
2. metadata-aware filtering / prior
3. dense + sparse hybrid retrieval
4. table/section reranking
5. atom extraction
6. local evidence graph construction

### 6.2 Pha suy luận

Đầu vào:

- câu hỏi
- local evidence graph

Các bước:

1. operand grounding
2. unit normalization
3. operation planning
4. DSL / Python generation
5. execution
6. verification / repair

---

## 7. Module nên có trong repo

### Bắt buộc

- `evidence_graph/`
- `metadata_schema/`
- `atom_retrieval/`
- `reasoning_executor/`
- `verifier/`
- `query_parser/`

### Có thể tái sử dụng từ repo hiện tại

- loader dataset
- benchmark retrieval
- template normalization
- một phần hard-negative generation

### Nên thay thế

- pairwise constraint builder hiện tại
- raw constraint scorer hiện tại
- logic chỉ lấy bảng markdown đầu tiên

---

## 8. Kết luận

Kiến trúc phù hợp nhất không phải là:

> KG để tăng MRR.

Mà là:

> một evidence graph tài chính có kiểu, dùng chung cho retrieval và reasoning, với executor và verifier bảo đảm khả năng kiểm chứng kết quả số học.

Đó mới là kiến trúc đủ mạnh để phát triển thành một hệ thống suy luận tài chính toàn trình có giá trị nghiên cứu cao.
