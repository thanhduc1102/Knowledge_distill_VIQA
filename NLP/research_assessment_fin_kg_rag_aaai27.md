# Đánh Giá Hướng Nghiên Cứu KG + RAG + Suy Luận Tài Chính Toàn Trình

Ngày cập nhật: 2026-06-13

## 1. Kết luận ngắn gọn

Ý tưởng **đáng theo đuổi**, nhưng **không nên dừng ở mức mở rộng GSR-CACL cho retrieval**.

Nếu chỉ phát triển theo hướng:

- dùng KG để rerank tài liệu,
- tinh chỉnh scoring retrieval,
- và báo cáo MRR/Recall tốt hơn,

thì giá trị học thuật có thể chưa đủ mạnh cho mục tiêu AAAI-27.

Nếu tái định nghĩa bài toán thành:

> một hệ thống suy luận tài chính toàn trình, trong đó KG đóng vai trò là cầu nối giữa truy xuất bằng chứng, định vị toán hạng, thực thi số học và kiểm chứng kết quả,

thì đây là một hướng nghiêm túc, có tính mới rõ ràng và có khả năng phát triển thành một công trình mạnh.

---

## 2. Hiện trạng thư mục `NLP`

Hiện tại, thư mục `NLP/` đã có một nền nghiên cứu thật sự, không chỉ là ghi chú ý tưởng:

- `baseline/source_simplification/`: các baseline retrieval trên T2-RAGBench.
- `ours/source/`: prototype GSR-CACL gồm:
  - xây dựng KG từ bảng,
  - template matching,
  - GAT encoder,
  - joint scorer,
  - CHAP hard negative,
  - benchmark retrieval,
  - training skeleton.
- `eda/`: phân tích dữ liệu.
- các tài liệu proposal, architecture, review và idea ở mức nghiên cứu.

Nói cách khác, dự án hiện đã có một **prototype retrieval-aware có yếu tố cấu trúc tài chính**, nhưng chưa phải một **hệ suy luận tài chính toàn trình**.

---

## 3. Điểm mạnh hiện tại

### 3.1 Bám benchmark đúng thực tế

Việc dùng T2-RAGBench là một quyết định rất tốt vì benchmark này buộc mô hình phải:

- retrieve context trước,
- sau đó mới reasoning,
- thay vì giả định context vàng đã có sẵn như nhiều setting cũ.

Điều này phù hợp hơn nhiều với thực tế tài liệu tài chính dài, nhiều bảng và nhiều đoạn văn.

### 3.2 Đã có hướng đưa tri thức kế toán vào retrieval

Prototype hiện tại không chỉ dựa vào semantic similarity thuần túy. Nó đã bắt đầu:

- chuẩn hóa header tài chính,
- sinh KG theo template kế toán,
- dùng graph encoder,
- dùng constraint score,
- khai thác metadata `company/year/sector`.

Đây là điểm khởi đầu rất có giá trị.

### 3.3 Đã có hạt nhân cho một bài báo retrieval

Nếu cần, hoàn toàn có thể viết một bài ở mức retrieval-centric dựa trên:

- GSR,
- CACL,
- CHAP,
- benchmark T2-RAGBench.

Tuy nhiên, đây chỉ nên được coi là **bước đệm**, không phải đích cuối.

---

## 4. Hạn chế cốt lõi của hệ hiện tại

### 4.1 KG hiện tại chưa trung thành với toán học kế toán

Ở code hiện tại, các ràng buộc dạng:

`Current Assets + Non-Current Assets = Total Assets`

được tách thành các cạnh cặp đôi đi vào `Total Assets`.

Biểu diễn này có ích cho cấu trúc retrieval, nhưng chưa đủ tốt cho reasoning vì:

- không giữ được một phương trình đầy đủ,
- không có node phép toán,
- không có hyperedge hay equation node,
- khó kiểm chứng chính xác nhiều toán hạng.

### 4.2 Constraint score chưa tương ứng với kiểm chứng số học đầy đủ

Constraint score hiện tại hoạt động như một tín hiệu cấu trúc xấp xỉ, không phải bộ kiểm định số học thực sự.

Hệ quả:

- bảng đúng vẫn có thể nhận điểm thấp,
- bảng không match template có thể nhận điểm trung tính hoặc cao,
- tín hiệu reasoning và tín hiệu retrieval chưa thật sự thống nhất.

### 4.3 Metadata đang bị dùng quá nông

Ba trường:

- công ty,
- năm,
- ngành,

hiện mới đóng vai trò gần như một bonus trong reranking, chưa phải một ontology truy xuất có kiểu.

Điều này khiến hệ chưa tận dụng được các lợi thế lớn nhất của miền tài chính:

- tính định danh thực thể rất rõ,
- quan hệ thời gian rất rõ,
- cấu trúc statement / section / report type rất rõ.

### 4.4 Chưa có cầu nối retrieval -> reasoning

Hiện tại hệ mới chủ yếu trả lời câu hỏi:

> tài liệu nào nên được đưa lên đầu?

Nhưng bài toán nghiên cứu mạnh hơn là:

> sau khi có top-k context, làm sao định vị đúng các bằng chứng con, chọn đúng toán hạng, thực hiện đúng phép toán và xác minh đúng kết quả?

Khoảng trống này chính là nơi cần đầu tư cho giai đoạn tiếp theo.

---

## 5. Đánh giá khách quan về tính khả thi

### 5.1 Ý tưởng có khả thi không?

Có, nhưng phải chuyển từ:

- KG cho retrieval

sang:

- KG / evidence graph dùng chung cho retrieval và reasoning.

### 5.2 Có đủ tính mới không?

Nếu chỉ dừng ở retrieval:

- tính mới có nhưng chưa sâu,
- dễ bị xem là một biến thể graph reranking.

Nếu mở rộng thành:

- hierarchical retrieval,
- metadata-aware retrieval,
- equation-centric evidence graph,
- executor + verifier cho numerical reasoning,

thì tính mới tăng lên đáng kể.

### 5.3 Có rủi ro gì?

Rủi ro lớn nhất là dàn trải quá nhiều ý tưởng mà không chốt được contribution lõi.

Do đó, nên tập trung vào 2 contribution chính:

1. Retrieval phân cấp có metadata và evidence graph.
2. Reasoning số học có grounding + executor + verifier trên local evidence graph.

---

## 6. Hướng chốt nên theo

Hướng tốt nhất để theo đuổi là:

> xây dựng một Financial Evidence Graph có kiểu, dùng chung cho hai pha retrieval và reasoning, trong đó metadata đóng vai trò ontology nhẹ, còn KG đóng vai trò neo bằng chứng và kiểm chứng logic tài chính.

Nói rõ hơn:

- retrieval không nên dừng ở document-level,
- phải đi tới table/section-level rồi evidence-atom-level,
- reasoning không nên chạy trực tiếp trên top-3 context thô,
- mà cần chạy trên một local evidence subgraph đã được triage.

---

## 7. Khuyến nghị cho mục tiêu AAAI-27

Muốn đủ mạnh cho AAAI-27, bài toán nên được trình bày như sau:

### Đặt bài toán

Hỏi-đáp tài chính toàn trình trên tài liệu dài, đa bảng, đa nguồn bằng chứng, có nhiễu retrieval.

### Đóng góp 1

Metadata-aware hierarchical retrieval cho tài liệu tài chính:

- từ document -> table/section -> evidence atom,
- với metadata được tích hợp như ontology truy xuất.

### Đóng góp 2

Equation-centric financial evidence graph:

- liên kết cell, row, sentence, footnote, thực thể, thời gian và phép toán,
- phục vụ đồng thời retrieval và reasoning.

### Đóng góp 3

Reasoning có thể kiểm chứng:

- sinh DSL/Python program,
- thực thi,
- kiểm tra lại bằng verifier và constraint graph.

---

## 8. Khuyến nghị cuối cùng

Tôi đánh giá:

- **nên tiếp tục triển khai**,
- nhưng **không nên khóa chặt bài báo vào retrieval-only**,
- và **cần chuyển trọng tâm sang evidence-grounded reasoning** càng sớm càng tốt.

Nếu thực hiện đúng, dự án có thể phát triển theo lộ trình:

1. hoàn thiện retrieval phân cấp,
2. nâng metadata thành ontology truy xuất,
3. thay KG hiện tại bằng evidence graph trung thành với phương trình,
4. thêm executor và verifier,
5. huấn luyện reasoning với SFT + preference optimization + RL có reward verifiable.

Đây là hướng mạnh hơn, sạch hơn và có khả năng tạo ra đóng góp học thuật rõ ràng hơn nhiều so với việc chỉ cải tiến điểm MRR/Recall.
