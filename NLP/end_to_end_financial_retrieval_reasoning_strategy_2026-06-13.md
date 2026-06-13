# Chiến Lược Toàn Trình Cho Retrieval Và Suy Luận Tài Chính

Ngày cập nhật: 2026-06-13

## 1. Mục đích

Tài liệu này tổng hợp:

- hiện trạng repo,
- bức tranh nghiên cứu liên quan,
- vai trò của metadata và KG,
- đánh giá các thuật toán huấn luyện ưu tiên như DPO, ORPO, GRPO,
- và chiến lược toàn trình phù hợp nhất cho mục tiêu bài báo mạnh.

Định nghĩa ngắn gọn của hướng nghiên cứu nên là:

> truy xuất đúng bằng chứng tài chính, định vị đúng toán hạng, thực thi suy luận số học chính xác và kiểm chứng kết quả dưới điều kiện retrieval có nhiễu.

---

## 2. Bài toán cần giải quyết

### 2.1 Đầu vào

- câu hỏi tài chính
- tài liệu dài gồm text, bảng, footnote
- corpus nhiều công ty, nhiều năm, nhiều statement

### 2.2 Đầu ra mong muốn

- câu trả lời cuối cùng
- bằng chứng đã grounding
- phép toán / chương trình thực thi
- khả năng kiểm chứng lại kết quả

### 2.3 Vì sao khó

Bài toán khó vì phải giải quyết đồng thời:

- retrieval dài ngữ cảnh,
- grounding text-table,
- phân biệt công ty và thời gian,
- chuẩn hóa đơn vị,
- reasoning nhiều bước,
- và kiểm chứng logic tài chính.

---

## 3. Quan sát quan trọng từ benchmark và dữ liệu

### 3.1 MRR@3 là chưa đủ

Ngay cả khi retrieval đạt MRR@3 tốt, reasoning vẫn có thể thất bại nếu:

- top-3 có 1 context đúng và 2 context nhiễu,
- đúng document nhưng sai row,
- đúng row nhưng sai year,
- đúng số nhưng sai đơn vị.

### 3.2 Phải đánh giá theo nhiều tầng

Không nên chỉ đo:

- MRR
- Recall

Mà cần bổ sung:

- độ chính xác định vị bằng chứng,
- độ chính xác chọn toán hạng,
- độ chính xác thực thi chương trình,
- độ nhất quán với ràng buộc tài chính.

---

## 4. Metadata nên được hiểu như một ontology nhẹ

### 4.1 Metadata hiện tại

Hiện tại có:

- company
- year
- sector

### 4.2 Vì sao hiện tại chưa đủ mạnh

Metadata đang bị dùng quá nông:

- chưa đi vào biểu diễn chunk một cách đầy đủ,
- chưa tạo hard-negative đúng bản chất,
- chưa đi vào reasoning như một ràng buộc logic.

### 4.3 Nên mở rộng metadata thành gì

Đề xuất schema mở rộng:

- company_id
- company_aliases
- sector
- industry
- report_type
- fiscal_year
- fiscal_quarter
- period_start
- period_end
- statement_type
- currency
- unit_scale
- section
- table_id
- row_header_path
- column_header_path

### 4.4 Metadata nên được dùng ở đâu

1. pre-filtering
2. contextual chunk embedding
3. hard-negative mining
4. operand compatibility check
5. verifier hậu nghiệm

---

## 5. Vai trò của KG

### 5.1 Vai trò hiện tại

KG hiện tại giúp:

- chuẩn hóa khái niệm tài chính,
- mang thêm tín hiệu cấu trúc vào retrieval,
- tạo graph embedding,
- tạo constraint score.

### 5.2 Vai trò tương lai cần có

KG cần trở thành một evidence graph dùng chung cho:

- retrieval phân cấp,
- định vị operand,
- gợi ý phương trình,
- và kiểm chứng reasoning.

### 5.3 Chốt định hướng đúng

Không nên dừng ở:

- KG để tăng MRR.

Nên chuyển sang:

- KG / evidence graph để bắc cầu từ retrieval sang reasoning.

---

## 6. Bài toán reasoning dưới top-k nhiễu

Đây là nút thắt quan trọng nhất.

Trong thực tế, module reasoning thường nhận:

- 1 context đúng,
- 2 context sai,
- hoặc 2 context gần đúng nhưng lẫn metric / year.

Nếu đưa nguyên top-k context vào LLM, mô hình rất dễ:

- trộn toán hạng giữa các context,
- lấy nhầm năm,
- lấy nhầm dòng,
- hoặc dùng số ở footnote như số chính.

### 6.1 Giải pháp đề xuất

Không reasoning trực tiếp trên top-k context.

Phải thêm một bước:

1. tách evidence atom
2. chấm điểm atom theo query
3. dựng local evidence subgraph
4. reasoning chỉ trên graph con này

Đây là chìa khóa để biến retrieval còn nhiễu thành reasoning vẫn đáng tin.

---

## 7. So sánh các thuật toán huấn luyện

## 7.1 DPO

Ưu điểm:

- ổn định
- offline
- dễ triển khai hơn RL online

Phù hợp nhất cho:

- hậu tinh chỉnh reasoning trace
- so sánh trace đúng và trace sai
- dạy mô hình tránh dùng bằng chứng nhiễu

Hạn chế:

- tối ưu ở mức toàn câu trả lời
- không mạnh bằng reward verifiable cho bài toán số học

Kết luận:

- nên dùng ở giai đoạn giữa, không phải thuật toán cuối cùng.

## 7.2 ORPO

Ưu điểm:

- rẻ hơn
- dễ làm baseline

Hạn chế:

- không phải công cụ mạnh nhất cho reasoning có executor

Kết luận:

- phù hợp làm baseline compute thấp, không phải chiến lược chính.

## 7.3 GRPO

Ưu điểm:

- phù hợp bài toán có reward tính được tự động
- rất hợp với numerical reasoning có executor
- không cần critic riêng như nhiều phương pháp RL khác

Hạn chế:

- reward thiết kế kém sẽ dẫn tới reward hacking
- cần verifier tốt

Kết luận:

- đây là thuật toán chính phù hợp nhất cho giai đoạn tối ưu reasoning cuối.

## 7.4 Step-DPO / process-level preference

Điểm mạnh:

- phù hợp với reasoning nhiều bước
- xử lý được lỗi trung gian như:
  - sai year
  - sai operand
  - sai unit
  - sai phép toán

Kết luận:

- nếu dùng DPO, nên ưu tiên biến thể theo bước hơn là DPO thuần.

---

## 8. Chiến lược huấn luyện nên chọn

Chiến lược tốt nhất là huấn luyện nhiều giai đoạn:

### Giai đoạn 0

Xây dữ liệu, metadata schema, evidence graph, executor, verifier.

### Giai đoạn 1

Huấn luyện retrieval:

- dense / sparse hybrid
- metadata-aware ranking
- hard negatives có cấu trúc

### Giai đoạn 2

Huấn luyện reasoning warm start bằng SFT:

- chương trình
- trace grounding
- bằng chứng đúng

### Giai đoạn 3

Offline alignment bằng Step-DPO:

- chosen: trace đúng, grounded, executable
- rejected: trace lẫn context nhiễu, sai year, sai row, sai program

### Giai đoạn 4

Tối ưu reasoning bằng GRPO / RLVR với reward verifiable:

- đúng đáp án
- đúng chương trình
- đúng grounding
- đúng company/year
- đúng unit/scale
- đúng constraint

---

## 9. Ý tưởng đóng góp nên chốt

### Đóng góp 1

Metadata-aware hierarchical retrieval cho tài liệu tài chính dài.

### Đóng góp 2

Equation-centric financial evidence graph dùng chung cho retrieval và reasoning.

### Đóng góp 3

Reasoning có thể kiểm chứng bằng executor + verifier, tối ưu bằng Step-DPO + GRPO.

---

## 10. Tại sao đây là hướng nên triển khai

Hướng này giải quyết đồng thời ba điểm yếu lớn nhất của hệ hiện tại:

1. retrieval hiện vẫn quá coarse
2. metadata chưa phát huy hết
3. reasoning chưa có nền bằng chứng và kiểm chứng

Nó cũng giúp bài báo có câu chuyện mạnh hơn:

- không chỉ tăng MRR,
- mà giải quyết cả bài toán “retrieve đúng bằng chứng và reasoning đúng dưới điều kiện nhiễu”.

---

## 11. Kết luận cuối cùng

Chiến lược nên theo là:

1. nâng retrieval thành hierarchical + metadata-aware
2. thay KG cũ bằng evidence graph trung thành với phương trình
3. thêm evidence triage trước reasoning
4. thêm executor và verifier
5. huấn luyện reasoning theo pipeline:
   - SFT
   - Step-DPO
   - GRPO

Đây là hướng vừa khả thi, vừa có chiều sâu, vừa có khả năng tạo ra đóng góp đủ mạnh cho một công trình nghiên cứu nghiêm túc.
