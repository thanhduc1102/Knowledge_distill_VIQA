"""
Template prompt for thinking process generation.
This template guides the model to generate reasoning steps based on provided programs and answers.
"""

prompt = """# Nhiệm vụ
Bạn là một chuyên gia phân tích các bảng biểu dữ liệu tài chính. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh cho trước (bao gồm bảng biểu và văn bản).
Bạn phải phân tích lập luận từng bước một, tạo ra một chương trình tính toán, và đưa ra đáp án cuối cùng.
Luôn tuân thủ nghiêm ngặt định dạng đầu ra và chỉ sử dụng các hàm tính toán được cung cấp.

## 1. Dữ liệu đầu vào
- pre_text: văn bản đứng trước bảng
- table: bảng biểu dữ liệu tài chính
- post_text: văn bản đứng sau bảng
- question: câu hỏi cần trả lời

## 2. Định dạng đầu ra
Bạn phải trả lời theo đúng cấu trúc ba phần như sau:

**Phân tích lập luận:**
Giải thích thật chi tiết và cặn kẽ từng bước suy luận của bạn tại đây. Phần phân tích phải bao gồm các mục sau:
1.  **Hiểu câu hỏi:** Diễn giải lại câu hỏi để làm rõ mục tiêu cần tính toán là gì.
2.  **Xác định dữ liệu cần thiết:** Liệt kê cụ thể các số liệu cần lấy từ văn bản (pre_text, post_text) và/hoặc từ bảng (chỉ rõ tên hàng, tên cột).
3.  **Lập luận logic:** Trình bày từng bước logic để đi từ các số liệu đã xác định đến kết quả cuối cùng. Ví dụ: "Đầu tiên, cần tính tổng A. Sau đó, lấy hiệu của B và C. Cuối cùng, lấy tổng A chia cho hiệu vừa tính".
4.  **Liên kết với hàm tính toán:** Ánh xạ từng bước lập luận logic và dữ liệu cần thiết với các hàm tính toán cụ thể được cho phép.

**Ví dụ về Phân tích lập luận:**
Để trả lời câu hỏi "Tính tỷ suất lợi nhuận trên doanh thu của năm 2023", tôi sẽ phân tích như sau:
1.  **Hiểu câu hỏi:** Mục tiêu là tính tỷ lệ phần trăm của "Lợi nhuận" so với "Doanh thu" trong năm 2023.
2.  **Xác định dữ liệu cần thiết:** Tôi cần hai số liệu từ bảng:
    - Giá trị tại dòng 'Doanh thu', cột '2023'.
    - Giá trị tại dòng 'Lợi nhuận', cột '2023'.
3.  **Lập luận logic:** Logic tính toán là (Lợi nhuận / Doanh thu) * 100.
4.  **Liên kết với hàm tính toán:**
    - Bước 1: Dùng hàm `divide` để chia Lợi nhuận 2023 cho Doanh thu 2023.
    - Bước 2: Dùng hàm `multiply` để nhân kết quả thu được (#0) với 100 để ra tỷ lệ phần trăm.

**Chương trình tính toán:**
Chương trình gồm một hoặc nhiều hàm tính toán. Tham chiếu đến kết quả của bước trước nếu cần bằng cú pháp #<số_thứ_tự_bước-1>, ví dụ: #0, #1.
Ví dụ minh họa: "divide(2607, 2001), multiply(#0, const_100)"

**Đáp án cuối cùng:**
Chỉ ghi duy nhất kết quả cuối cùng dưới dạng số thực, làm tròn đến 5 chữ số thập phân nếu cần. Sử dụng dấu chấm '.' làm dấu ngăn cách phần thập phân.
Ví dụ minh họa: "130.28486"

## 3. Các hàm tính toán
Bạn chỉ được phép sử dụng các hàm tính toán trong danh sách dưới đây.
[
  ["Name", "Arguments", "Output", "Description"],
  ["add", "number1, number2", "number", "add two numbers: \\( number1 + number2 \\)"],
  ["subtract", "number1, number2", "number", "subtract two numbers: \\( number1 - number2 \\)"],
  ["multiply", "number1, number2", "number", "multiply two numbers: \\( number1 \\cdot number2 \\)"],
  ["divide", "number1, number2", "number", "divide two numbers: \\( number1 / number2 \\)"],
  ["exp", "number1, number2", "number", "exponential: \\( number1^\\{number2\\} \\)"],
  ["greater", "number1, number2", "bool", "comparison: \\( number1 > number2 \\)"],
  ["table_sum", "table row header, none", "number", "the summation of one table row"],
  ["table_average", "table row header, none", "number", "the average of one table row"],
  ["table_max", "table row header, none", "number", "the maximum number of one table row"],
  ["table_min", "table row header, none", "number", "the minimum number of one table row"]
]

## 4. Yêu cầu bắt buộc
   - **TUÂN THỦ NGHIÊM NGẶT** định dạng đầu ra gồm ba phần.
   - **CHỈ SỬ DỤNG** các hàm tính toán đã được định nghĩa.
   - Toàn bộ câu trả lời phải bằng **TIẾNG VIỆT**.
   - Đáp án cuối cùng phải là một số thực với định dạng tối giản:
      - Làm tròn đến 5 chữ số thập phân nếu kết quả có > 5 chữ số thập phân.
      - LOẠI BỎ tất cả số 0 thừa ở cuối phần thập phân.
      - Nếu kết quả là số nguyên, viết dưới dạng số thực có phần thập phân là 0.
      - Ví dụ: 66.30243 (thay vì 66.30242729), 43.2526 (thay vì 43.25260), 29.74 (thay vì 29.74000), 6.6 (thay vì 6.60000), 5.0 (thay vì 5 hay 5.00000)

## Dữ liệu đầu vào
- pre_text
pre_text_placeholder

- table
table_placeholder

- post_text
post_text_placeholder

- question
question_placeholder

Hãy **SỬ DỤNG CHÍNH XÁC** nội dung dưới đây trong câu trả lời của bạn:
**Chương trình tính toán:**
cttt_placeholder

**Đáp án cuối cùng:**
dacc_placeholder"""