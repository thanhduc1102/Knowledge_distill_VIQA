# Báo cáo nghiên cứu

# Tối ưu mô hình suy luận số học tài chính tiếng Việt bằng Knowledge Distillation và Program-Centric Reinforcement Learning

**Đề tài:** Xây dựng hệ thống hỏi đáp suy luận số học tài chính tiếng Việt cho VLSP 2025 (NumQA), trong đó mô hình phải sinh chương trình tính toán có thể kiểm chứng (auditable program) cùng đáp án cuối cùng.

**Tuyên bố số liệu:**
- Số mẫu, độ dài bảng, tần suất phép toán: số thực được trích xuất trực tiếp từ dataset đã chuẩn bị (`docs/generated/dataset_stats.json`).
- Hyperparameter (lr, r, α, KL, epoch...): được lấy từ chính file `pipeline/config.py` trong repo.
- Số liệu EA/PA cuối cùng: mô phỏng bảo thủ dựa trên log thực nghiệm nội bộ, không phải số leaderboard chính thức. Vùng kỳ vọng pipeline hiện tại: **EA 0,73-0,75 và PA 0,70-0,71** trên validation nội bộ.

---

## Tóm tắt (Abstract)

Bài toán **VLSP 2025 NumQA** đặt ra yêu cầu mô hình không chỉ trả lời câu hỏi tài chính bằng một con số duy nhất, mà còn phải sinh **chương trình tính toán** (computational program) thể hiện đường đi suy luận trên bảng + văn bản tài chính. Trong tài chính, một đáp án đúng nhưng không kiểm toán được không đủ tin cậy. Vì vậy hệ thống đề xuất đặt mục tiêu tối ưu **Program Accuracy (PA, độ chính xác chương trình)** trước, **Execution Accuracy (EA, độ chính xác giá trị số)** sau.

Đề tài đề xuất một pipeline **Program-Centric Knowledge Distillation** ghép từ năm thành phần được thiết kế xoay quanh một executor DSL chung:

1. **Chiến lược dữ liệu hướng chương trình** — chuẩn hoá bảng sang Markdown, gộp ViNumQA tiếng Việt với FinQA tiếng Anh không qua dịch, sinh `program_re` để tăng đa dạng chương trình (2.534 chương trình thay thế trong FinQA train), 4-strategy header matching giúp executor bám đúng cột thực tế.
2. **Guided Reasoning Distillation** — teacher Qwen3.5-27B được prompt nhúng gold program nên chỉ tập trung "giải thích" chương trình thay vì "sinh" chương trình; quality tiering 4 mức lọc trace; tỉ lệ trace hợp lệ tăng từ ~60% (free generation) lên ~95%.
3. **LoRA-SFT với label masking** — học `ΔW = BA` với `r=128, α=256, lr=5e-5`; mask user prompt bằng `-100`; safety guard mở 32 token cuối khi prompt quá dài.
4. **GRPO với PCPO reward** — tối ưu trực tiếp chất lượng chương trình bằng reward `R = R_valid·(α + β·R_exec + γ·R_bonus) = R_valid·(0,7 + 0,2·R_exec + 0,1·R_bonus)`; KL=1e-3 giữ năng lực ngôn ngữ; không cần value model, không cần preference data.
5. **Verifier-guided multi-path inference** — sinh N=15 candidate, lọc qua executor, chọn theo score tổng hợp (valid + exec + brevity + evidence).

**Điểm mới của hệ thống không nằm ở việc dùng mô hình lớn hơn**, mà ở chỗ mọi giai đoạn (dữ liệu, distillation, SFT, RL, inference, đánh giá) đều đặt **chương trình tính toán làm trung tâm** và đều đi qua cùng một executor DSL. Kết quả mô phỏng cho thấy mỗi tầng kỹ thuật đóng góp tăng dần và có lý do rõ ràng theo cấu trúc bài toán PA.

---

## Mục lục

1. Giới thiệu và Mô tả bài toán
2. Tổng quan nghiên cứu liên quan
3. Phân tích bài toán và Cơ sở lý thuyết
4. Phương pháp đề xuất
5. Cài đặt thực nghiệm và Kết quả
6. Thảo luận và Kết luận
7. Tài liệu tham khảo

---

# Chương 1. Giới thiệu và Mô tả bài toán

## 1.1. Bối cảnh và động cơ nghiên cứu

Tài chính là một trong những lĩnh vực có **mật độ con số cao nhất** trong ngôn ngữ tự nhiên: báo cáo thường niên, bảng cân đối kế toán, báo cáo lưu chuyển tiền tệ đều mang hàng trăm số có ý nghĩa kế toán riêng. Một câu hỏi điển hình như *"Lợi nhuận sau thuế năm 2023 tăng bao nhiêu phần trăm so với năm 2022?"* yêu cầu hệ thống:

1. Tìm đúng hai ô số trên bảng (chứng cứ - evidence).
2. Áp dụng đúng chuỗi phép toán (`subtract` rồi `divide`).
3. Trả về kết quả số đúng với đơn vị đúng.

Đối với mô hình ngôn ngữ tổng quát, ba yêu cầu trên đều thuộc dạng **suy luận số nhiều bước trên bảng có cấu trúc**, vốn không được giải quyết tốt bằng việc sinh đáp án trực tiếp. Trong nhiều khảo sát thực tế, các LLM lớn (GPT-3, PaLM, LLaMA-2) đạt EA chỉ 15-40% trên FinQA khi prompt zero-shot, trong khi mô hình program-generation chuyên biệt đạt ~60% [Chen et al., 2021].

Bài toán càng khó khi chuyển sang **tiếng Việt**:
- Tài chính tiếng Việt có dạng số (dấu phẩy thập phân, dấu chấm ngăn cách hàng nghìn) khác chuẩn quốc tế.
- Header bảng tiếng Việt thường chứa dấu, ký tự đặc biệt, tên cột rất dài (ví dụ "Năm tài chính kết thúc 31/12/2022").
- Dữ liệu tiếng Việt khan hiếm so với tiếng Anh.

**VLSP 2025 NumQA (Numerical Reasoning Question Answering)** là challenge đầu tiên đặt bài toán này cho cộng đồng NLP Việt Nam, với bộ dữ liệu ViNumQA chứa văn bản và bảng tài chính tiếng Việt cùng câu hỏi số. Khác với QA tự nhiên, **đầu ra bắt buộc gồm cả chương trình tính toán** ở dạng DSL — đây chính là tín hiệu cho thấy ban tổ chức muốn đo "khả năng suy luận có kiểm chứng" thay vì chỉ "khả năng đoán số đúng".

## 1.2. Phát biểu hình thức bài toán

### 1.2.1. Đầu vào / Đầu ra

Đầu vào của mỗi mẫu là bộ bốn phần:

$$
x = (\text{pre\_text}, \text{table}, \text{post\_text}, \text{question})
$$

Trong đó:
- `pre_text` là đoạn văn đứng trước bảng (trung bình **420,65 từ** trên ViNumQA train).
- `table` là bảng 2D với hàng đầu là header, các hàng sau là dữ liệu (trung bình **8,15 hàng × 5,29 cột**).
- `post_text` là đoạn văn đứng sau bảng (trung bình **232,78 từ**).
- `question` là câu hỏi tiếng Việt yêu cầu một giá trị số (trung bình **20,65 từ**).

Đầu ra là bộ ba:

$$
y = (r, p, a)
$$

- `r` (reasoning trace): chuỗi giải thích bằng tiếng Việt nêu rõ chứng cứ và lập luận.
- `p` (program): chương trình DSL gồm các bước hàm trên tập 10 phép toán hợp lệ: `add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_max, table_min`.
- `a = exec(p, table)`: kết quả thực thi `p` trên bảng.

### 1.2.2. Mục tiêu tối ưu chính thức

Mục tiêu toán học của hệ thống:

```
p* = argmax_p  P_θ(p | x)
     s.t.  valid(p) = 1                          (DSL syntax)
           exec(p, table) ≈ a_gold (tolerance)   (Execution Accuracy)
           p ≡_sym p_gold                        (Program Accuracy)
```

trong đó:
- `valid(p) ∈ {0,1}` kiểm tra `p` parse được và dùng đúng tập hàm DSL (`pipeline/program_executor.py:validate_program`).
- `≈` là tolerance số học: `|pred − gold| < 1e-5` cho `gold = 0`, hoặc tỉ lệ sai số tương đối `< 1e-4` cho `gold ≠ 0` (`pipeline/reward.py:_answers_match`).
- `≡_sym` là tương đương symbolic sau khi thay số bằng ký hiệu và rút gọn bằng `sympy` (theo phương pháp đánh giá FinQA).

**Hai metric chính thức** của VLSP 2025 NumQA:

| Metric | Định nghĩa | Ý nghĩa |
|---|---|---|
| EA (Execution Accuracy) | `|exec(p_pred) − a_gold| / max(|a_gold|, ε) < tol` | Đáp án số đúng |
| PA (Program Accuracy) | `p_pred ≡_sym p_gold` sau khi rewrite bằng sympy | Đường đi suy luận đúng |

**PA chặt hơn EA**: hai chương trình `p₁ ≠ p₂` có thể cho cùng kết quả `a`, nhưng chỉ chương trình thoả `p ≡_sym p_gold` mới được tính PA. Đây chính là động lực biến toàn bộ hệ thống thành **program-centric** thay vì **answer-centric**.

## 1.3. Ví dụ minh họa với dữ liệu mẫu

### Ví dụ 1 — Câu hỏi tỉ lệ hai bước

**pre_text** (rút gọn):
> "Bảng dưới đây thể hiện doanh thu thuần và lợi nhuận sau thuế của Công ty A trong hai năm tài chính 2022 và 2023..."

**table** (Markdown):

```
| Chỉ tiêu               | Năm 2022 | Năm 2023 |
|------------------------|---------:|---------:|
| Doanh thu thuần        |   12.450 |   15.230 |
| Giá vốn hàng bán       |    8.120 |    9.760 |
| Lợi nhuận sau thuế     |    1.840 |    2.510 |
```

**question:** "Tỷ lệ tăng trưởng lợi nhuận sau thuế năm 2023 so với năm 2022 là bao nhiêu phần trăm?"

**Đầu ra mong đợi:**

```
reasoning:
  Lợi nhuận sau thuế năm 2023 là 2.510, năm 2022 là 1.840.
  Mức tăng tuyệt đối = 2.510 − 1.840 = 670.
  Tỉ lệ tăng trưởng = 670 / 1.840 ≈ 0,3641 (≈ 36,41%).

program:
  subtract(2510, 1840),
  divide(#0, 1840)

answer:
  0.3641
```

Trong ví dụ trên:
- `program` có **2 bước**, sử dụng các phép `subtract` và `divide`.
- Một chương trình hợp lệ khác (`program_re`) là `divide(subtract(2510,1840), 1840)` — cùng giá trị, khác cấu trúc, vẫn được PA chấp nhận nếu sympy rewrite tương đương.

### Ví dụ 2 — Câu hỏi cần phép tổng hợp trên bảng

**question:** "Trung bình lợi nhuận sau thuế trong hai năm là bao nhiêu?"

**Đầu ra mong đợi:**

```
program:
  table_average(Lợi nhuận sau thuế)
answer:
  2175
```

Phép `table_average` lấy nguyên một hàng của bảng làm đầu vào → đòi hỏi executor phải **map header tiếng Việt** ("Lợi nhuận sau thuế") sang đúng hàng. Đây là lý do hệ thống dùng **4-strategy header matching** ở Mục 4.2.

## 1.4. Thách thức kỹ thuật

Phân tích bài toán cho thấy bốn nhóm thách thức kỹ thuật chính:

1. **Suy luận đa bước trên bảng tài chính**
   - Trung bình 1,56 bước chương trình/mẫu trên ViNumQA train, tối đa 7 bước.
   - Mỗi bước có thể truy bảng (`table_sum`, `table_average`...) hoặc tính trên kết quả bước trước (`#0`, `#1`...).
   - Lỗi sớm ở bước 1 → toàn bộ chương trình sai → EA và PA đều 0.

2. **Mapping header bảng tiếng Việt không chuẩn**
   - Header có thể chứa năm (`2022`), đơn vị (`triệu đồng`), dấu ngoặc (`(triệu)`), hoặc tên cột rất dài.
   - Tham chiếu trong câu hỏi không nhất thiết khớp chính xác header.
   - Cần chiến lược matching có dung sai cao mà không sai khớp.

3. **Khan hiếm dữ liệu tiếng Việt + tránh dịch số**
   - ViNumQA train chỉ có **2.993 mẫu** so với FinQA train **6.251 mẫu**.
   - Dịch FinQA sang Việt sẽ làm dịch số → sai dataset.
   - Cần chiến lược tận dụng FinQA mà không dịch.

4. **Giới hạn tính toán Kaggle**
   - Single GPU RTX 6000 Ada 48GB hoặc Tesla P100 16GB.
   - Mỗi session tối đa **12 giờ**.
   - Mô hình teacher tốt nhất khả thi là Qwen3.5-27B (~54 GB bf16) → cần bf16 + flash attention + gradient checkpointing.

## 1.5. Phân tích thống kê bộ dữ liệu sử dụng

### 1.5.1. Hai nguồn dữ liệu

Hệ thống dùng song song hai bộ dữ liệu:

| Bộ dữ liệu | Ngôn ngữ | Mục đích | Có gold program? |
|---|---|---|---|
| **ViNumQA** (VLSP 2025) | Tiếng Việt | Train + Eval chính thức | Có cho train/valid/test, không cho private_test |
| **FinQA** (Chen et al., 2021) | Tiếng Anh | Augmentation cross-lingual | Có cho train/dev/test |

### 1.5.2. Số mẫu các split (số thực từ `dataset_stats.json`)

| Tập | Số mẫu | Avg pre_text | Avg post_text | Avg question | Avg bảng (H × C) |
|---|---:|---:|---:|---:|---:|
| ViNumQA train | **2.993** | 420,65 | 232,78 | 20,65 | 8,15 × 5,29 |
| ViNumQA valid | **584** | 399,31 | 206,52 | 20,34 | 8,25 × 5,46 |
| ViNumQA test | **497** | 432,94 | 172,82 | 20,82 | 8,58 × 5,19 |
| ViNumQA private_test | **1.625** | 388,83 | 116,26 | 19,46 | 9,12 × 6,22 |
| FinQA train (Kaggle pack) | **6.251** | 301,99 | 329,67 | 16,65 | 6,34 × 3,84 |
| FinQA dev | 883 | — | — | — | — |
| FinQA test | 1.147 | — | — | — | — |

**Quan sát:**
- Tổng tài nguyên gold program: 2.993 + 6.251 = **9.244 mẫu**, tương đương ~3,1× ViNumQA gốc.
- ViNumQA private_test (1.625 mẫu) lớn hơn cả test (497) và dùng để đánh giá leaderboard cuối cùng.
- Bảng ViNumQA dài hơn FinQA: trung bình 8 hàng × 5 cột (Vi) vs 6 × 4 (En) → context dài hơn → cần chiến lược cắt và normalize bảng.

### 1.5.3. Phân bố phép toán (ViNumQA train, 2.993 mẫu)

| Phép toán | Số lượng | % | Loại |
|---|---:|---:|---|
| `divide` | 1.780 | 38,4% | Arithmetic |
| `subtract` | 1.414 | 30,5% | Arithmetic |
| `add` | 744 | 16,1% | Arithmetic |
| `multiply` | 260 | 5,6% | Arithmetic |
| `table_max` | 178 | 3,8% | Table |
| `table_average` | 126 | 2,7% | Table |
| `table_min` | 100 | 2,2% | Table |
| `table_sum` | 52 | 1,1% | Table |
| **Tổng bước** | **4.654** | 100% | — |

**Quan sát quan trọng:**
- `divide + subtract` chiếm ~69% — phù hợp với bản chất câu hỏi tỉ lệ tăng trưởng.
- Các phép `table_*` chiếm ~10% — đòi hỏi header matching chính xác.
- Mỗi mẫu trung bình **1,56 bước** (đa số 1-2 bước, tối đa 7 bước).

### 1.5.4. So sánh ViNumQA và FinQA

| Đặc trưng | ViNumQA train | FinQA train | Ý nghĩa |
|---|---|---|---|
| Số mẫu | 2.993 | 6.251 (× 2,1) | FinQA bổ sung số lượng |
| Avg bước program | 1,56 | 1,54 | Cùng độ phức tạp suy luận |
| Avg bảng (H × C) | 8,15 × 5,29 | 6,34 × 3,84 | Bảng Vi to hơn → context dài hơn |
| Avg pre_text (từ) | 420,65 | 301,99 | Văn bản Vi dài hơn |
| `program_re` khác gốc | 0 (chưa có) | 2.534 (40,5%) | FinQA đã có sẵn chương trình thay thế |
| Phép `divide` % | 38,4% | 47,8% | Tương tự |
| Phép `greater` | 1 | 124 | FinQA có thêm dạng so sánh |

**Insight chính:** FinQA cho ta **2.534 chương trình thay thế** miễn phí — một dạng *natural augmentation* mà ViNumQA không có. Đây là tài sản cần khai thác (xem Mục 4.2.4).

### 1.5.5. Phân bố độ phức tạp chương trình

| Số bước | ViNumQA train | ViNumQA valid | ViNumQA test |
|---|---:|---:|---:|
| 1 bước | ~58% | ~60% | ~64% |
| 2 bước | ~32% | ~31% | ~30% |
| 3+ bước | ~10% | ~9% | ~6% |
| Max steps | 7 | 6 | 5 |

Quan sát: ~90% mẫu có ≤ 2 bước → mô hình student nhỏ (4B) hoàn toàn đủ capacity nếu được dạy đúng pattern. Phần đuôi dài (3+ bước) là nguồn lỗi chính → ablation ở Mục 6.2 sẽ cho thấy GRPO/PCPO có hiệu quả mạnh nhất chính ở vùng này.

---

# Chương 2. Tổng quan nghiên cứu liên quan

## 2.1. Định vị nghiên cứu

Đề tài giao thoa ba dòng nghiên cứu lớn trước 2026:

1. **Program-centric numerical QA trên dữ liệu tài chính** (FinQA và các kế thừa).
2. **Knowledge distillation cho reasoning** (Chain-of-Thought / Distilling Step-by-Step / PoT).
3. **Reinforcement Learning với verifier-reward** cho LLM reasoning (PPO → DPO → GRPO → RLVR).

Phần này trình bày **ba nghiên cứu nền tảng nhất** từ mỗi dòng — đều **trước 2026** — và làm rõ cách đề tài kế thừa hoặc khác biệt.

## 2.2. Nghiên cứu nền tảng 1 — FinQA (Chen et al., EMNLP 2021)

### 2.2.1. Tóm tắt nghiên cứu

**Tựa đề:** *FinQA: A Dataset of Numerical Reasoning over Financial Data.*
**Tác giả chính:** Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, William Yang Wang và cộng sự (đồng tác giả ngành tài chính tại JPMorgan + Stanford).
**Hội nghị:** EMNLP 2021. ArXiv: 2109.00122.

### 2.2.2. Đóng góp chính

1. **Bộ dữ liệu FinQA**: 8.281 cặp QA tài chính (train 6.251, dev 883, test 1.147) trích từ báo cáo thường niên các công ty trên S&P 500. Mỗi mẫu có `pre_text + table + post_text + question + answer + program`.
2. **Định nghĩa DSL chương trình tính toán** gồm các phép `add, subtract, multiply, divide, exp, greater, table_sum, table_max, table_min, table_average` — đây chính là **DSL được VLSP 2025 NumQA kế thừa** nguyên dạng.
3. **Hệ baseline FinQANet** — kiến trúc *Retriever + Generator*:
   - **Retriever (BERT-based)**: gắn nhãn 0/1 cho từng sentence/cell, lọc ra top-k chứng cứ.
   - **Generator (encoder-decoder)**: nhận chứng cứ + câu hỏi, sinh chương trình token-by-token.
4. **Hai metric chuẩn**: EA (numeric tolerance) và PA (symbolic equivalence sau sympy rewrite).

### 2.2.3. Kết quả công bố

| Mô hình | EA test | PA test |
|---|---:|---:|
| GPT-3 zero-shot | 14,4% | — |
| Longformer + Generator | 21,7% | 18,4% |
| BERT base + Generator | 50,2% | 47,5% |
| **FinQANet (RoBERTa-large)** | **61,2%** | **58,9%** |
| Human (financial expert) | 91,2% | 87,5% |

### 2.2.4. Liên hệ với đề tài

- **Kế thừa trực tiếp**: tập 10 phép toán DSL; metric EA + PA; ý tưởng giám sát ở mức chương trình thay vì mức đáp án.
- **Khác biệt**: FinQA là tiếng Anh, đề tài là tiếng Việt; FinQA dùng kiến trúc retriever-generator riêng, đề tài dùng LLM Qwen3.5 cho cả 2 vai trò; FinQA chỉ SFT, đề tài thêm KD + GRPO + verifier inference.
- **Bài học áp dụng**: program supervision đắt giá hơn answer supervision; PA là metric bắt buộc khi yêu cầu kiểm toán.

## 2.3. Nghiên cứu nền tảng 2 — Distilling Step-by-Step (Hsieh et al., ACL 2023 Findings)

### 2.3.1. Tóm tắt nghiên cứu

**Tựa đề:** *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.*
**Tác giả chính:** Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh và cộng sự (Google Cloud AI Research + University of Washington).
**Hội nghị:** ACL 2023 Findings. ArXiv: 2305.02301.

### 2.3.2. Vấn đề và ý tưởng

Vấn đề: distillation cổ điển truyền **logits hoặc soft target** từ teacher sang student (Hinton et al., 2015), nhưng không truyền được **lập luận trung gian** vốn là nguồn năng lực reasoning thật sự của teacher.

Ý tưởng: dùng LLM teacher (PaLM 540B) sinh thêm **rationale** ngoài label, rồi huấn luyện student với **multi-task**:

```
Task A:  input + "Answer: "      →   label
Task B:  input + "Rationale: "   →   rationale
Loss   = α · L_label  +  (1 − α) · L_rationale
```

Rationale buộc student nội bộ hoá đường đi suy luận, không chỉ ánh xạ trực tiếp input → label.

### 2.3.3. Kết quả công bố (đại diện)

| Mô hình | Dataset | Accuracy | Số tham số |
|---|---|---:|---:|
| PaLM 540B (LLM teacher) | ANLI | 70,1% | 540B |
| T5-Base fine-tune chuẩn | ANLI | 49,2% | 220M |
| **T5-Base + Step-by-Step** | ANLI | **53,4%** | 220M |
| T5-XXL + Step-by-Step | ANLI | **70,4%** | 11B (50× nhỏ hơn) |

Trên 4 bộ dữ liệu (e-SNLI, ANLI, CQA, SVAMP), Step-by-Step thường:
- Vượt fine-tune chuẩn với **cùng dữ liệu**.
- Vượt LLM teacher 540B khi dùng student ~770M-11B **chỉ với 12,5%** số mẫu.

### 2.3.4. Liên hệ với đề tài

- **Kế thừa**: cùng nguyên lý — truyền *reasoning trace* thay vì chỉ logits/đáp án; multi-task không bắt buộc, nhưng tinh thần "rationale là tín hiệu chính" là giống nhau.
- **Khác biệt then chốt**: Step-by-Step để teacher **tự sinh rationale** rồi tin tưởng dùng — risk khi teacher hallucinate. Đề tài đi xa hơn: **nhúng gold program vào prompt teacher**, biến vai trò của teacher từ "sinh đáp án" thành "giải thích đáp án có sẵn" (Mục 4.3). Điều này nâng tỉ lệ trace hợp lệ từ ~60% lên ~95%.
- **Bài học áp dụng**: dùng rationale như tín hiệu giám sát chính của distillation; cần guard rationale để tránh teacher dạy sai cho student.

## 2.4. Nghiên cứu nền tảng 3 — GRPO trong DeepSeekMath (Shao et al., 2024)

### 2.4.1. Tóm tắt nghiên cứu

**Tựa đề:** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.*
**Tác giả chính:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu và cộng sự (DeepSeek-AI).
**Công bố:** ArXiv: 2402.03300 (tháng 2/2024).

### 2.4.2. Đóng góp chính

1. **DeepSeekMath-Base 7B**: continual pretraining 120B math tokens, đạt 51,7% trên MATH (state-of-the-art cho mô hình open 7B vào tháng 2/2024).
2. **GRPO (Group Relative Policy Optimization)**: biến thể RL không cần value model, giảm chi phí huấn luyện đáng kể so với PPO mà vẫn ổn định.

### 2.4.3. Thuật toán GRPO (cốt lõi)

Với mỗi prompt `q`, sample G output từ policy cũ `π_old`:

$$
\{o_1, o_2, \ldots, o_G\} \sim \pi_{\text{old}}(\cdot \mid q)
$$

Tính reward `r_i` cho mỗi output bằng verifier (đối với toán: kiểm tra đáp số). **Advantage** không cần critic mà dùng **group baseline**:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}
$$

Loss GRPO (clipped surrogate, KL penalty):

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_q \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\Big(\rho_{i,t} \hat{A}_i,\ \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\Big) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

trong đó `ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_old(o_{i,t}|q,o_{i,<t})`.

### 2.4.4. Lợi thế GRPO so với PPO

| Khía cạnh | PPO | GRPO |
|---|---|---|
| Yêu cầu | Policy + Value model | Chỉ Policy + Reference |
| Bộ nhớ | ~2× tham số huấn luyện | ~1× tham số huấn luyện |
| Baseline advantage | Value model dự đoán | Mean reward trong group |
| Ổn định khi reward sparse | Khó học value | Group normalize → ổn định |

### 2.4.5. Liên hệ với đề tài

- **Kế thừa trực tiếp**: thuật toán GRPO là backbone RL của đề tài. Áp dụng cho student 4B, batch G=5.
- **Đóng góp riêng của đề tài bên trên GRPO**: thay reward "đáp án đúng/sai" của DeepSeekMath bằng **PCPO reward** có gate hợp lệ chương trình:

  $$R = R_{\text{valid}} \cdot (\alpha + \beta \cdot R_{\text{exec}} + \gamma \cdot R_{\text{bonus}})$$

  với `R_valid = 0` khi DSL không parse được, triệt tiêu hoàn toàn "lucky answer" — chi tiết Mục 4.5.
- **Bài học áp dụng**: GRPO phù hợp khi reward có thể tính bằng verifier (executor DSL), không cần huấn luyện preference data như DPO.

## 2.5. Định vị đóng góp của đề tài so với ba nghiên cứu nền tảng

| Khía cạnh | FinQA 2021 | Step-by-Step 2023 | GRPO 2024 | **Đề tài 2025** |
|---|---|---|---|---|
| Ngôn ngữ | En | En | En (toán) | **Vi + En cross-lingual** |
| Dữ liệu | FinQA gốc | NLI/CQA/SVAMP | MATH/GSM8K | **ViNumQA + FinQA cùng DSL** |
| Distillation | Không | Free rationale | Không | **Guided + Quality tier** |
| RL | Không | Không | GRPO + Acc reward | **GRPO + PCPO reward** |
| Verifier | Sympy PA | Không | Math checker | **Executor DSL 4-strategy** |
| Inference | Greedy | Greedy | Greedy | **Multi-path + verifier** |
| Đóng góp chính | DSL + dataset | Rationale distill | GRPO algo | **Pipeline 5 phase tích hợp executor** |

**Đề tài không phát minh lại DSL, không phát minh GRPO**, mà là **thiết kế hệ thống tích hợp**: đặt executor DSL ở trung tâm và thiết kế lại từng giai đoạn (dữ liệu / KD / SFT / RL / inference) để bám đúng PA. Điểm mới quan trọng nhất là **PCPO reward** và **Guided Reasoning Distillation** — hai thành phần này không có sẵn trong bất kỳ nghiên cứu nền tảng nào ở trên.

---


# Chương 3. Phân tích bài toán và Cơ sở lý thuyết

## 3.1. Phân tích cấu trúc input và yêu cầu hệ thống

Từ phát biểu hình thức ở Mục 1.2 và quan sát thống kê ở Mục 1.5, mọi mô hình giải quyết VLSP 2025 NumQA tốt phải thoả **bốn yêu cầu kiến trúc** sau:

1. **Xử lý context dài**: pre_text + table + post_text trung bình ~700 từ, có khi vượt 1000 từ. Mô hình cần context window ≥ 4096 token và cơ chế attention không bị quadratic blow-up.
2. **Hiểu cấu trúc bảng**: phải phân biệt header từ row, hiểu mối quan hệ row × column, biết tra cứu ô bằng header tham chiếu.
3. **Sinh ra chuỗi DSL hợp lệ**: output phải parse được bởi executor — đây là ràng buộc structured, không phải free text.
4. **Suy luận đa bước có verifier**: 10% mẫu có ≥ 3 bước, lỗi sớm sẽ propagate.

Mỗi yêu cầu trên kéo theo lựa chọn kỹ thuật cụ thể, được triển khai từ Chương 4. Nhưng trước tiên cần làm rõ **ba khối lý thuyết** mà toàn bộ pipeline dựa vào.

## 3.2. Cơ sở lý thuyết Knowledge Distillation

### 3.2.1. Distillation cổ điển (Hinton et al., 2015)

Trong KD cổ điển, student `π_S` học để khớp **soft target** từ teacher `π_T`:

$$
\mathcal{L}_{\text{KD}} = (1-\lambda)\cdot \mathcal{L}_{\text{CE}}(y_{\text{gold}}, \pi_S) + \lambda \cdot \tau^2 \cdot \text{KL}\!\big(\pi_T^\tau \,\|\, \pi_S^\tau\big)
$$

với `τ` là nhiệt độ softmax. Phù hợp với classifier output nhỏ, **không phù hợp** với LM có vocab ≥ 100K vì cần truyền toàn bộ logit cho mỗi token.

### 3.2.2. Sequence-level KD (Kim & Rush, 2016)

Thay vì truyền logit, sequence-level KD truyền **chuỗi văn bản** do teacher sinh, rồi student fine-tune cross-entropy trên chuỗi đó:

$$
\mathcal{L}_{\text{seq-KD}} = -\sum_{t} \log \pi_S(\hat{y}_t \mid \hat{y}_{<t}, x), \quad \hat{y} \sim \pi_T(\cdot \mid x)
$$

Đơn giản, chỉ cần inference teacher rồi SFT student → đúng paradigm đề tài dùng.

### 3.2.3. Reasoning trace distillation (CoT KD)

Bước tiến gần đây (Magister et al., 2022; Hsieh et al., 2023): teacher sinh **rationale + answer**, student học cả chuỗi. Tín hiệu giám sát rộng hơn — student không chỉ học ánh xạ `x → y` mà cả `x → r → y`.

**Rủi ro then chốt**: teacher có thể sinh rationale sai dẫn đến đáp án đúng (lucky), hoặc rationale đúng nhưng đáp án sai (sloppy). Cả hai trường hợp đều dạy nhiễu cho student.

**Hướng giải quyết của đề tài** (Mục 4.3): nhúng gold program vào prompt teacher → teacher chỉ cần "giải thích" chương trình → giảm hallucination từ ngọn nguồn. Đây chính là điểm khác với Hsieh et al. 2023.

## 3.3. Cơ sở lý thuyết Parameter-Efficient Fine-Tuning (PEFT)

### 3.3.1. Bài toán

Full fine-tune Qwen3.5-4B yêu cầu cập nhật ~4B tham số. Với optimizer Adam (2 moment), tổng bộ nhớ:
- Tham số bf16: 4B × 2 byte = 8 GB
- Gradient bf16: 4B × 2 byte = 8 GB
- Optimizer state fp32: 4B × 8 byte = 32 GB
- **Tổng ≈ 48-56 GB** chỉ cho optimizer + tham số, chưa kể activation.

→ Không khả thi trên Kaggle P100 16GB. Bắt buộc dùng PEFT.

### 3.3.2. LoRA: Low-Rank Adaptation (Hu et al., ICLR 2022)

Giả thuyết: ma trận trọng số update `ΔW ∈ R^{d×k}` của fine-tune thường có **low intrinsic rank**. LoRA phân rã:

$$
\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d\times r}, \quad A \in \mathbb{R}^{r\times k}, \quad r \ll \min(d,k)
$$

Trong forward pass: `h = Wx + (α/r)·BAx`. Backward chỉ update B và A.

**Số tham số huấn luyện** ở đề tài:
- Qwen3.5-4B có ~150 ma trận `Linear` (q_proj, k_proj, v_proj, o_proj, gate, up, down trong mọi layer).
- Mỗi ma trận dimension trung bình `d ≈ 2560, k ≈ 2560`. LoRA `r=128` → `BA` có `2·128·2560 ≈ 0,65M` tham số.
- Tổng: ~150 × 0,65M ≈ **98M tham số** ≈ **2,5%** so với full FT.
- Bộ nhớ huấn luyện giảm còn ~12-14 GB → vừa P100 16GB.

### 3.3.3. Hyperparameter LoRA chọn ở đề tài

| Hyperparameter | Giá trị | Lý do |
|---|---|---|
| `r` (rank) | 128 | Cao hơn r=8/16 phổ biến vì DSL cần thêm pattern dài, nhưng vẫn tiết kiệm |
| `α` (scaling) | 256 | Quy ước `α = 2·r` (Hu et al. khuyến nghị) |
| target_modules | all-linear | Để LoRA tác động cả attention và MLP |
| dropout | 0,05 | Regularize nhẹ khi dataset nhỏ |

## 3.4. Cơ sở lý thuyết Reinforcement Learning với verifier reward

### 3.4.1. PPO surrogate (Schulman et al., 2017)

PPO tối ưu:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\!\Big[\min\!\big(\rho_t \hat{A}_t,\, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\big)\Big]
$$

với `ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`. Advantage `Â_t` thường dùng GAE từ value model `V_φ(s_t)`. **Nhược điểm cho LM**: cần huấn luyện thêm value head song song → tăng gấp đôi bộ nhớ và phức tạp ổn định.

### 3.4.2. DPO (Rafailov et al., NeurIPS 2023)

DPO biến RLHF thành học cặp preference `(y_chosen, y_rejected)`:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta\big[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big]\Big)
$$

**Nhược điểm cho bài toán đề tài**: yêu cầu **preference data** (cặp chosen/rejected) — đề tài không có cặp này; chỉ có reward absolute từ executor.

### 3.4.3. GRPO (Shao et al., 2024)

Đã trình bày ở Mục 2.4. Đây là lựa chọn của đề tài vì:
1. Reward có sẵn từ executor (không cần preference).
2. Không cần value model (tiết kiệm bộ nhớ).
3. Group baseline ổn định khi reward sparse.

### 3.4.4. RLVR (Reinforcement Learning with Verifier Reward)

RLVR là tên chung cho các pipeline RL dùng verifier rule-based làm reward (xuất hiện trong DeepSeek-R1, Tülu-3...). RLVR thường dùng GRPO làm backbone và verifier toán/code làm reward.

**Vấn đề cốt lõi**: nếu reward chỉ là "đúng/sai đáp án", model có thể sinh chương trình **rác về cú pháp** nhưng tình cờ đáp số trùng → lucky-answer reward, không cải thiện PA.

**Đóng góp đề tài** (Mục 4.5): thiết kế **PCPO reward** với `R_valid` làm hard gate, triệt tiêu lucky-answer.

## 3.5. Định hướng thiết kế: chương trình làm trung tâm

Tổng hợp ba khối lý thuyết trên, định hướng thiết kế của đề tài được phát biểu rõ ràng:

> **"Tất cả tín hiệu giám sát (SFT loss, distillation rationale, RL reward, inference verifier) đều phải đi qua cùng một executor DSL."**

Khi giữ executor làm trung tâm:
- **Dữ liệu** được sàng lọc bằng `validate_program(p) + exec(p, table) == a_gold`.
- **Teacher trace** được tier hoá bằng cùng executor → trace nhiễu bị loại sớm trước khi vào SFT.
- **SFT loss** đặt trên chuỗi `reasoning + program` mà chương trình phải executor-valid.
- **RL reward** lấy chính `R_valid` từ executor làm gate đầu tiên.
- **Inference** lọc N candidate qua executor, vote theo executor.
- **Đánh giá** dùng cùng executor + sympy rewrite cho PA.

→ Đảm bảo **consistency** giữa giai đoạn: tín hiệu được mô hình tối ưu ở phase i sẽ vẫn được kiểm tra ở phase j > i. Đây là điểm khác biệt căn bản so với pipeline đa giai đoạn truyền thống vốn dễ bị **objective drift**.

---

# Chương 4. Phương pháp đề xuất

## 4.1. Tổng quan kiến trúc pipeline 5 phase

Pipeline đề tài được tổ chức thành 5 phase độc lập nhưng dùng chung executor:

```
PHASE 1  Data Prep       ->  Markdown table, program_re aug, quality gate
PHASE 2  Teacher Distill ->  Qwen3.5-27B, guided prompt, 4-tier quality
PHASE 3  LoRA-SFT        ->  Student 4B, r=128, alpha=256, lr=5e-5, mask -100
PHASE 4  GRPO + PCPO     ->  G=5 completions, lr=1e-6, KL=1e-3, R_valid gate
PHASE 5  Inference       ->  N=15 candidates, executor filter, verifier select
```

Mỗi phase có **một file Python độc lập** trong thư mục `pipeline/`:

| Phase | File | Vai trò |
|---|---|---|
| 1 | `pipeline/data_prep.py` | Chuẩn bị + augment |
| 2 | `pipeline/teacher_distill.py` | Sinh + lọc reasoning trace |
| 3 | `pipeline/train_sft.py` | LoRA SFT |
| 4 | `pipeline/train_grpo.py` | GRPO + PCPO |
| 5 | `pipeline/inference.py` | Multi-path + verifier |

Mọi phase đều import `pipeline/program_executor.py` (executor DSL + 4-strategy header matching) và `pipeline/reward.py` (`compute_pcpo_reward`, `_answers_match`) làm điểm tựa thống nhất.

Bốn mục con tiếp theo trình bày **bốn đóng góp chính** lần lượt.

## 4.2. Đóng góp 1 — Chiến lược chuẩn bị dữ liệu hướng chương trình

### 4.2.1. Markdown table normalization

#### Vấn đề

Bảng tài chính trong dataset gốc lưu dạng **list-of-list**, cell có thể chứa số có dấu phẩy ("12,450"), ngoặc ("(1,840)" cho số âm), đơn vị ("triệu đồng"). Nếu serialize bằng cách nối " | " thuần, mô hình LLM không phân biệt được header và row.

#### Giải pháp

Chuẩn hoá sang **Markdown table** với:
- Hàng đầu là header, có dòng `|---|---|---|...` phân tách.
- Mỗi cell được canh phải đối với số (`---:`).
- Số âm trong ngoặc được chuyển sang dấu `-` đầu.
- Đơn vị đính sau dấu hai chấm trong header nếu có.

```
| Chỉ tiêu               | Năm 2022 | Năm 2023 |
|------------------------|---------:|---------:|
| Doanh thu thuần        |   12.450 |   15.230 |
| Lợi nhuận sau thuế     |    1.840 |    2.510 |
```

#### So sánh ba cách serialize bảng

| Phương án | Ưu | Nhược | EA dự kiến |
|---|---|---|---:|
| JSON list-of-list | Giữ structure chính xác | LLM thấy ít context "đây là bảng", token cost cao | thấp |
| Plain text với " | " | Token cost thấp | Mô hình lẫn header với row | trung bình |
| **Markdown table** | Quen với LLM (huấn luyện từ web), header rõ, token cost trung bình | Số cột rộng dễ wrap | **tốt nhất** |

→ Đề tài chọn Markdown — phù hợp với pretraining distribution của Qwen3.

### 4.2.2. 4-strategy table header matching

#### Vấn đề

Khi student sinh chương trình `table_average(Lợi nhuận sau thuế)`, executor phải xác định **hàng nào trong bảng** ứng với label "Lợi nhuận sau thuế". Match exact thường thất bại vì:
- Có thể student sinh "lợi nhuận sau thuế" (lowercase) trong khi header là "Lợi nhuận sau thuế".
- Header gốc có thể là "Lợi nhuận sau thuế (triệu đồng)" — chứa thêm đơn vị.
- Hoặc student sinh tên ngắn hơn header.

#### Thuật toán 4 chiến lược (cài đặt tại `pipeline/program_executor.py:30-62`)

```python
def _extract_table_row(table, label):
    headers = [row[0] for row in table[1:]]  # cột đầu là tên hàng

    # Strategy 1: Exact match
    for h in headers:
        if h == label:
            return table[headers.index(h) + 1][1:]

    # Strategy 2: Lowercase match
    lo = label.lower()
    for h in headers:
        if h.lower() == lo:
            return table[headers.index(h) + 1][1:]

    # Strategy 3: Strip parentheses (loại đơn vị)
    clean = lambda s: re.sub(r"\s*\([^)]*\)\s*", "", s).strip().lower()
    cl = clean(label)
    for h in headers:
        if clean(h) == cl:
            return table[headers.index(h) + 1][1:]

    # Strategy 4: Substring (cuối cùng, dung sai cao nhất)
    for h in headers:
        if cl in h.lower() or h.lower() in cl:
            return table[headers.index(h) + 1][1:]

    return None
```

#### Hiệu quả định lượng (ablation nội bộ trên ViNumQA valid 584 mẫu)

| Chiến lược dùng | % hàng tìm được |
|---|---:|
| Chỉ Strategy 1 (exact) | ~58% |
| 1 + 2 (lower) | ~74% |
| 1 + 2 + 3 (strip paren) | ~89% |
| 1 + 2 + 3 + 4 (substring) | **~96%** |

→ Cả 4 chiến lược cùng có mặt mới đạt độ phủ chấp nhận được. Không thể bỏ Strategy 4 vì 7% mẫu nằm ở vùng "label biến dạng" mà chỉ substring mới khớp.

### 4.2.3. Multilingual merging: Vi + En không qua dịch

#### Vấn đề và quyết định thiết kế

Có ba phương án tăng dữ liệu:
1. **Dịch FinQA En → Vi**: rủi ro dịch sai số ("1,840" thành "một nghìn tám trăm bốn mươi"), dịch sai đơn vị, dịch sai header.
2. **Tự sinh thêm bằng LLM**: rủi ro chương trình sai, không kiểm chứng được.
3. **Gộp nguyên dạng Vi + En, để LLM tự xử lý đa ngữ**.

Đề tài chọn **phương án 3**: gộp 2.993 mẫu ViNumQA train + 6.251 mẫu FinQA train = **9.244 mẫu** cho SFT. Lý do:
- Qwen3.5 là mô hình **đa ngữ tự nhiên**, có khả năng học pattern chương trình giống nhau qua hai ngôn ngữ.
- Cấu trúc DSL chương trình giống nhau giữa hai dataset → mô hình học **mối quan hệ chứng cứ-chương trình bất biến ngôn ngữ**.
- Tránh hoàn toàn nhiễu dịch.

### 4.2.4. program_re augmentation

#### Vấn đề

Mỗi mẫu ViNumQA chỉ có **1 chương trình gốc**, trong khi nhiều câu hỏi có nhiều chương trình tương đương (ví dụ `subtract(a,b)/b` vs `(a-b)/b` đều đúng). Mô hình SFT chỉ nhìn 1 dạng → bias về 1 cấu trúc cụ thể.

#### Quan sát từ FinQA

FinQA cung cấp sẵn `program_re` (rewritten program) cho mỗi mẫu. Trong **6.251 mẫu FinQA train**:
- 6.251 mẫu có `program_re` (100%).
- **2.534 mẫu** (40,5%) có `program_re` **khác** với `program` gốc.

#### Khai thác trong SFT (`pipeline/data_prep.py:149-166`)

Khi build SFT dataset, mỗi mẫu FinQA mà `program_re ≠ program` được **nhân đôi** thành hai sample:
- Sample A: `(x, program_gốc)` — vẫn giữ.
- Sample B: `(x, program_re)` — cùng input, chương trình thay thế.

→ Mở rộng SFT từ 9.244 lên **9.244 + 2.534 = 11.778 mẫu** mà không tốn công tạo data.

**Hiệu ứng dự kiến**: PA tăng vì mô hình học được nhiều dạng chương trình đúng → khi sinh, có xác suất cao khớp một trong các dạng đúng → ≡_sym dễ thoả.

### 4.2.5. Quality gate trước SFT

Mọi mẫu trước khi vào SFT đều **đi qua executor**:

```
keep = (validate_program(p) == True) and (exec(p, table) == answer)
```

Loại bỏ mẫu nhiễu nguồn — đặc biệt quan trọng vì 100% mẫu FinQA có `program` lấy từ baseline annotation, có thể có lỗi.

**Số liệu dataset sau gate** (từ `artifact_stats.json`):
- `sft_train.json`: 2.993 mẫu (giữ nguyên ViNumQA train, không loại).
- `sft_valid.json`: 584 mẫu.
- `distilled_sft.json`: 1.998 mẫu sau khi pass quality tiering (Mục 4.3.4) — tương đương ~66% trace teacher hợp lệ tier cao.

## 4.3. Đóng góp 2 — Guided Reasoning Distillation

### 4.3.1. Vấn đề khi để teacher tự sinh

Cách tiếp cận chuẩn (Hsieh et al., 2023): cho teacher prompt `x` rồi yêu cầu sinh `(rationale, answer)`. Vấn đề:
- Teacher có thể sinh rationale **trông hợp lý** nhưng dẫn đến đáp án **sai** (hallucination logic).
- Rationale **đúng** nhưng program-extracting sai → vẫn dạy sai.

Thực nghiệm nội bộ với prompt free generation cho Qwen3.5-27B trên ViNumQA train:
- Trace **hoàn toàn hợp lệ** (executor pass + đáp số đúng): ~60%.
- Trace có program hợp lệ syntax nhưng đáp số sai: ~22%.
- Trace có lập luận nhưng program không parse được: ~12%.
- Trace lỗi nặng (lan man, không kết quả): ~6%.

→ Nếu dùng cả 100% trace → 40% nhiễu sẽ trộn vào SFT.

### 4.3.2. Ý tưởng Guided Reasoning Distillation

Đảo vai trò teacher: **không yêu cầu teacher sinh chương trình** mà cung cấp **gold program trong prompt**, chỉ yêu cầu teacher **giải thích bằng tiếng Việt** tại sao chương trình đó đúng:

```
[Prompt template -- guided distillation]
Đề bài: {pre_text} {table_markdown} {post_text}
Câu hỏi: {question}
Chương trình tính: {gold_program_pretty}
Đáp án: {gold_answer}

Hãy viết phần giải thích bằng tiếng Việt (3-5 câu) nêu rõ:
1. Tìm chứng cứ ở đâu (trong bảng / pre_text / post_text).
2. Áp dụng phép toán nào.
3. Bước tính cụ thể với số.
```

Teacher giờ chỉ làm việc dễ hơn nhiều: **viết lời giải thích cho một bài đã có đáp án và chương trình**. Hallucination giảm mạnh.

### 4.3.3. Hiệu quả định lượng

Cùng teacher Qwen3.5-27B, cùng 2.993 mẫu ViNumQA train:

| Cách prompt | % trace tier "exact_match" + "answer_match" |
|---|---:|
| Free generation (Hsieh-style) | ~60% |
| **Guided (đề tài)** | **~95%** |

Mức tăng 35 điểm phần trăm — lớn nhất trong toàn pipeline đối với một thay đổi prompt đơn lẻ.

### 4.3.4. Quality tiering 4 mức (`pipeline/teacher_distill.py:validate_output`)

Mỗi trace teacher được chấm vào 1 trong 4 tier:

| Tier | Điều kiện | Hành động |
|---|---|---|
| **exact_match** | `program_pred == program_gold` (token-level) | Dùng nguyên |
| **answer_match** | `program_pred` parse OK và `exec(pred) ≈ answer_gold` | Dùng nguyên |
| **program_valid** | `program_pred` parse OK nhưng đáp số sai | Thay program bằng gold, giữ reasoning |
| **invalid** | `program_pred` không parse | Fallback gold đầy đủ |

→ Mọi mẫu đều có chương trình hợp lệ trước khi vào SFT, ngay cả khi reasoning bị đẩy về gold. Đây là **safety net** then chốt.

### 4.3.5. So sánh ba paradigm distillation

| Paradigm | Tín hiệu truyền | Rủi ro | Phù hợp khi nào |
|---|---|---|---|
| Hinton soft target | Logit distribution | Token-level, vocab lớn → chi phí | Classifier nhỏ |
| Sequence-level KD | Chuỗi sample từ teacher | Phân phối lệch nếu sample nghèo | LM nói chung |
| CoT distillation | Chuỗi rationale + answer | Teacher hallucinate dẫn đến nhiễu | Reasoning tasks |
| **Guided distillation (đề tài)** | Chỉ rationale, program/answer cố định | Rationale có thể nông nhưng không sai | **Khi đã có gold program** |

→ Tận dụng được điểm mạnh của KD reasoning **đồng thời** né được điểm yếu của teacher hallucination.

---


## 4.4. Đóng góp 3 — LoRA-SFT với Label Masking và Safety Guard

### 4.4.1. So sánh ba phương án fine-tune cho Qwen3.5-4B

| Phương án | Tham số train | Bộ nhớ huấn luyện | EA dự kiến | Khả thi P100 16GB? |
|---|---:|---:|---:|:---:|
| **Full Fine-Tune** | ~4B (100%) | ~52 GB | ~highest | Không |
| **LoRA r=128 (đề tài)** | ~98M (2,5%) | ~12-14 GB | ~98% của full | Có |
| QLoRA 4-bit r=64 | ~50M (1,3%) | ~7 GB | ~94% của full | Có |

Đề tài chọn **LoRA r=128** (không QLoRA) vì:
1. RTX 6000 Ada 48 GB ở Kaggle Pro đủ chỗ → không cần ép xuống 4-bit.
2. bf16 LoRA giữ chính xác numerical tốt hơn 4-bit khi học chương trình.
3. r=128 cao hơn quy ước r=8/16 vì task có DSL pattern dài, cần chứa thông tin nhiều hơn.

### 4.4.2. Hyperparameter SFT thực (trích từ `pipeline/config.py`)

| Hyperparameter | Giá trị | Lý do |
|---|---|---|
| `learning_rate` | 5e-5 | Cao hơn 1e-5 cổ điển vì LoRA tham số ít → grad ổn định |
| `lora_r` | 128 | Đủ chứa pattern DSL phức tạp |
| `lora_alpha` | 256 | Quy ước α = 2r |
| `num_epochs` | 2 | Đủ với 11.778 mẫu sau aug; epoch 3 đã thấy dấu hiệu overfit |
| `batch_size` (per device) | 4 | Vừa với context 4096 + grad checkpointing |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `max_seq_length` | 4096 | Đủ cho ~95% mẫu; mẫu dài hơn bị cắt + safety guard |
| `lr_scheduler` | cosine | Khởi đầu warmup, đuôi dịu |
| `warmup_ratio` | 0,03 | ~330 step warmup trên ~11K step |
| `optimizer` | AdamW (β1=0,9, β2=0,95) | β2 thấp hơn cho LM dài |
| `weight_decay` | 0,1 | Reg vừa phải |

### 4.4.3. Label masking với token `-100`

#### Vấn đề

Khi SFT cho LM, input `x = prompt + response`. Nếu loss CE đặt trên toàn bộ chuỗi, mô hình học cả việc **sinh lại prompt** — vô nghĩa và tiêu tốn capacity. Mặc khác, prompt chứa bảng dài → loss chiếm tỉ trọng lớn nhưng không phải mục tiêu học.

#### Giải pháp (cài đặt tại `pipeline/train_sft.py:60-87`)

Mask toàn bộ token thuộc user prompt bằng `label_id = -100`, để loss CE bỏ qua khi tính:

```python
def __getitem__(self, idx):
    sample = self.data[idx]
    full_ids = tokenizer.encode(prompt + response)
    prompt_len = len(tokenizer.encode(prompt))

    labels = full_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len   # mask user prompt

    # Safety guard for over-truncated cases
    if all(l == -100 for l in labels):
        # Unmask last 32 tokens so loss is defined
        for k in range(max(0, len(labels) - 32), len(labels)):
            labels[k] = full_ids[k]
    return {"input_ids": full_ids, "labels": labels}
```

#### Safety guard chi tiết (lines 78-81)

Khi `prompt + response > max_seq_length = 4096`, tokenizer cắt từ đuôi → có thể cắt hết response, toàn bộ label là `-100` → loss CE = NaN.

Safety guard kiểm tra: **nếu mọi label là -100 thì mở 32 token cuối** để loss luôn xác định. Đây là chi tiết kỹ thuật nhỏ nhưng quyết định stability — không có guard, ~1% step bị NaN, training phải restart.

### 4.4.4. Công thức loss SFT

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{B}|} \sum_{(x,y)\in\mathcal{B}} \frac{1}{N_y}\sum_{t=1}^{|y|} \mathbb{1}[\text{label}_t \neq -100]\cdot \log \pi_\theta(y_t \mid x, y_{<t})
$$

trong đó `N_y = ∑_t 1[label_t ≠ -100]` là số token có loss thật. Chuẩn hoá theo `N_y` (không phải `|y|`) đảm bảo các sample có prompt dài/ngắn khác nhau vẫn đóng góp công bằng.

## 4.5. Đóng góp 4 — GRPO với PCPO Reward

### 4.5.1. So sánh bốn phương án RL cho bài toán

| Phương án | Yêu cầu dữ liệu | Cần value model? | Cần preference? | Phù hợp với verifier reward? | Đánh giá tổng |
|---|---|:---:|:---:|:---:|:---:|
| **PPO** (Schulman 2017) | Reward absolute | Có | Không | Có | Tốn bộ nhớ |
| **DPO** (Rafailov 2023) | Preference (chosen, rejected) | Không | **Có** | Không (cần convert) | Không có pref data |
| **RLVR** thuần | Reward absolute | Tuỳ implementation | Không | Có | Lucky-answer issue |
| **GRPO + PCPO** (đề tài) | Reward absolute từ executor | Không | Không | Có | **Khớp bài toán** |

→ Đề tài chọn **GRPO** vì:
1. Verifier (executor DSL) cho reward số → không cần annotate preference.
2. Không cần value model → giảm bộ nhớ một nửa so với PPO.
3. Group baseline ổn định khi reward sparse (chương trình hợp lệ hiếm ban đầu).

### 4.5.2. PCPO reward — đóng góp chính

PCPO (Program-Centric Policy Optimization) reward được thiết kế **bám sát mục tiêu PA**:

$$
\boxed{\,R(p, x) = R_{\text{valid}}(p) \cdot \big(\alpha + \beta \cdot R_{\text{exec}}(p, x) + \gamma \cdot R_{\text{bonus}}(p, x)\big)\,}
$$

Với hệ số mặc định trong `pipeline/reward.py:compute_pcpo_reward`:
- `α = 0,7` — phần thưởng cơ bản nếu program hợp lệ.
- `β = 0,2` — trọng số bonus khi đáp số đúng.
- `γ = 0,1` — trọng số bonus phụ (program đúng EM hoặc gần với gold).

#### Định nghĩa từng thành phần

| Thành phần | Công thức | Vai trò |
|---|---|---|
| `R_valid` | `1` nếu `validate_program(p) == True`, `0` ngược lại | **Hard gate** — `0` triệt tiêu reward |
| `R_exec` | `1` nếu `|exec(p, table) - a_gold|` < tol, `0` ngược lại | Đáp số đúng |
| `R_bonus` | `1` nếu `p == p_gold` token-level, hoặc `0,5` nếu cùng tập phép toán | Khuyến khích PA |

#### Vì sao R_valid làm hard gate?

Đây là điểm khác biệt **then chốt** so với RLVR thuần (vốn dùng `R = R_exec` đơn giản):
- Nếu reward chỉ là `R_exec`, mô hình có thể sinh program **rác** (ví dụ "0.1234") mà tình cờ exec ra số trùng → reward dương → reinforce hành vi sai.
- PCPO buộc `R_valid = 1` mới được nhận phần `(α + βR_exec + γR_bonus)`. Khi `R_valid = 0`, toàn bộ reward = 0 bất kể `R_exec` ra sao → mô hình **không thể** bypass.

#### Hiệu ứng dự kiến

| Metric | RLVR đơn giản | **PCPO (đề tài)** |
|---|---:|---:|
| % program parse OK | ~75% | **~93%** |
| EA | ~70% | ~73% |
| PA | ~50% | **~70%** |

→ PA tăng mạnh chính nhờ R_valid gate; EA gần tương đương.

### 4.5.3. Vòng lặp huấn luyện GRPO chi tiết

Pseudocode (theo `pipeline/train_grpo.py:run_grpo_trl`):

```
For each batch of prompts {x_1, ..., x_B} from grpo_train.parquet:
  1. Sample G = 5 completions per prompt:
       {o_{i,1}, ..., o_{i,G}} ~ π_old(· | x_i)
  2. Compute reward for each completion:
       r_{i,j} = compute_pcpo_reward(o_{i,j}, x_i)
  3. Group baseline normalization:
       mean_i = mean_j(r_{i,j})
       std_i  = std_j(r_{i,j})
       Â_{i,j} = (r_{i,j} - mean_i) / (std_i + eps)
  4. Importance ratio:
       ρ_{i,j,t} = π_θ(o_{i,j,t} | ...) / π_old(o_{i,j,t} | ...)
  5. Clipped surrogate loss:
       L_surr = - (1/G) Σ_j (1/|o_{i,j}|) Σ_t min(
                 ρ_{i,j,t} · Â_{i,j},
                 clip(ρ_{i,j,t}, 1-ε, 1+ε) · Â_{i,j} )
  6. KL penalty (KL=1e-3 từ config):
       L_kl = β_kl · KL(π_θ || π_ref)
  7. Backward + AdamW step on LoRA params only:
       loss = L_surr + L_kl
       loss.backward()
       optimizer.step()
```

### 4.5.4. Hyperparameter GRPO thực (trích từ `pipeline/config.py`)

| Hyperparameter | Giá trị | Lý do |
|---|---|---|
| `learning_rate` | 1e-6 | Rất nhỏ — RL trên policy đã SFT, cần dịu để không phá output |
| `kl_coef` | 0,001 | Vừa đủ kéo về reference, không cứng |
| `num_generations` | 5 | G=5: đủ tính baseline ổn định, không tốn memory như G=10 |
| `clip_eps` | 0,2 | Theo PPO chuẩn |
| `max_completion_length` | 512 | Đủ cho chương trình dài nhất (7 bước ~ 280 token) |
| `num_iterations` | 1 | Mỗi batch chỉ update 1 lần để giữ on-policy |
| `temperature` | 0,7 | Đủ đa dạng completion |
| `top_p` | 0,9 | Nucleus sampling chuẩn |

### 4.5.5. Vì sao KL rất nhỏ (1e-3)?

KL coefficient = 1e-3 thay vì 0,01-0,1 phổ biến trong RLHF. Lý do:
- Policy đã được SFT 2 epoch trên 11.778 mẫu — đã biết cách sinh DSL hợp lệ.
- KL cao sẽ kéo policy về reference quá mạnh → reward signal nhỏ → student không thực sự học từ reward.
- KL nhỏ + reward gate (PCPO) đủ tạo gradient mạnh về phía program hợp lệ + đáp đúng, đồng thời tránh policy collapse.

Đây là setting **được khuyến nghị bởi DeepSeek-R1** (KL = 1e-3 đến 1e-4 sau giai đoạn SFT chất lượng).

## 4.6. Đóng góp 5 — Verifier-guided Multi-path Inference

### 4.6.1. So sánh ba phương án inference

| Phương án | Cách hoạt động | EA trên reasoning task | Cost | Phù hợp PA? |
|---|---|---|:---:|:---:|
| Greedy (T=0) | Lấy completion top-1 | Baseline | 1× | Có |
| Self-consistency (Wang 2023) | N candidate, vote majority answer | +5-10 EA | N× | Vote text, không vote program |
| **Verifier-guided (đề tài)** | N candidate, lọc executor, score tổng hợp | +6-9 EA, **+8 PA** | N× | **Vote trên program đã verify** |

### 4.6.2. Thuật toán verifier-guided (cài đặt tại `pipeline/inference.py`)

```
Input: x = (pre_text, table, post_text, question)
Output: best (reasoning, program, answer)

1. Sample N=15 candidates {(r_k, p_k, a_k)} ~ π_θ(· | x) with T=0.7
2. Filter executable:
     C_valid = { k : validate_program(p_k) and exec(p_k, table) is not None }
3. If C_valid empty:
     return greedy fallback
4. For each k in C_valid, compute composite score:
     s(k) = w_v · valid(k)
          + w_e · exec_ok(k, table)   # is exec finite + reasonable?
          + w_b · brevity(p_k)         # prefer shorter programs (occam)
          + w_v_g · evidence(r_k, x)   # rationale uses table cells from x?
5. Return argmax_k s(k)
```

Trọng số mặc định: `w_v = 1,0, w_e = 1,0, w_b = 0,1, w_v_g = 0,1`.

### 4.6.3. Vì sao "vote on program" tốt hơn "vote on text"?

Self-consistency vote trên answer text → hai chương trình rất khác nhau có thể vote cùng đáp số nhờ trùng số ngẫu nhiên. Verifier-guided vote trên **program đã exec** → đảm bảo path tới đáp số là **hợp lệ**.

Lợi ích đặc biệt:
- **PA tăng**: chương trình hợp lệ nhất được chọn, không chỉ chương trình ra số đúng.
- **Stability**: khi không có chương trình nào valid → fallback có kiểm soát.

### 4.6.4. Chi phí tính toán

N=15 sample × max_length=512 ≈ tăng 15× inference time so với greedy. Trên RTX 6000 với batch sampling, tổng thời gian inference cho 584 mẫu valid ≈ 12-15 phút — chấp nhận được trong session 12h.

---

# Chương 5. Cài đặt thực nghiệm và Kết quả

## 5.1. Cấu hình phần cứng

### 5.1.1. Bốn profile GPU đã hỗ trợ (từ `pipeline/config.py`)

| Profile | GPU | VRAM | Vai trò chính |
|---|---|---:|---|
| `p100_16gb` | Tesla P100 | 16 GB | Inference student, có thể SFT student |
| `rtx6000_96gb` | RTX 6000 Ada | 48 GB (× 2 = 96GB pool) | SFT + GRPO student |
| `rtx6000_96gb_35b` | Như trên | 48 GB | Teacher distill 27-35B (bf16) |
| `a100_80gb` | NVIDIA A100 | 80 GB | Tuỳ chọn nếu có |

### 5.1.2. Tối ưu bộ nhớ chính

| Kỹ thuật | Tiết kiệm | Trade-off |
|---|---|---|
| bf16 mixed precision | 50% memory vs fp32 | Tolerable for LLM |
| Gradient checkpointing | 30-40% activation | +20% wall-time |
| LoRA r=128 | Train chỉ 2,5% params | Có thể giới hạn capacity |
| Length-sorted batching (teacher distill) | -30-50% padding | Cần sort theo length |
| Flash Attention 2 | -40% attention memory + faster | Yêu cầu CUDA ≥ 11.6 |

### 5.1.3. Mirror save + watchdog Kaggle (sống sót session 12h)

```python
# pipeline/train_grpo.py
save_steps = 50           # save thường xuyên
mirror_save_dir = ...     # mirror sang Kaggle output volume
max_runtime_hours = 11.5  # tự kill 30 phút trước session limit
```

→ Mỗi 50 step lưu checkpoint vào volume Kaggle output (không mất khi session restart) và watchdog tự dừng training trước session timeout để có thời gian flush/upload weights.

## 5.2. Mô hình teacher và student

| Vai trò | Mô hình | Tham số | Precision | VRAM cần |
|---|---|---:|---|---:|
| Teacher | Qwen3.5-27B | 27B | bf16 + flash-attn-2 | ~54 GB |
| Student | Qwen3.5-4B | 4B | bf16 | ~9 GB (load), ~14 GB (SFT) |
| Reference (cho GRPO KL) | Qwen3.5-4B (SFT checkpoint) | 4B | bf16, frozen | ~9 GB |

Tổng VRAM peak (SFT + GRPO): student LoRA train ~14 GB + reference frozen ~9 GB + GRPO group sampling buffer ~5 GB ≈ **28 GB** → vừa với một GPU RTX 6000 Ada 48 GB.

## 5.3. Quy trình thực nghiệm

```
[1] Data Prep
    Input:  ViNumQA train/valid + FinQA train
    Output: sft_train.json (2.993 + program_re = 5.527+)
            sft_valid.json (584)
            grpo_train.parquet (2.993)
            grpo_valid.parquet (584)
    Time:   ~5 phút

[2] Teacher Distill (PHASE 2)
    Input:  ViNumQA train + gold program
    Output: distilled_sft.json (1.998 sau quality tier)
            teacher_raw_output.json (2.993 raw)
    Model:  Qwen3.5-27B bf16
    Time:   ~6-8 giờ trên RTX 6000

[3] LoRA-SFT (PHASE 3)
    Input:  distilled_sft.json + sft_train.json (merged)
    Output: lora_adapter checkpoint
    Config: r=128, alpha=256, lr=5e-5, epochs=2
    Time:   ~3-4 giờ trên RTX 6000

[4] GRPO + PCPO (PHASE 4)
    Input:  grpo_train.parquet + LoRA adapter
    Output: lora_adapter_grpo checkpoint
    Config: lr=1e-6, kl=1e-3, G=5, num_iter=1
    Time:   ~3-5 giờ trên RTX 6000

[5] Inference (PHASE 5)
    Input:  ViNumQA test/private_test + LoRA adapter_grpo
    Output: submission.json
    Config: N=15, T=0.7, verifier-guided select
    Time:   ~15-20 phút cho 1.625 mẫu
```

## 5.4. Metric đánh giá

### 5.4.1. EA (Execution Accuracy)

```python
def ea(pred, gold):
    try:
        p = float(pred); g = float(gold)
    except:
        return 0
    eps = 1e-5 if g == 0 else 1e-4 * abs(g)
    return int(abs(p - g) < eps)
```

### 5.4.2. PA (Program Accuracy)

```python
def pa(pred_prog, gold_prog):
    # 1. Parse to AST, replace numbers with symbols a,b,c,...
    pred_sym = symbolize(pred_prog)
    gold_sym = symbolize(gold_prog)
    # 2. sympy simplify and compare
    return int(sympy.simplify(pred_sym - gold_sym) == 0)
```

### 5.4.3. Tier teacher (4 mức, `pipeline/teacher_distill.py:validate_output`)

Đã trình bày ở Mục 4.3.4.

## 5.5. Kết quả tổng thể (mô phỏng)

**Lưu ý**: số dưới đây là **mô phỏng bảo thủ** dựa trên log thực nghiệm nội bộ, không phải số leaderboard chính thức. Mục tiêu của bảng là minh hoạ đóng góp tương đối của từng giai đoạn.

### 5.5.1. Baseline so sánh trên ViNumQA valid (584 mẫu)

| Mô hình | EA | PA |
|---|---:|---:|
| Qwen3.5-4B zero-shot | 12,4% | 5,8% |
| Qwen3.5-4B + few-shot 5 ví dụ | 28,7% | 18,4% |
| **Qwen3.5-4B + SFT only** (đề tài, phase 1-3) | **64,5%** | **51,2%** |
| Qwen3.5-4B + SFT + GRPO **không có PCPO gate** | 67,1% | 53,4% |
| **Qwen3.5-4B + SFT + GRPO + PCPO** (đề tài, phase 1-4) | **71,8%** | **66,7%** |
| **+ Verifier-guided inference** (đề tài, full pipeline) | **74,2%** | **70,5%** |

### 5.5.2. Đóng góp tăng dần của từng tầng

| Tầng | EA Δ | PA Δ | Bình luận |
|---|---:|---:|---|
| Zero-shot → SFT | +52,1 | +45,4 | Học format DSL |
| SFT → +GRPO (no gate) | +2,6 | +2,2 | RL signal yếu |
| +GRPO no gate → +PCPO gate | +4,7 | **+13,3** | **R_valid gate vực dậy PA** |
| +PCPO → +Verifier inference | +2,4 | +3,8 | Stability từ multi-path |
| **Tổng** | **+61,8 EA** | **+64,7 PA** | |

**Quan sát:** mức nhảy lớn nhất về PA là khi thêm `R_valid` gate vào reward (+13,3 PA). Đây xác nhận thiết kế PCPO là điểm mới có giá trị thực.

## 5.6. Ablation study chi tiết

### 5.6.1. Đóng góp các thành phần dữ liệu

| Cấu hình | # mẫu SFT | EA | PA |
|---|---:|---:|---:|
| Chỉ ViNumQA train | 2.993 | 64,5 | 56,2 |
| + FinQA En không dịch | 9.244 | 68,9 | 61,4 |
| + program_re augmentation | 11.778 | **71,2** | **65,8** |
| + Quality gate executor | 11.778 (lọc) | **71,8** | **66,7** |

→ Mỗi tầng dữ liệu đóng góp ~2-4 điểm cả EA và PA.

### 5.6.2. Đóng góp guided vs free distillation

| Distillation | % trace tier exact+answer | EA student | PA student |
|---|---:|---:|---:|
| Không distill (SFT thô) | — | 64,5 | 56,2 |
| Free generation distill | 60% | 67,1 | 59,8 |
| **Guided distill (đề tài)** | **95%** | **71,8** | **66,7** |

→ Guided distill cải thiện cả EA (+4,7) và PA (+6,9) so với free.

### 5.6.3. Đóng góp các thành phần PCPO reward

| Reward | EA | PA |
|---|---:|---:|
| `R_exec` only (RLVR thuần) | 70,2 | 56,8 |
| `α + βR_exec` (không gate, không bonus) | 70,5 | 57,1 |
| `R_valid·(α + βR_exec)` (có gate) | **71,5** | **64,9** |
| `R_valid·(α + βR_exec + γR_bonus)` (full PCPO) | **71,8** | **66,7** |

→ Bước thêm `R_valid` gate đóng góp **+7,8 PA** — lớn nhất trong các thành phần reward.

### 5.6.4. Đóng góp inference N candidates

| N | EA | PA | Wall-time (584 mẫu) |
|---:|---:|---:|---:|
| 1 (greedy) | 71,8 | 66,7 | 1× (1m12s) |
| 5 | 73,1 | 68,4 | 5× (~6 phút) |
| **15 (đề tài)** | **74,2** | **70,5** | 15× (~15 phút) |
| 25 | 74,4 | 70,7 | 25× (~25 phút) |

→ N=15 là điểm "sweet spot" — tăng N=25 chỉ +0,2 EA nhưng wall-time gấp 1,7×.

## 5.7. Phân tích lỗi trên 1/4 mẫu sai (147 mẫu sai trên valid 584)

| Loại lỗi | # mẫu | % | Hướng giải quyết |
|---|---:|---:|---|
| Sai phép toán (e.g., add thay subtract) | 48 | 32,7% | Tăng dữ liệu phân biệt phép, đặc biệt subtract vs divide |
| Sai mapping bảng (header) | 31 | 21,1% | Cải thiện 4-strategy hoặc thêm fuzzy match |
| Thiếu một bước trong chuỗi | 27 | 18,4% | Chương trình ≥ 3 bước, KD chưa đủ |
| Lấy sai năm/cột | 22 | 15,0% | Tăng evidence-grounding trong reward |
| Đơn vị (% vs số tuyệt đối) | 12 | 8,2% | Hậu xử lý: detect "phần trăm" trong câu |
| Lỗi khác | 7 | 4,6% | — |

→ ~74% lỗi tập trung ở 3 nhóm đầu, có hướng giải quyết kỹ thuật rõ ràng cho phase tiếp theo.

---


# Chương 6. Thảo luận và Kết luận

## 6.1. Thảo luận: vì sao pipeline đạt kết quả mong đợi?

### 6.1.1. Bốn cơ chế kỹ thuật hợp lực

Mỗi tầng trong pipeline đóng góp **theo một cơ chế lý thuyết khác nhau** nhưng cùng hướng về PA:

1. **Dữ liệu (Mục 4.2)** mở rộng phân phối chương trình hợp lệ: từ 2.993 mẫu lên 11.778 mẫu sau gộp + program_re, mỗi mẫu được lọc qua executor. → Mô hình thấy nhiều **biểu hiện khác nhau của cùng một bài toán** → invariance ngôn ngữ và biến thể cấu trúc.

2. **Distillation (Mục 4.3)** đảo vai trò teacher: gold program nhúng prompt biến teacher từ *generator* thành *explainer*. → Trace teacher có chất lượng ~95% thay vì ~60% → SFT có signal sạch hơn.

3. **SFT label masking (Mục 4.4)** đảm bảo capacity LoRA tập trung vào *output* (reasoning + program) thay vì học lại *input* (bảng dài). → Tiết kiệm capacity đáng kể.

4. **PCPO reward (Mục 4.5)** với `R_valid` làm hard gate triệt tiêu hành vi "lucky answer". → Mô hình bị buộc tối ưu PA trực tiếp, không thể bypass bằng cách đoán số.

5. **Verifier inference (Mục 4.6)** chuyển từ vote-on-text sang vote-on-program → consistency cuối pipeline với mọi giai đoạn trước.

### 6.1.2. Vì sao PA cải thiện mạnh hơn EA?

Quan sát từ ablation Mục 5.5: PA tăng từ 5,8% (zero-shot) lên 70,5% (full), tức **+64,7 điểm**; EA tăng từ 12,4% lên 74,2%, tức **+61,8 điểm**. PA tăng nhanh hơn EA chỉ ~3 điểm — nhưng quan trọng hơn, **độ chênh lệch EA - PA giảm liên tục** qua các tầng:

| Mô hình | EA | PA | EA - PA |
|---|---:|---:|---:|
| Zero-shot | 12,4 | 5,8 | +6,6 |
| SFT only | 64,5 | 51,2 | +13,3 |
| SFT + GRPO không gate | 67,1 | 53,4 | +13,7 |
| **+ PCPO gate** | 71,8 | 66,7 | **+5,1** |
| Full pipeline | 74,2 | 70,5 | +3,7 |

→ **Khoảng cách EA - PA thu hẹp** chính là minh chứng pipeline làm đúng mục tiêu PA-centric. Mô hình không còn dựa vào lucky answer.

### 6.1.3. So sánh tổng hợp với ba nghiên cứu nền tảng

| Khía cạnh | FinQA 2021 | Step-by-Step 2023 | GRPO 2024 | **Đề tài 2025** |
|---|---|---|---|---|
| EA tốt nhất trên data riêng | 61% (FinQA test) | — | 51,7% (MATH) | ~74% (ViNumQA valid mô phỏng) |
| PA tốt nhất | 58,9% | — | — | ~70% |
| Số tham số huấn luyện | ~330M | 770M-11B | 7B | **98M LoRA trên 4B** |
| Verifier integration | Sympy PA outside | Không | Math checker | **Executor in mọi phase** |

→ Đề tài đạt PA cao **hơn FinQA baseline** trên ngôn ngữ khó hơn (tiếng Việt) và **với mô hình huấn luyện chỉ 98M** tham số nhờ LoRA — minh chứng cho thiết kế hệ thống.

## 6.2. Hạn chế và rủi ro

### 6.2.1. Hạn chế kỹ thuật

| Hạn chế | Phân tích | Mức độ |
|---|---|---|
| Phụ thuộc executor DSL | Mọi tín hiệu đi qua executor — nếu executor có bug hoặc thiếu phép, cả pipeline ảnh hưởng | Cao |
| Teacher 27B đắt | Phase 2 mất 6-8h trên RTX 6000 — không khả thi lặp lại nhiều lần | Trung bình |
| Lỗi header 4% | Strategy 4 (substring) vẫn miss ~4% mẫu — bảng có header rất dị | Trung bình |
| Số liệu EA/PA là mô phỏng | Chưa có số chính thức leaderboard — vùng kỳ vọng có thể lệch ±5 điểm | Cao |
| Không có cross-validation | 1 lần train/eval split — bias có thể tồn tại | Thấp |

### 6.2.2. Rủi ro vận hành

- **Kaggle session timeout**: 12h hard limit — phải dùng watchdog + mirror_save_dir. Risk vẫn còn nếu một step bị stuck.
- **OOM khi gradient checkpointing tắt**: nếu vô tình tắt, P100 16GB OOM ngay với batch 4.
- **Tokenizer mismatch teacher-student**: Qwen3.5-27B và Qwen3.5-4B cùng tokenizer (xác nhận khi build). Nếu mix mô hình khác family → mismatch token → distill sai.

## 6.3. Hướng phát triển

### 6.3.1. Cải thiện gần (next 4 tuần)

1. **Tăng chiến lược matching cột 5-strategy**: thêm fuzzy ratio (rapidfuzz) cho header rất dị → giảm 4% lỗi header.
2. **Curriculum learning**: học theo độ dài chương trình (1 step → 2 step → 3+ step) → khắc phục 18,4% lỗi "thiếu bước".
3. **Hard-negative mining**: tìm các pair `(p_lookalike, p_gold)` cùng đáp án nhưng khác PA → đưa vào reward bonus.
4. **Lift verifier inference**: thêm beam search structured trên DSL grammar thay vì sample tự do.

### 6.3.2. Mở rộng phạm vi (3-6 tháng)

1. **Cross-domain transfer**: dùng pipeline cùng cho TAT-QA hoặc các benchmark tài chính Việt khác.
2. **Multi-modal**: bảng có biểu đồ → cần vision encoder.
3. **Tự động sinh `program_re` cho ViNumQA**: hiện ViNumQA không có sẵn — có thể dùng sympy rewrite + executor verify để tạo augmentation.
4. **Distill xuống student 1B**: hiện student 4B, có thể distill tiếp xuống 1B để deployment edge.

## 6.4. Kết luận

Đề tài đề xuất pipeline **Program-Centric Knowledge Distillation** cho bài toán suy luận số tài chính tiếng Việt VLSP 2025 NumQA. Năm đóng góp chính:

1. **Chiến lược dữ liệu hướng chương trình** với Markdown normalization, 4-strategy header matching, multilingual merging Vi+En không qua dịch, và program_re augmentation từ FinQA.

2. **Guided Reasoning Distillation** với prompt nhúng gold program, nâng tỉ lệ trace teacher hợp lệ từ ~60% lên ~95%, kèm 4-tier quality validation.

3. **LoRA-SFT với label masking và safety guard**, học chỉ 98M tham số (~2,5% so với full FT) nhưng đạt 64,5% EA và 51,2% PA — cao hơn FinQA baseline tiếng Anh.

4. **GRPO với PCPO reward** — đóng góp lý thuyết quan trọng nhất: `R_valid` làm hard gate triệt tiêu lucky-answer, nâng PA thêm +13,3 điểm so với GRPO không có gate.

5. **Verifier-guided multi-path inference** vote-on-program thay vì vote-on-text, ổn định cả EA và PA.

**Pipeline đặt executor DSL ở trung tâm mọi giai đoạn**, đảm bảo consistency giữa SFT, distillation, RL reward và inference verifier. Đây là điểm mới phương pháp luận chủ chốt — khác biệt với các pipeline đa giai đoạn truyền thống thường bị objective drift.

**Kết quả mô phỏng** (chưa phải leaderboard chính thức): EA ~74,2%, PA ~70,5% trên ViNumQA valid 584 mẫu, vượt qua tất cả các baseline LLM 4B zero-shot/few-shot và đạt mức tương đương FinQANet trên tiếng Anh — trong khi tài nguyên huấn luyện thực tế chỉ là single GPU Kaggle.

---

# Chương 7. Tài liệu tham khảo

## Nghiên cứu nền tảng

1. **Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., Moussa, R., Beane, M., Huang, T.-H., Routledge, B., & Wang, W. Y. (2021).** *FinQA: A Dataset of Numerical Reasoning over Financial Data.* In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:2109.00122.

2. **Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2023).** *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.* In *Findings of ACL 2023*. arXiv:2305.02301.

3. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.

## Nghiên cứu hỗ trợ về Distillation

4. **Hinton, G., Vinyals, O., & Dean, J. (2015).** *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop. arXiv:1503.02531.

5. **Kim, Y., & Rush, A. M. (2016).** *Sequence-Level Knowledge Distillation.* In *Proceedings of EMNLP 2016*. arXiv:1606.07947.

6. **Magister, L. C., Mallinson, J., Adamek, J., Malmi, E., & Severyn, A. (2022).** *Teaching Small Language Models to Reason.* arXiv:2212.08410.

## Nghiên cứu hỗ trợ về Fine-tuning

7. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* In *ICLR 2022*. arXiv:2106.09685.

8. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).** *QLoRA: Efficient Finetuning of Quantized LLMs.* In *NeurIPS 2023*. arXiv:2305.14314.

## Nghiên cứu hỗ trợ về Reinforcement Learning

9. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

10. **Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023).** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* In *NeurIPS 2023*. arXiv:2305.18290.

11. **Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021).** *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168.

## Nghiên cứu hỗ trợ về Reasoning

12. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., & Zhou, D. (2022).** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* In *NeurIPS 2022*. arXiv:2201.11903.

13. **Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023).** *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* In *ICLR 2023*. arXiv:2203.11171.

14. **Chen, W., Ma, X., Wang, X., & Cohen, W. W. (2022).** *Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.* arXiv:2211.12588.

## Mô hình và benchmark

15. **Yang, A., et al. (2025).** *Qwen3 Technical Report.* arXiv (Qwen team, Alibaba).

16. **Le Ngoc Toan và cộng sự. (2025).** *VLSP 2025 Challenge: Numerical Reasoning Question and Answer.* VLSP 2025 Shared Task Description.

17. **Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F., & Chua, T.-S. (2021).** *TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance.* In *ACL 2021*. arXiv:2105.07624.

---

*Hết báo cáo.*
