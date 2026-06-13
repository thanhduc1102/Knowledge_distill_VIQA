# Báo Cáo Tiến Độ Nghiên Cứu

Người thực hiện: Nghiên cứu sinh / học viên thực hiện đề tài  
Ngày cập nhật: 2026-06-13  
Mục đích: Báo cáo tiến độ hiện tại, tổng hợp những gì đã thực hiện, những gì đã rút ra từ khảo sát và phân tích code, đồng thời chốt hướng triển khai tiếp theo cho bài toán truy xuất và suy luận tài chính toàn trình.

---

## 1. Mục tiêu chung của đề tài

Đề tài hướng tới xây dựng một hệ thống hỏi-đáp tài chính toàn trình có khả năng:

1. truy xuất chính xác tài liệu / bảng / đoạn văn liên quan từ kho tài liệu tài chính dài;
2. xác định đúng bằng chứng cần dùng cho câu hỏi;
3. thực hiện suy luận số học nhiều bước trên dữ liệu bảng và văn bản;
4. kiểm chứng lại kết quả suy luận trước khi sinh câu trả lời cuối cùng.

Khác với hướng RAG truyền thống chỉ tập trung vào “truy xuất đúng ngữ cảnh”, đề tài này hướng đến bài toán mạnh hơn:

> truy xuất đúng bằng chứng và suy luận đúng trong điều kiện context dài, đa bảng, đa năm và có nhiễu retrieval.

---

## 2. Định nghĩa bài toán nghiên cứu

### 2.1 Đầu vào

Đầu vào của hệ gồm:

- câu hỏi tài chính;
- tập tài liệu tài chính dài;
- các bảng số liệu, văn bản mô tả, footnote và metadata đi kèm.

### 2.2 Đầu ra

Đầu ra mong muốn của hệ không chỉ là một đáp án cuối cùng, mà nên bao gồm:

- đáp án số hoặc đáp án dạng văn bản ngắn;
- bằng chứng nguồn đã sử dụng;
- các toán hạng trung gian;
- chương trình / biểu thức tính toán;
- kết quả kiểm chứng hậu nghiệm.

### 2.3 Khó khăn cốt lõi

Bài toán khó ở các điểm sau:

- tài liệu dài, nhiều phần không liên quan;
- nhiều tài liệu gần giống nhau về mặt ngôn ngữ;
- câu hỏi thường gắn chặt với công ty, kỳ báo cáo, loại statement và đơn vị;
- câu trả lời đòi hỏi kết hợp cả bảng và văn bản;
- dễ xảy ra nhầm lẫn giữa số liệu đúng chủ đề nhưng sai năm, sai công ty hoặc sai đơn vị;
- reasoning không thể chỉ dựa trên tương đồng ngữ nghĩa, mà phải tôn trọng logic kế toán và quan hệ số học.

---

## 3. Benchmark và dữ liệu nghiên cứu

### 3.1 Benchmark chính hiện tại

Hiện tại, benchmark phù hợp nhất và đã được gắn với repo là `T2-RAGBench`.

Lý do chọn:

- đây là benchmark retrieval-first;
- phản ánh đúng thực tế rằng context không có sẵn;
- buộc hệ phải giải bài toán truy xuất trước khi suy luận;
- hỗ trợ tốt cho việc đánh giá trên FinQA, ConvFinQA và TAT-DQA.

### 3.2 Các nguồn dữ liệu chính

Các dataset trọng tâm đã và đang được xem xét:

- `FinQA`: bộ dữ liệu suy luận số học tài chính kinh điển, có gold program.
- `ConvFinQA`: suy luận nhiều bước theo hội thoại, khó hơn về dependency giữa câu hỏi.
- `TAT-QA`: kết hợp bảng và văn bản, phù hợp để đánh giá hybrid evidence.
- `DocFinQA`: tăng độ dài ngữ cảnh, phù hợp với retrieval thực tế.
- `T2-RAGBench`: benchmark truy xuất + suy luận trong setting text-bảng.

### 3.3 Định nghĩa benchmark nên dùng cho giai đoạn tiếp theo

Benchmark không nên chỉ dừng ở `MRR@3`.

Cần đánh giá theo 4 tầng:

1. `Retrieval`
   - MRR@3
   - Recall@1/3/5
   - NDCG@3

2. `Evidence grounding`
   - độ chính xác xác định đúng table/section
   - độ chính xác chọn đúng cell/span
   - độ chính xác company/year
   - độ chính xác unit/scale

3. `Reasoning`
   - answer accuracy
   - execution accuracy
   - program validity
   - grounded reasoning accuracy

4. `Robustness`
   - reasoning khi top-3 có nhiễu
   - nhầm year
   - nhầm company
   - nhầm scale
   - multi-table reasoning

Như vậy, benchmark sau này phải đánh giá toàn trình chứ không chỉ retrieval.

---

## 4. Hiện trạng triển khai trong repo

### 4.1 Baseline hiện có

Trong `NLP/baseline/source_simplification/` đã có các baseline retrieval như:

- dense retrieval với FAISS;
- hybrid BM25 + dense retrieval;
- HyDE;
- summarization-based query rewriting.

Các baseline này đóng vai trò làm mốc so sánh cho hướng đề xuất.

### 4.2 Prototype đề xuất hiện có

Trong `NLP/ours/source/` đã có một prototype tương đối đầy đủ cho retrieval có cấu trúc:

- `GSR`: Graph-Structured Retrieval
- `CACL`: Constraint-Aware Contrastive Learning
- builder tạo KG từ bảng
- thư viện template kế toán
- edge-aware GAT encoder
- joint scorer kết hợp text/entity/constraint
- CHAP hard-negative sampler
- benchmark script
- training script

Điều này cho thấy đề tài không xuất phát từ số 0, mà đã có một nền kỹ thuật khá rõ.

---

## 5. Những gì đã phân tích và rút ra từ code hiện tại

### 5.1 Về GSR

GSR hiện đã được cài đặt ở mức retrieval thật sự.

Luồng hoạt động:

1. load corpus và query từ T2-RAGBench;
2. build FAISS index trên text;
3. trích bảng markdown đầu tiên trong mỗi document;
4. match bảng với template tài chính;
5. sinh KG;
6. encode KG bằng GAT;
7. dùng joint scorer để rerank candidate từ FAISS.

### 5.2 Về CACL

CACL hiện tồn tại ở mức training strategy:

- Stage 1: học tín hiệu identity / metadata;
- Stage 2: học tín hiệu cấu trúc;
- Stage 3: joint finetuning với CHAP negatives.

Tuy nhiên, CACL hiện chưa được nối kín hoàn toàn vào benchmark inference hiện tại, nên hệ chủ yếu vẫn phản ánh GSR inference mạnh hơn là full GSR+CACL end-to-end.

### 5.3 Về KG hiện tại

KG hiện tại giúp:

- chuẩn hóa tên line item tài chính;
- mã hóa cấu trúc bảng;
- sinh thêm tín hiệu constraint và graph embedding.

Tuy nhiên, KG này vẫn mang bản chất:

- graph cho retrieval có cấu trúc,
- chưa phải graph cho reasoning số học đầy đủ.

### 5.4 Về metadata

Metadata hiện tại gồm:

- company
- year
- sector

Đây là một khởi đầu tốt, nhưng đang bị dùng quá nông. Nó chưa trở thành:

- ontology nhẹ cho retrieval,
- ràng buộc chọn toán hạng,
- hay tín hiệu logic trong reasoning.

---

## 6. Đóng góp thực tế đã có tính đến hiện tại

### 6.1 Đóng góp mức retrieval

Đề tài hiện đã có các đóng góp rõ ở mức retrieval:

- thêm cấu trúc tài chính vào truy xuất;
- dùng template-based KG để khai thác cấu trúc bảng;
- dùng graph encoder để sinh thêm tín hiệu reranking;
- tạo hard negative có chủ đích bằng CHAP;
- benchmark được trên dữ liệu retrieval-first.

### 6.2 Đóng góp mức tư duy nghiên cứu

Qua quá trình phân tích, đề tài đã làm rõ thêm một định hướng quan trọng:

> không nên coi KG chỉ là công cụ tăng MRR, mà nên coi nó là bộ khung trung tâm để nối retrieval với reasoning.

Đây là một bước phát triển quan trọng về mặt tư duy bài toán.

---

## 7. Những khoảng trống và vấn đề còn tồn tại

### 7.1 Pairwise constraint chưa đủ cho reasoning

Các phương trình nhiều toán hạng hiện đang bị rút gọn thành các cạnh cặp đôi.

Điều này:

- đủ để thêm cấu trúc cho retrieval,
- nhưng chưa đủ để thực thi suy luận số học đúng nghĩa.

### 7.2 Constraint score chưa phải bộ kiểm định toán học đầy đủ

Constraint score hiện mang tính heuristic có cấu trúc, chứ chưa phải phép kiểm chứng phương trình đúng/sai một cách trung thành.

### 7.3 Retrieval vẫn còn coarse

Hiện chủ yếu mới ở document-level reranking.

Nhưng bài toán reasoning yêu cầu:

- table-level retrieval,
- section-level retrieval,
- evidence-atom retrieval.

### 7.4 Chưa có module reasoning có thể kiểm chứng

Hiện chưa có:

- DSL / Python program generator chính thức,
- executor,
- verifier,
- local evidence subgraph cho reasoning.

### 7.5 Top-k retrieval noise chưa được xử lý triệt để

Nếu reasoning nhận 3 context mà chỉ 1 context đúng, mô hình rất dễ:

- trộn toán hạng,
- dùng sai năm,
- dùng sai row,
- hoặc dùng một bằng chứng phụ nhưng sai vai trò.

Đây là nút thắt lớn nhất của bài toán toàn trình.

---

## 8. Chốt ý tưởng nghiên cứu nên theo

Sau quá trình phân tích code, benchmark và tài liệu liên quan, hướng nên chốt là:

> xây dựng một Financial Evidence Graph có kiểu, dùng chung cho retrieval và reasoning, trong đó metadata được nâng thành ontology truy xuất, còn reasoning được thực hiện qua executor và được verifier kiểm chứng.

Nói cách khác, contribution chính không nên chỉ là:

- KG tốt hơn cho retrieval

mà nên là:

- evidence graph thống nhất để giải quyết cả retrieval và reasoning.

---

## 9. Phương pháp triển khai nên chọn

### 9.1 Đối với retrieval

Nên phát triển retrieval thành dạng phân cấp:

1. document retrieval
2. table / section retrieval
3. evidence atom retrieval

Metadata phải tham gia vào:

- pre-filtering
- contextual chunk embedding
- hard-negative mining
- reranking

### 9.2 Đối với KG

Nên thay KG hiện tại bằng evidence graph có:

- provenance đầy đủ,
- unit / scale,
- entity / time,
- equation node / operator node,
- liên kết text-table-footnote.

### 9.3 Đối với reasoning

Không nên reasoning trực tiếp trên top-k context.

Cần pipeline:

1. evidence triage
2. local subgraph construction
3. operand grounding
4. DSL / Python generation
5. execution
6. verification

### 9.4 Đối với huấn luyện

Chiến lược nên dùng:

1. retrieval training bằng contrastive / ranking
2. reasoning SFT
3. Step-DPO hoặc process-aware DPO
4. GRPO / RLVR với reward verifiable

Không nên dùng ORPO làm hướng chính, chỉ nên dùng làm baseline compute thấp.

---

## 10. Cơ sở lựa chọn DPO, ORPO, GRPO

### 10.1 DPO

Phù hợp để:

- học từ cặp trace đúng / sai,
- dạy mô hình tránh dùng context nhiễu,
- tăng tính kỷ luật cho reasoning trace.

Nhưng:

- DPO thuần chưa đủ mạnh nếu reward số học có thể tính chính xác bằng executor.

### 10.2 ORPO

Phù hợp để:

- làm baseline gọn nhẹ.

Nhưng:

- không phải lựa chọn tối ưu cho bài toán numerical reasoning có kiểm chứng.

### 10.3 GRPO

Phù hợp nhất cho giai đoạn cuối vì:

- bài toán có reward tính được tự động;
- có thể chấm:
  - answer correctness,
  - program correctness,
  - grounding correctness,
  - metadata consistency,
  - constraint consistency.

Kết luận:

- `SFT -> Step-DPO -> GRPO`
là pipeline hợp lý nhất.

---

## 11. Kế hoạch triển khai dự kiến

### Giai đoạn 1: Củng cố retrieval

Mục tiêu:

- retrieval phân cấp, sạch và có metadata-aware scoring.

Việc cần làm:

- mở rộng metadata schema;
- thêm metadata-aware chunk embedding;
- hard-negative mining mạnh hơn;
- thêm table/section reranking.

### Giai đoạn 2: Xây evidence graph

Mục tiêu:

- thay pairwise KG bằng graph trung thành với reasoning.

Việc cần làm:

- thiết kế schema node/edge;
- thêm provenance;
- thêm equation node;
- liên kết text-table-footnote.

### Giai đoạn 3: Xây module reasoning

Mục tiêu:

- reasoning có khả năng thực thi và kiểm chứng.

Việc cần làm:

- parser câu hỏi;
- operand grounding;
- DSL / Python program;
- executor;
- verifier.

### Giai đoạn 4: Huấn luyện và đánh giá

Mục tiêu:

- tạo pipeline reasoning mạnh và ổn định.

Việc cần làm:

- SFT trên trace / program;
- Step-DPO;
- GRPO / RLVR;
- benchmark end-to-end;
- robustness evaluation.

---

## 12. Các ý tưởng phát triển mới cần ưu tiên

### Ý tưởng 1: Metadata như ontology truy xuất

Thay vì coi metadata chỉ là bonus, cần dùng nó như:

- điều kiện định tuyến candidate,
- điều kiện sinh hard-negative,
- ràng buộc kiểm chứng toán hạng.

### Ý tưởng 2: Local evidence graph cho top-3 noisy contexts

Khi retrieval chỉ đạt top-3, thay vì đưa nguyên 3 context sang LLM, cần:

- bẻ nhỏ thành atom,
- lọc atom,
- dựng graph con,
- rồi mới reasoning.

### Ý tưởng 3: Reasoning có verifier

Điểm mới không chỉ nằm ở sinh program, mà nằm ở khả năng:

- kiểm chứng ngược,
- phát hiện sai company/year,
- phát hiện sai unit/scale,
- phát hiện sai phép toán.

---

## 13. Những gì đã làm được đến thời điểm hiện tại

Tổng hợp ngắn gọn:

### Đã làm được

- khảo sát benchmark và repo hiện tại;
- phân tích code baseline và GSR-CACL;
- xác định GSR và CACL đã được triển khai tới đâu;
- phân tích sâu vai trò của KG, edge-aware GAT, constraint scoring;
- xác định các vấn đề thiết kế hiện tại;
- khảo sát thêm các hướng như HierFinRAG, FT-RAG, APOLLO, FinanceReasoning, metadata-driven retrieval và preference/RL optimization;
- chốt được hướng chiến lược toàn trình.

### Chưa hoàn thành

- retrieval phân cấp hoàn chỉnh;
- evidence graph mới;
- reasoning executor;
- verifier;
- pipeline SFT -> DPO -> GRPO;
- benchmark end-to-end.

---

## 14. Kết luận và đề xuất với GVHD

Tại thời điểm hiện tại, đề tài đã đi qua giai đoạn:

- từ ý tưởng rời rạc,
- sang prototype retrieval có cấu trúc,
- và nay đã chốt được hướng phát triển thành bài toán toàn trình.

Nhận định quan trọng nhất là:

> nếu chỉ tiếp tục mở rộng KG cho retrieval thì đóng góp có thể chưa đủ mạnh; nếu phát triển KG thành evidence graph dùng chung cho retrieval và reasoning, kết hợp executor và verifier, thì hướng nghiên cứu trở nên vững và có giá trị học thuật cao hơn rõ rệt.

Vì vậy, hướng triển khai đề xuất tiếp theo là:

1. củng cố retrieval phân cấp có metadata-aware scoring;
2. thiết kế lại graph theo hướng equation-centric evidence graph;
3. xây module reasoning có thể thực thi;
4. thêm verifier;
5. huấn luyện theo pipeline SFT -> Step-DPO -> GRPO.

Đây là hướng được đánh giá là hợp lý, có chiều sâu và phù hợp nhất để phát triển thành kết quả nghiên cứu mạnh trong giai đoạn tiếp theo.
