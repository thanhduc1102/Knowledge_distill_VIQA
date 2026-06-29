# Báo Cáo Tiến Độ Nghiên Cứu

Ngày cập nhật: 2026-06-13  
Mục đích: Tổng hợp hiện trạng nghiên cứu, những nội dung đã triển khai, các đóng góp hiện có, các giới hạn còn tồn tại và kế hoạch phát triển tiếp theo cho bài toán truy xuất và suy luận tài chính toàn trình.

---

## 1. Mục tiêu nghiên cứu và phạm vi bài toán

Mục tiêu chung của đề tài là xây dựng một hệ thống hỏi-đáp tài chính toàn trình có khả năng xử lý tài liệu dài, nhiều bảng, nhiều đoạn văn và nhiều footnote, trong đó hệ thống không chỉ truy xuất đúng ngữ cảnh mà còn xác định đúng bằng chứng, định vị đúng toán hạng và thực hiện suy luận số học đáng tin cậy. Khác với các hệ RAG thông thường chủ yếu tập trung vào việc đưa về một số đoạn văn có vẻ liên quan, bài toán trong miền tài chính đòi hỏi nhiều hơn: thông tin đúng thường bị phân tán ở nhiều vị trí, nhiều bảng có bề mặt ngôn ngữ gần giống nhau, và câu trả lời cuối cùng thường phải được tạo ra bằng một chuỗi thao tác số học có thể kiểm chứng.

Do đó, bài toán nghiên cứu của đề tài được xác định lại theo hướng mạnh hơn:

> truy xuất đúng bằng chứng tài chính, chọn đúng toán hạng, thực hiện đúng chương trình suy luận số học và kiểm chứng được kết quả cuối cùng trong điều kiện retrieval có nhiễu.

Theo cách nhìn này, hệ thống tương lai phải có bốn năng lực chính. Thứ nhất là truy xuất phân cấp, từ cấp tài liệu xuống cấp bảng, mục và bằng chứng con. Thứ hai là grounding, tức là định vị được đúng hàng, đúng cột, đúng ô hoặc đúng câu văn cần dùng. Thứ ba là reasoning, tức là sinh và thực thi được chuỗi phép toán cần thiết. Thứ tư là verification, tức là có cơ chế kiểm tra ngược xem kết quả sinh ra có thực sự phù hợp với công ty, năm, đơn vị, scale và ràng buộc tài chính hay không.

---

## 2. Benchmark, dữ liệu và cách đặt bài toán thực nghiệm

### 2.1 Benchmark chính hiện tại

Benchmark trung tâm đang được sử dụng là `T2-RAGBench`. Việc lựa chọn benchmark này là hợp lý vì nó phản ánh đúng setting thực tế hơn các bài toán QA tài chính cổ điển: hệ thống không được cấp sẵn context đúng, mà phải truy xuất từ tập tài liệu text-bảng trước khi reasoning. Đây là khác biệt rất quan trọng, bởi nhiều sai số trong hệ tài chính không đến từ năng lực tính toán thuần túy mà đến từ việc lấy nhầm tài liệu, nhầm bảng hoặc nhầm bằng chứng ngay từ đầu.

### 2.2 Các bộ dữ liệu liên quan

Các bộ dữ liệu đang và sẽ được dùng trong đề tài gồm:

- `FinQA`: bộ dữ liệu kinh điển cho suy luận số học tài chính, có gold program, phù hợp để huấn luyện và đánh giá reasoning có thực thi.
- `ConvFinQA`: mở rộng bài toán sang setting hội thoại, nơi reasoning chain dài hơn và phụ thuộc nhiều hơn vào diễn biến câu hỏi.
- `TAT-QA`: phù hợp để đánh giá truy xuất và reasoning kết hợp giữa bảng và văn bản.
- `DocFinQA`: hữu ích cho việc đánh giá retrieval và reasoning trong bối cảnh ngữ cảnh dài hơn.
- `T2-RAGBench`: dùng để đo retrieval-first và làm nền cho đánh giá end-to-end sau này.

### 2.3 Cách đánh giá cần được mở rộng

Một kết luận quan trọng rút ra trong quá trình phân tích là không thể dùng duy nhất `MRR@3` hoặc `Recall@k` để đại diện cho năng lực của hệ toàn trình. Lý do là trong bài toán tài chính, hệ có thể đạt MRR@3 tương đối tốt nhưng vẫn thất bại ở reasoning, ví dụ khi top-3 context gồm 1 tài liệu đúng và 2 tài liệu nhiễu có cùng công ty nhưng khác năm, hoặc đúng năm nhưng sai loại statement. Trong tình huống đó, retrieval có vẻ đúng ở cấp document, nhưng hệ vẫn chọn sai toán hạng và cho ra đáp án sai.

Vì vậy, benchmark cần được tổ chức lại thành nhiều tầng:

1. `Retrieval`: MRR@3, Recall@1/3/5, NDCG@3.
2. `Evidence grounding`: độ chính xác chọn đúng bảng, đúng section, đúng cell/span.
3. `Reasoning`: answer accuracy, program validity, execution accuracy.
4. `Robustness`: đánh giá trong điều kiện top-k có nhiễu, sai year, sai scale, sai company hoặc multi-table reasoning.

Đây là một điểm then chốt vì nó làm rõ rằng đích cuối của đề tài không phải là “cải thiện điểm retrieval”, mà là tối ưu hóa toàn trình từ retrieval đến reasoning.

---

## 3. Hiện trạng triển khai trong repo và những gì đã làm được

### 3.1 Bức tranh tổng thể hiện tại

Qua khảo sát mã nguồn và tài liệu hiện có trong thư mục `NLP/`, có thể kết luận rằng đề tài đã đi qua giai đoạn ý tưởng ban đầu và đang ở mức một prototype nghiên cứu thật sự. Repo hiện tại gồm hai khối chính. Khối thứ nhất là các baseline retrieval trong `baseline/source_simplification`, dùng làm mốc so sánh. Khối thứ hai là nhánh đề xuất trong `ours/source`, nơi đã hiện thực hóa một phiên bản retrieval có cấu trúc thông qua GSR-CACL.

Nói cách khác, đề tài hiện không còn ở mức “ý tưởng trên giấy”, mà đã có một hệ thống có thể chạy benchmark retrieval, sinh KG từ bảng, mã hóa đồ thị và đánh giá chất lượng truy xuất. Đây là nền tảng rất quan trọng để phát triển tiếp sang giai đoạn reasoning toàn trình.

### 3.2 Những gì đã triển khai ở mức baseline

Ở nhánh baseline, repo đã có:

- dense retrieval với FAISS;
- hybrid retrieval với BM25 kết hợp dense retrieval;
- một số dạng query rewriting như HyDE và summarization-based retrieval.

Vai trò của khối baseline này là cung cấp đường cơ sở để so sánh, đồng thời giúp nhìn rõ giá trị gia tăng của việc đưa cấu trúc tài chính vào retrieval. Đây là điều cần thiết vì mọi đề xuất mới ở retrieval đều phải được chứng minh là tốt hơn hoặc ít nhất hợp lý hơn các baseline chuẩn.

### 3.3 Những gì đã triển khai ở mức đề xuất

Ở nhánh `ours/source`, các thành phần quan trọng đã có gồm:

- bộ loader và dataset wrapper cho T2-RAGBench;
- hệ `GSR` cho retrieval có cấu trúc;
- bộ sinh `Constraint KG` từ bảng markdown;
- thư viện template tài chính;
- `Edge-aware GAT` để mã hóa đồ thị;
- `Joint Scorer` để kết hợp text score, entity score và constraint signal;
- `CACL` như một khung huấn luyện contrastive có tri thức ràng buộc;
- `CHAP` negative sampler để tạo hard negatives;
- benchmark retrieval;
- training script theo ba giai đoạn.

Việc các thành phần này đã tồn tại và có thể đối chiếu trực tiếp với `contribution1.pdf` là một bằng chứng rõ ràng cho thấy hướng nghiên cứu hiện tại có nền móng triển khai thật sự.

---

## 4. Các nội dung đã làm được cần trình bày rõ: GSR, CACL và Knowledge Graph

### 4.1 GSR đã làm được gì

GSR, tức `Graph-Structured Retrieval`, là đóng góp kỹ thuật cốt lõi đang hiện diện rõ nhất trong repo hiện tại. Tinh thần của GSR là dense retrieval thuần text không đủ cho tài liệu tài chính vì các báo cáo của cùng một công ty qua nhiều năm thường có bề mặt từ vựng rất giống nhau, trong khi các bảng tài chính chứa đựng nhiều cấu trúc số học mà mô hình text-only không nắm được.

Luồng của GSR hiện tại có thể mô tả như sau. Trước hết, hệ dùng dense retrieval để lấy một tập ứng viên tài liệu. Sau đó, với mỗi tài liệu ứng viên, hệ trích bảng markdown đầu tiên, chuẩn hóa tiêu đề bảng bằng template tài chính và xây một `Constraint KG`. Từ đồ thị này, hệ tạo ra hai dạng tín hiệu bổ sung cho retrieval: một `graph embedding` thông qua `Edge-aware GAT`, và một `constraint score` phản ánh mức độ nhất quán cấu trúc của bảng. Hai tín hiệu này cùng với `entity score` và `text score` được gộp lại trong `Joint Scorer` để tái xếp hạng các ứng viên.

Giá trị thực sự của GSR nằm ở chỗ nó biến retrieval từ một bài toán semantic similarity thuần túy thành một bài toán semantic + structure reranking. Trong miền tài chính, đây là một cải tiến có ý nghĩa vì nhiều tài liệu sai vẫn rất giống với tài liệu đúng về mặt từ vựng. Bằng việc đưa thêm tín hiệu cấu trúc, GSR giúp hệ phân biệt tốt hơn các ứng viên gần đúng nhưng khác nhau về logic kế toán hoặc khác nhau về company/year.

Điểm cần nhấn mạnh trong báo cáo là GSR **đã được triển khai thật** và **đã có benchmark retrieval thật**, chứ không còn chỉ là proposal. Tuy nhiên, GSR hiện vẫn mới giải quyết tốt nhất ở mức document retrieval có cấu trúc, chứ chưa đi xuống đến cấp evidence atom hay reasoning.

### 4.2 CACL đã làm được gì

`CACL` là viết tắt của `Constraint-Aware Contrastive Learning`. Nếu GSR chủ yếu giải quyết phần inference-time retrieval, thì CACL là phần training-time giúp hệ học cách tách biệt tài liệu đúng và tài liệu sai dưới góc nhìn ràng buộc tài chính.

Điểm mạnh của CACL nằm ở triết lý sinh hard-negative. Thay vì chỉ chọn các tài liệu ngẫu nhiên hoặc các tài liệu sai chủ đề, hệ dùng `CHAP` để sinh các negative khó theo ba dạng: làm sai nhẹ một giá trị trong bảng, làm sai scale hoặc hoán đổi metadata như công ty và năm. Đây là một thiết kế phù hợp với miền tài chính vì lỗi thực tế của retriever thường không đến từ việc lấy nhầm một tài liệu hoàn toàn vô nghĩa, mà đến từ việc lấy một tài liệu “rất giống nhưng sai”.

Training hiện tại được chia thành ba giai đoạn:

1. `Identity pretraining`: dạy mô hình phân biệt cặp `(company, year, sector)`.
2. `Structural pretraining`: dạy mô hình học tín hiệu từ đồ thị và ràng buộc.
3. `Joint finetuning`: tối ưu cùng lúc text encoder, graph encoder và scorer với CHAP negatives.

Điểm quan trọng cần trình bày khách quan là CACL **đã tồn tại ở mức code và training pipeline**, nhưng **chưa được nối kín một cách đầy đủ vào benchmark inference hiện tại**. Nói cách khác, CACL là một đóng góp có thật, nhưng hệ hiện tại đang phản ánh rõ GSR retrieval hơn là một hệ GSR+CACL hoàn chỉnh từ train đến inference.

### 4.3 Knowledge Graph hiện tại đã làm được gì

Knowledge graph hiện tại nên được gọi chính xác hơn là `Constraint KG`. Đây là đồ thị được xây từ bảng markdown, trong đó:

- mỗi ô là một node;
- các tiêu đề cột được chuẩn hóa về các khái niệm tài chính canonical;
- các cạnh biểu diễn quan hệ accounting hoặc quan hệ positional.

Vai trò chính của KG hiện tại là:

- chuẩn hóa ý nghĩa của bảng;
- biểu diễn cấu trúc bảng dưới dạng có hướng;
- sinh thêm tín hiệu cho `Edge-aware GAT`;
- sinh thêm `constraint score`.

Nhờ vậy, KG hiện tại đã đóng góp thiết thực cho retrieval vì nó giúp mô hình nhìn bảng như một cấu trúc tài chính thay vì chỉ như một chuỗi markdown. Đây là một đóng góp đáng kể của hệ hiện tại và cần được trình bày rõ như một thành quả kỹ thuật đã có.

Tuy nhiên, cũng cần chỉ ra rõ giới hạn: KG hiện tại mới là `retrieval graph`, chưa phải `reasoning graph`. Nhiều phương trình nhiều toán hạng đang bị phân tách thành các cạnh cặp đôi, điều này tạo được tín hiệu cấu trúc nhưng chưa đủ trung thành cho suy luận số học chính xác.

---

## 5. Các đóng góp hiện có và giá trị của chúng

Tại thời điểm hiện tại, có thể tóm tắt các đóng góp thực chất của đề tài như sau.

Thứ nhất, đề tài đã xây dựng được một prototype retrieval-first bám sát benchmark thực tế, thay vì chỉ làm trên dữ liệu có sẵn context vàng. Đây là một điểm mạnh vì nó phản ánh đúng khó khăn của tài liệu tài chính dài.

Thứ hai, đề tài đã đưa được tri thức tài chính vào retrieval thông qua template matching, constraint graph, graph encoder và joint scorer. Đây là bước vượt lên trên retrieval thuần semantic.

Thứ ba, đề tài đã đề xuất và hiện thực hóa một cơ chế huấn luyện hard-negative có ý nghĩa miền thông qua CACL và CHAP. Đây là một đóng góp đáng giá vì nó giải quyết đúng loại negative mà retrieval tài chính thường gặp.

Thứ tư, đề tài đã làm rõ một hướng phát triển tiếp theo giàu tiềm năng: chuyển từ `KG cho reranking` sang `evidence graph dùng chung cho retrieval và reasoning`. Về mặt nghiên cứu, đây là bước phát triển tư duy quan trọng, bởi nó mở ra một bài toán lớn hơn và có khả năng tạo đóng góp học thuật mạnh hơn nhiều.

---

## 6. Những giới hạn của hệ hiện tại và vì sao cần thay đổi định hướng

### 6.1 Metadata hiện tại chưa phát huy hết tác dụng

Metadata hiện tại gồm `company`, `year`, `sector`. Mặc dù ý tưởng dùng chúng để giúp hệ tránh nhầm thực thể là đúng, nhưng cách khai thác hiện tại còn khá nông. Metadata mới chủ yếu tham gia vào một nhánh scoring đơn giản, chưa trở thành:

- ontology truy xuất;
- điều kiện pre-filter;
- thành phần của chunk embedding;
- điều kiện sinh hard-negative mạnh;
- ràng buộc kiểm chứng toán hạng trong reasoning.

Điều này đồng nghĩa với việc repo hiện đã “động vào đúng điểm”, nhưng chưa khai thác đủ sâu để biến metadata thành lợi thế lớn của miền tài chính.

### 6.2 Constraint và retrieval cần được tối ưu sâu hơn

Constraint score hiện tại có giá trị như một tín hiệu cấu trúc, nhưng chưa phải bộ kiểm định toán học đầy đủ. Đồng thời, retrieval hiện vẫn chủ yếu ở cấp tài liệu, nên khi đi vào reasoning, hệ còn phải đối mặt với bài toán rất khó là top-k context vẫn chứa nhiều nhiễu.

Nếu tiếp tục chỉ tối ưu dense retrieval hoặc joint scoring ở cấp tài liệu, hệ có thể tăng một số chỉ số retrieval nhưng vẫn thất bại ở bài toán chọn toán hạng và thực thi đúng phép toán. Điều này cho thấy retrieval và constraint hiện tại cần được xem như bước đệm, chứ chưa phải lời giải cuối cùng.

### 6.3 KG hiện tại chưa phục vụ reasoning toàn trình

Điểm yếu rõ nhất của KG hiện tại là nó không được xây để làm bằng chứng thống nhất cho reasoning. Nó mạnh ở chỗ cung cấp cấu trúc cho retrieval, nhưng chưa mạnh ở chỗ:

- lưu provenance đầy đủ;
- liên kết text-table-footnote;
- lưu unit/scale rõ ràng;
- biểu diễn phương trình dưới dạng executable;
- hỗ trợ verifier.

Chính vì vậy, nếu muốn đi tới một hệ reasoning toàn trình, việc mở rộng KG thành evidence graph là bước bắt buộc.

---

## 7. Định hướng chốt: xây dựng Financial Evidence Graph dùng chung cho retrieval và reasoning

Đây là kết luận quan trọng nhất của toàn bộ giai đoạn phân tích.

Định hướng nghiên cứu nên được chốt như sau:

> xây dựng một Financial Evidence Graph có kiểu, dùng chung cho retrieval và reasoning, trong đó metadata được nâng thành ontology truy xuất, còn reasoning được thực hiện qua executor và được verifier kiểm chứng.

Nói theo cách khác, thay vì để retrieval và reasoning là hai khối gần như tách rời nhau, hệ thống tương lai sẽ có một lớp trung gian thống nhất là `evidence graph`. Lớp này chịu trách nhiệm:

- liên kết document, section, table, row, column, cell, sentence, footnote;
- gắn company, period, statement type, unit và scale cho từng evidence atom;
- biểu diễn được các quan hệ phương trình hoặc quan hệ dẫn xuất;
- cho phép retriever chọn đúng bằng chứng con;
- cho phép reasoning chọn đúng toán hạng và kiểm chứng lại kết quả.

Đây không chỉ là một thay đổi kỹ thuật, mà là một thay đổi quan trọng về cách đặt bài toán. Bài toán từ đây không còn là “tìm đúng tài liệu”, mà là “xây một cấu trúc bằng chứng thống nhất để giải quyết cả retrieval và reasoning”.

---

## 8. Đề xuất các bước tiếp theo: metadata, constraint, retrieval và graph cho reasoning

### 8.1 Tận dụng metadata như một ontology truy xuất

Metadata cần được nâng cấp từ bộ ba hiện tại sang một schema phong phú hơn, bao gồm:

- company và aliases;
- fiscal year, fiscal quarter, period start/end;
- report type và statement type;
- unit, currency và scale;
- table id, section, row-header path, column-header path.

Sau khi có schema này, metadata sẽ được dùng ở bốn vị trí:

1. `Pre-filtering`: lọc hoặc bias candidate theo company/time khi parser truy vấn đủ chắc chắn.
2. `Contextual chunk embedding`: đưa metadata vào chính biểu diễn của chunk thay vì chỉ cộng thêm một scalar score về sau.
3. `Hard-negative mining`: tạo negative kiểu cùng công ty sai năm, cùng năm sai company, cùng company-year nhưng sai statement, đúng statement nhưng sai row.
4. `Reasoning-time verification`: kiểm tra toán hạng có cùng company/year/unit hay không trước khi executor chạy.

Cách dùng này mới thực sự biến metadata thành một ontology nhẹ cho retrieval và reasoning.

### 8.2 Tối ưu constraint và retrieval

Constraint hiện tại cần được nâng từ mức `edge-wise score` sang mức `equation-faithful verification`. Điều này có nghĩa là các ràng buộc kế toán nhiều toán hạng không nên tiếp tục bị biểu diễn chỉ bằng các cạnh cặp đôi, mà cần được biểu diễn bằng:

- equation node;
- operator node;
- hoặc hyperedge.

Ở phía retrieval, hệ cần tiến hóa từ:

- document-level retrieval

sang:

- document retrieval;
- table/section retrieval;
- evidence-atom retrieval.

Chỉ khi đi xuống đến atom-level, hệ mới có cơ hội thực sự giải bài toán chọn đúng toán hạng dưới điều kiện top-k có nhiễu.

### 8.3 Xây graph để tận dụng cho reasoning

Graph mới không nên chỉ là một KG của bảng. Nó phải là một `Financial Evidence Graph`, trong đó có đủ các node và edge cần thiết cho cả retrieval lẫn reasoning.

Các node chính nên có:

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

Các edge chính nên có:

- `belongs_to`
- `mentions`
- `same_company`
- `same_period`
- `same_metric`
- `has_unit`
- `has_scale`
- `part_of_equation`
- `equation_result`
- `supports`
- `derived_from`
- `referenced_by_footnote`

Khi graph được xây theo schema này, nó sẽ có thể phục vụ cả hai phía:

- phía retrieval: giúp xếp hạng đúng bằng chứng;
- phía reasoning: giúp neo toán hạng, gợi ý phép toán và kiểm chứng lời giải.

---

## 9. Kế hoạch triển khai chi tiết: các phần định làm và cách làm

### 9.1 Giai đoạn 1: củng cố retrieval

Mục tiêu của giai đoạn này là làm cho retrieval mạnh và đáng tin hơn trước khi nối sang reasoning. Công việc cụ thể gồm:

- thống nhất lại preprocessing giữa train và inference;
- mở rộng metadata schema;
- đưa metadata vào biểu diễn chunk;
- xây thêm hard-negative theo company/year/statement/row;
- thêm bước table/section reranking.

Đầu ra của giai đoạn này phải là một hệ retrieval phân cấp có khả năng đưa ra candidate sạch hơn và có thể đánh giá không chỉ ở document-level mà cả table/section-level.

### 9.2 Giai đoạn 2: thiết kế và xây Financial Evidence Graph

Mục tiêu là thay graph cũ bằng graph có thể sống được qua cả retrieval và reasoning. Công việc gồm:

- định nghĩa schema node/edge;
- thêm provenance đầy đủ cho mọi evidence atom;
- biểu diễn unit/scale rõ ràng;
- thêm equation node hoặc operator node;
- liên kết text, table và footnote.

Đầu ra của giai đoạn này là một graph builder mới, tạo được local evidence graph từ top-k context hoặc trực tiếp từ candidate table/section.

### 9.3 Giai đoạn 3: xây reasoning substrate

Mục tiêu là chuyển từ “có bằng chứng” sang “thực thi được reasoning”.

Các module cần xây:

- `Query parser`: tách company, thời gian, metric, loại phép toán.
- `Operand grounding`: chọn các cell/span/row phù hợp từ evidence graph.
- `Program generator`: sinh DSL hoặc Python code.
- `Executor`: chạy chương trình và trả ra kết quả số học.
- `Verifier`: kiểm tra grounding, unit, company/year, constraint consistency.

Điểm rất quan trọng là verifier không chỉ là phần hậu kiểm, mà còn là nền của reward khi chuyển sang RL.

### 9.4 Giai đoạn 4: huấn luyện reasoning

Khi retrieval và graph đã đủ ổn, reasoning sẽ được huấn luyện theo ba lớp:

1. `SFT`: huấn luyện mô hình sinh trace và chương trình grounded từ bằng chứng đúng.
2. `Step-DPO`: tạo cặp chosen/rejected giữa trace đúng và trace sai do context nhiễu, sai year, sai row, sai unit.
3. `GRPO / RLVR`: tối ưu cuối cùng bằng reward có thể kiểm chứng từ executor và verifier.

Reward ở giai đoạn cuối cần bao gồm:

- đúng đáp án số;
- đúng chương trình;
- đúng company/year;
- đúng unit/scale;
- đúng grounding;
- đúng constraint.

### 9.5 Giai đoạn 5: benchmark toàn trình và viết bài

Sau khi pipeline hoàn chỉnh, cần đánh giá:

- retrieval theo nhiều tầng;
- evidence grounding;
- answer accuracy;
- execution accuracy;
- robustness dưới top-3 noisy contexts;
- các case study minh họa.

Đây là giai đoạn để chốt contribution và chuẩn bị bài báo.

---

## 10. Những nội dung cần nhấn mạnh khi báo cáo tiến độ

Khi báo cáo với GVHD, có thể chốt lại mạch tiến độ như sau.

Trước hết, đề tài **đã có nền triển khai thực sự** chứ không phải chỉ có ý tưởng. GSR, CACL và Constraint KG đều đã hiện diện dưới dạng code và benchmark. Đây là phần “đã làm được”.

Tiếp theo, qua phân tích code và benchmark, đề tài **đã xác định đúng các giới hạn hiện tại**: metadata còn yếu, constraint chưa trung thành với reasoning, retrieval còn coarse và KG chưa sống được qua reasoning. Đây là phần “đã hiểu được vấn đề ở đâu”.

Cuối cùng, đề tài **đã chốt được định hướng rõ ràng cho giai đoạn tiếp theo**: xây một Financial Evidence Graph dùng chung cho retrieval và reasoning, nâng metadata thành ontology truy xuất, và đưa reasoning sang hướng executor + verifier + SFT/Step-DPO/GRPO. Đây là phần “sắp triển khai như thế nào”.

Theo logic đó, báo cáo tiến độ không chỉ chứng minh rằng đã có code, mà còn chứng minh rằng đã có sự trưởng thành về mặt nhận thức bài toán: từ một prototype retrieval có cấu trúc tiến dần sang một kiến trúc reasoning toàn trình có khả năng tạo đóng góp học thuật mạnh hơn.

---

## 11. Kết luận

Có thể kết luận ngắn gọn rằng đề tài hiện đang ở một trạng thái rất quan trọng: phần retrieval đã có một prototype đủ tốt để làm nền, nhưng nếu chỉ tiếp tục tối ưu retrieval thì giá trị nghiên cứu có thể chưa đủ mạnh. Hướng đi nên theo là mở rộng phần KG hiện có thành một Financial Evidence Graph thống nhất, dùng để giải quyết cả retrieval lẫn reasoning.

Như vậy, các thành phần đã làm được gồm:

- GSR cho retrieval có cấu trúc;
- CACL cho contrastive training với hard negatives có ý nghĩa miền;
- Constraint KG cho việc đưa cấu trúc tài chính vào retrieval.

Còn các bước tiếp theo cần triển khai gồm:

- nâng metadata thành ontology truy xuất;
- tối ưu lại constraint theo hướng trung thành với phương trình;
- phát triển retrieval phân cấp;
- xây evidence graph phục vụ cả retrieval và reasoning;
- thêm executor và verifier;
- huấn luyện reasoning theo pipeline SFT → Step-DPO → GRPO.

Đây là hướng vừa có tính kế thừa từ những gì đã làm được, vừa đủ mạnh để hình thành một công trình nghiên cứu sâu hơn trong giai đoạn tiếp theo.
