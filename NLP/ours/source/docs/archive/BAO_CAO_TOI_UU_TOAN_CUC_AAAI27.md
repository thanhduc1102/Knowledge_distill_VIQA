# Báo cáo tối ưu toàn cục cho định hướng AAAI-27

## 1. Trọng tâm nghiên cứu sau các vòng kiểm chứng

Trọng tâm không còn là "KG giúp LLM trả lời đúng hơn trong mọi trường hợp". Kết quả thực
nghiệm phản đối claim đó. Trọng tâm phù hợp hơn là:

**Structure-Aware, Auditable Retrieval for Financial RAG**

Nói ngắn gọn: trong QA tài chính, hệ thống phải giải hai bài toán:

1. **Tìm đúng bằng chứng trong cụm cùng công ty/tài liệu**, nơi dense embedding và reranker
   tổng quát thường không phân biệt tốt các bảng số gần giống nhau.
2. **Biết câu trả lời số nào có thể kiểm toán được**, tức câu trả lời có khớp fact/cell,
   period, phép tính và provenance hay không.

Vì vậy đóng góp nên xoay quanh:

- Structure-level KG: đồ thị document/table/row/column/fact/concept/period.
- `loclex` / conditional salience: baseline sparse mạnh trong pool cùng công ty/tài liệu.
- Fact Ledger: lớp kiểm toán, provenance, risk-coverage và verify-then-reask.
- Đánh giá artifact-controlled: tách hiệu ứng metadata/pool nhỏ khỏi năng lực rank thực sự.

## 2. Các vòng tối ưu đã thực thi

### Vòng 1: Mở rộng benchmark ngoài T2-RAGBench

Đã triển khai `scripts/research/external_financebench_eval.py` trên FinanceBench open-source
evidence retrieval.

Kết quả mới nhất:

| Method | MRR@3 | R@1 | R@3 | R@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.3211 | 0.2533 | 0.4133 | 0.4867 |
| BM25 masked company/year | 0.2056 | 0.1733 | 0.2533 | 0.3200 |
| company random | 0.3933 | 0.2200 | 0.6200 | 0.7667 |
| company-year random | 0.6300 | 0.4800 | 0.8133 | 0.9600 |
| company loclex | 0.6867 | 0.5533 | 0.8533 | 0.9467 |
| company-year loclex | 0.8144 | 0.7267 | 0.9200 | 0.9933 |
| BM25 + cross-encoder rerank | 0.4556 | 0.3667 | 0.5733 | 0.6667 |
| company loclex + cross-encoder | 0.6900 | 0.5667 | 0.8400 | 0.9533 |

Bootstrap:

- company loclex vs BM25: ΔMRR@3 `+0.3656`, CI95 `[0.3022, 0.4278]`.
- company-year loclex vs company-year random: ΔMRR@3 `+0.1844`, CI95 `[0.1233, 0.2478]`.
- company-year loclex vs company loclex: ΔMRR@3 `+0.1278`, CI95 `[0.0844, 0.1756]`.

Đánh giá khách quan:

- Kết quả ủng hộ conditional salience ngoài T2-RAGBench.
- Tuy nhiên FinanceBench hiện được đánh giá ở mức **evidence retrieval**, không phải full-PDF
  retrieval. Không được viết như benchmark full 10-K.
- `company-year random` đã rất cao vì pool nhỏ. Do đó claim đúng không phải "metadata là
  đóng góp", mà là: **sau khi metadata tạo pool nhỏ, loclex vẫn cải thiện ranking đáng kể so
  với random trong cùng pool**.
- Cross-encoder generic cải thiện BM25 nhưng gần như không thêm nhiều khi đã dùng company
  loclex. Điều này củng cố luận điểm sparse conditional salience rất mạnh trong bảng tài chính.

### Vòng 2: Faithfulness không dùng leakage

Đã sửa phân tích risk-coverage để confidence không dùng `reward`, vì `reward` trong prediction
có thành phần gold-match. Sau khi bỏ leakage:

| Dataset | grounded acc | ungrounded acc | gap CI95 | AUROC | AURC |
|---|---:|---:|---:|---:|---:|
| FinQA | 0.0966 | 0.0999 | [-0.0504, 0.0446] | 0.4976 | 0.8992 |
| ConvFinQA | 0.4504 | 0.1073 | [0.3135, 0.3715] | 0.7230 | 0.6234 |
| TAT-DQA | 0.2696 | 0.0952 | [0.1214, 0.2290] | 0.6381 | 0.8090 |

Đánh giá khách quan:

- ConvFinQA và TAT-DQA: grounding flag là tín hiệu đáng tin.
- FinQA: grounding flag hiện **không phân tách correctness**. Claim faithfulness phải ghi rõ
  là có điều kiện theo dataset/model/parser.
- Nguyên nhân lớn: verifier hiện kiểm "số có trong ledger" hơn là kiểm đủ concept + period +
  role. Điều này tạo `grounded_wrong`.

### Vòng 3: Verify-then-reask thay vì KG override

Policy triển khai được:

| Dataset | keep raw NM | reask if ungrounded NM |
|---|---:|---:|
| FinQA | 0.0994 | 0.1526 |
| ConvFinQA | 0.2432 | 0.2840 |
| TAT-DQA | 0.1399 | 0.1407 |

Oracle/offline headroom:

| Dataset | oracle reask-if-improves NM |
|---|---:|
| FinQA | 0.2058 |
| ConvFinQA | 0.3250 |
| TAT-DQA | 0.1923 |

Đánh giá khách quan:

- Verify-then-reask là hướng đúng hơn symbolic override vì không bỏ raw table context khi câu
  trả lời đã grounded.
- Hiệu quả thật còn khiêm tốn; TAT-DQA gần như không cải thiện với extractive re-ask.
- Có nhiều trường hợp raw đúng nhưng ungrounded và re-ask làm hỏng:
  - FinQA: 97 raw-correct-ungrounded; 61 bị reask làm sai.
  - ConvFinQA: 224 raw-correct-ungrounded; 142 bị reask làm sai.
  - TAT-DQA: 81 raw-correct-ungrounded; 59 bị reask làm sai.
- Do đó cần verifier tốt hơn trước khi re-ask quyết liệt.

### Vòng 4: Learned coordinate matcher

Đã triển khai weakly learned row/column matcher, học từ answer supervision trên gold doc.

| Dataset | heuristic | coord | learned | any-path |
|---|---:|---:|---:|---:|
| FinQA | 0.3459 | 0.2579 | 0.2642 | 0.3836 |
| ConvFinQA | 0.4189 | 0.3470 | 0.3790 | 0.4817 |
| TAT-DQA | 0.0975 | 0.1191 | 0.0939 | 0.1480 |

Đánh giá khách quan:

- Learned coordinate không thắng standalone.
- Giá trị thực là **tính bổ trợ giữa các path**: `any-path` cao hơn từng path riêng.
- Không nên claim "learned coordinate solves grounding". Claim phù hợp là "multipath
  evidence exposes complementary grounding signals and motivates calibrated selection".

### Vòng 5: Failure audit toàn cục

Retrieval:

| Dataset | top1 | top3 | miss |
|---|---:|---:|---:|
| FinQA | 0.6417 | 0.8675 | 152 |
| ConvFinQA | 0.7279 | 0.9274 | 251 |
| TAT-DQA | 0.3260 | 0.6198 | 435 |

Generation bucket:

| Dataset | grounded correct | grounded wrong | ungrounded correct | ungrounded wrong |
|---|---:|---:|---:|---:|
| FinQA | 17 | 159 | 97 | 874 |
| ConvFinQA | 617 | 753 | 224 | 1864 |
| TAT-DQA | 79 | 214 | 81 | 770 |

Đánh giá khách quan:

- TAT-DQA vẫn là điểm nghẽn retrieval thật sự: top3 chỉ `0.6198`.
- FinQA faithfulness yếu vì grounded-correct rất thấp so với grounded-wrong.
- ConvFinQA là dataset thể hiện tốt nhất giá trị verifier hiện tại.
- Nhiều lỗi retrieval thuộc `ratio`, `lookup`, `difference`, `percent_change`; đây là nhóm cần
  concept/period/value-role grounding, không chỉ local lexical.

## 3. Claim registry sau tối ưu

### Claim được giữ

1. **Conditional salience là đóng góp retrieval chính.**
   - Có bằng chứng trên T2-RAGBench và FinanceBench evidence retrieval.
   - Có bootstrap CI trên FinanceBench.

2. **KG/Fact Ledger nên dùng làm verifier và provenance layer, không phải answer override.**
   - KG evidence hard-filter từng làm giảm NM với model mạnh.
   - Verify-then-reask an toàn hơn và có cải thiện trên FinQA/ConvFinQA.

3. **Faithfulness là đóng góp có điều kiện, không đồng nhất.**
   - ConvFinQA/TAT có tín hiệu rõ.
   - FinQA hiện là negative result, phải báo cáo.

4. **Multipath grounding hữu ích hơn một matcher đơn lẻ.**
   - Learned coordinate không thắng, nhưng any-path cho thấy các tín hiệu bổ trợ.

### Claim bị loại

1. "KG giúp tăng Number Match trên mọi dataset."
2. "Accounting identity verifier là trung tâm."
3. "Fact-level neural/learned coordinate đã giải quyết grounding."
4. "FinanceBench result chứng minh full-document RAG." Hiện mới là evidence retrieval.
5. "Reward-threshold reask là deployable." Reward hiện có gold-match nên chỉ là oracle/offline.

## 4. Phương pháp cuối cùng nên viết trong bài

Pipeline nên được trình bày như sau:

1. **Entity-cluster candidate construction**
   - Dùng company/year như query metadata công khai.
   - Báo cáo random-within-pool để kiểm soát artifact pool nhỏ.

2. **Conditional-salience reranking**
   - Recompute sparse IDF trong pool.
   - Dùng local lexical score như tín hiệu chính.
   - So với BM25, masked BM25, random pool, cross-encoder rerank.

3. **Fact Ledger extraction**
   - Trích `(concept, period, value, unit, scale, provenance)`.
   - Không claim accounting identities là tín hiệu chính.

4. **Annotation-free verifier**
   - Grounding: answer value có match ledger không.
   - Arithmetic/process: nếu có phép tính rõ.
   - Provenance: cell/fact trace.
   - Cần nâng lên concept-period-aware grounding.

5. **Verify-then-reask**
   - Raw LLM answer trước.
   - Nếu ungrounded, re-ask bằng evidence/provenance.
   - Không symbolic override cứng.

6. **Risk-aware reporting**
   - grounded vs ungrounded accuracy.
   - AUROC/AURC.
   - hallucination catch proxy.
   - failure buckets.

## 5. Điểm chưa đủ cho AAAI nếu dừng tại đây

1. **External benchmark vẫn chưa đủ mạnh.**
   - FinanceBench hiện là evidence retrieval.
   - Cần thêm full-PDF/long-document setting hoặc DocFinQA full document nếu muốn claim tổng quát.

2. **Verifier còn nông.**
   - Grounded-wrong còn lớn.
   - Cần concept-period-role aware verification.

3. **TAT-DQA retrieval còn yếu.**
   - top3 `0.6198` làm generation bị chặn.
   - Cần structural/table-layout retrieval thay vì chỉ loclex.

4. **Reask policy còn thô.**
   - Reask-if-ungrounded làm hỏng nhiều câu raw đúng.
   - Cần calibrated verifier hoặc selective risk objective.

5. **Provenance precision chưa có audit người/annotation.**
   - Hiện vẫn là proxy.
   - Cần lấy 100 mẫu audit cell-level thủ công hoặc bán tự động.

## 6. Hướng tối ưu tiếp theo có giá trị nghiên cứu thật

### A. Concept-period-role verifier

Thay vì chỉ hỏi "số có trong ledger không", verifier phải kiểm:

- concept trong answer có khớp intent không;
- period/year có đúng không;
- value role là numerator/denominator/old/new/total hay không;
- scale/unit có nhất quán không.

Đây là hướng quan trọng nhất để giảm `grounded_wrong`.

### B. Calibrated reask selector

Huấn luyện hoặc fit một selector nhỏ trên annotation-free features:

- grounded flag;
- concept score;
- period score;
- arithmetic score;
- multipath agreement;
- retrieval margin;
- evidence conflict count.

Mục tiêu không phải tăng mọi câu, mà tối ưu risk-coverage: reask ít hơn nhưng đúng hơn.

### C. Full-document external evaluation

FinanceBench evidence retrieval là chưa đủ. Cần thêm:

- full PDF/page corpus nếu tải được FinanceBench documents;
- DocFinQA/MultiHiertt nếu dữ liệu có context dài;
- ít nhất báo cáo setting rõ ràng: evidence-level vs document-level.

### D. Table-layout-aware retrieval cho TAT-DQA

TAT-DQA cần tín hiệu cấu trúc:

- row/column header alignment;
- table section/caption;
- numerical role;
- multi-table context.

Không nên kỳ vọng dense hoặc loclex tự giải quyết hoàn toàn.

### Vòng 6: Structure-level KG arbitration

Đã triển khai `src/gsr_cacl/kg/structure_graph.py` và
`scripts/research/structure_graph_eval.py`.

Graph gồm các node:

- document;
- table;
- row;
- column;
- fact/cell;
- concept;
- period.

Edges:

- document -> table;
- table -> row/column;
- row/column -> fact;
- fact -> concept/period;
- temporal same-concept edges;
- accounting-support edges khi có.

Kết quả structure graph riêng:

| Dataset | Original top1 | Structure-only top1 | Best structure policy |
|---|---:|---:|---:|
| FinQA | 0.6417 | 0.5902 | 0.6504 |
| ConvFinQA | 0.7279 | 0.6486 | 0.7377 |
| TAT-DQA | 0.3260 | 0.3663 | 0.3767 |

Sau khi tích hợp vào KG bridge:

| Dataset | Original top1 | Best gated KG/structure top1 | Delta |
|---|---:|---:|---:|
| FinQA | 0.6417 | 0.6617 | +0.0201 |
| ConvFinQA | 0.7279 | 0.7420 | +0.0142 |
| TAT-DQA | 0.3260 | 0.3829 | +0.0568 |

Đánh giá khách quan:

- Đây là bằng chứng KG/structure-level tốt nhất hiện tại.
- Structure-only không thay thế retrieval được, nhưng khi dùng rank/confidence gating thì cải thiện
  cả 3 dataset.
- TAT-DQA hưởng lợi lớn nhất, đúng với giả thuyết rằng bảng phi chuẩn cần cấu trúc hơn lexical.
- Đây nên là contribution KG chính, còn `loclex` là strong sparse baseline/conditional salience.

## 7. Kết luận

Sau các vòng tối ưu, hệ thống đã rõ hướng nghiên cứu hơn:

- Có đóng góp retrieval thật: conditional salience.
- Có đóng góp reliability thật nhưng có điều kiện: Fact Ledger faithfulness/provenance.
- Có chính sách inference hợp lý hơn: verify-then-reask.
- Có negative results cần viết thẳng: FinQA faithfulness, learned coordinate, KG override.

Trạng thái hiện tại tốt hơn nhiều về mặt trung thực nghiên cứu, nhưng để đạt mức AAAI mạnh
cần thêm ít nhất một trong hai nâng cấp:

1. full-document external benchmark; hoặc
2. concept-period-role verifier đủ mạnh làm giảm grounded-wrong và tăng selective reliability.
