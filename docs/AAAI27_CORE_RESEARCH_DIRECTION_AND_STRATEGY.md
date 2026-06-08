# AAAI-27 — Core Research Direction and Implementation Strategy

**Mục đích:** tái định vị toàn bộ nghiên cứu về đúng trục học thuật cốt lõi, bỏ trục `cross-lingual` và `multilingual` ra khỏi framing chính. Trọng tâm của bài báo và triển khai không còn là ngôn ngữ, mà là:

> **verifier-native, auditable, evidence-grounded financial numerical reasoning over tables and financial documents**.

Ngôn ngữ chỉ còn giữ vai trò **external portability study**: kiểm tra xem phương pháp lõi có còn hiệu lực khi áp dụng sang ngôn ngữ khác như tiếng Việt hay không.

---

## 0. Kết luận ngắn gọn

### 0.1. Vấn đề định vị trước đây

Nếu lõi benchmark và phần lớn tài nguyên đều tập trung ở tiếng Anh, thì việc biến trục ngôn ngữ thành headline chính sẽ làm nghiên cứu bị nhiễu. Nó kéo trọng tâm ra khỏi thứ reviewer thực sự quan tâm ở các công trình gần đây: 

- tính **kiểm chứng được** của suy luận,
- tính **đúng đắn theo bước** của lập luận,
- khả năng **ground evidence** từ bảng/tài liệu,
- năng lực reasoning trên **long document**,
- và **diagnostic completeness** trong bối cảnh tài chính rủi ro cao.

### 0.2. Framing đúng nên là gì

Framing đúng cho repo ở giai đoạn này nên là:

> **Verifier-native financial numerical reasoning with auditable programs, grounded evidence, and structure-aware inference.**

Trong framing này:

- tiếng Anh là **benchmarking substrate chính**,
- các ngôn ngữ khác là **external generalization probe**,
- và câu hỏi khoa học trung tâm là: *làm thế nào để mô hình suy luận tài chính đúng hơn, minh bạch hơn, và dễ kiểm toán hơn?*

### 0.3. Hệ quả chiến lược

Điều này dẫn đến hai lớp nghiên cứu tách biệt:

1. **Core paper**: thiết kế và kiểm chứng một framework verifier-native cho financial numerical reasoning trên benchmark tiếng Anh hiện đại.
2. **External language extension**: kiểm tra xem framework đó có còn hiệu lực khi mang sang tiếng Việt hoặc ngôn ngữ khác hay không.

Ngôn ngữ không còn là trục novelty chính. Nó là **phép thử tính bền của phương pháp**.

---

## 1. Landscape học thuật mới cần bám

### 1.1. Xu hướng 2024–2026 thật sự của lĩnh vực

Từ các benchmark và paper gần đây, có 5 trục nghiên cứu nổi lên rõ nhất:

1. **Verifiable step-level reasoning**
   - Đại diện: **FinChain**
   - Thông điệp: final answer là chưa đủ; cần đo step-level consistency.

2. **Credible and comprehensive financial reasoning evaluation**
   - Đại diện: **FinanceReasoning**
   - Thông điệp: benchmark cần credible hơn, chuẩn hóa hơn, và giàu công thức hơn; frontier models vẫn lỗi numerical precision.

3. **Long-document and hybrid-content reasoning**
   - Đại diện: **DocMath-Eval**, **FinDVer**
   - Thông điệp: mô hình degrade mạnh khi phải đọc tài liệu dài, nhiều bảng, nhiều loại evidence.

4. **Rule-governed and audit-style financial reasoning**
   - Đại diện: **FinRule-Bench**, **FinAuditing**
   - Thông điệp: trong tài chính, bài toán không chỉ là QA mà còn là rule verification, diagnosis, consistency checking.

5. **Agentic retrieval + self-verification**
   - Đại diện: **FinAgent-RAG**
   - Thông điệp: compositional reasoning cần retrieval lặp, tool use, và verification-aware selection, không chỉ one-pass retrieve-then-generate.

### 1.2. Hệ quả cho nghiên cứu của bạn

Nếu muốn AAAI-worthy, bài báo nên nằm ở giao của bốn thứ:

- **program-native reasoning**,
- **evidence grounding**,
- **verifier-based training/inference**,
- **structure-aware evaluation**.

Đây là điểm reviewer sẽ xem là nghiên cứu thực sự, thay vì một pipeline engineering ghép nhiều mô-đun.

---

## 2. Thesis mới cho paper

### 2.1. Thesis trung tâm

> Financial numerical reasoning should be optimized in a verifier-native space of executable programs and grounded financial evidence, rather than in the surface space of free-form answers.

### 2.2. Ý nghĩa của thesis này

Thesis này chuyển trọng tâm từ:

- “mô hình có trả lời đúng không?”

sang:

- “mô hình có suy luận đúng không?”
- “mô hình có dùng đúng evidence không?”
- “mô hình có cho ra một lời giải kiểm toán được không?”

### 2.3. Vì sao thesis này mạnh hơn trục ngôn ngữ

Vì nó trực tiếp khớp với pain point của các paper mới nhất:

- **FinChain**: thiếu đánh giá step-level verifiable reasoning.
- **FinanceReasoning**: frontier models vẫn lỗi precision dù điểm cao.
- **FinDVer**: long-context + explainability vẫn yếu.
- **FinRule-Bench**: rule diagnosis và diagnostic completeness rất khó.
- **FinAgent-RAG**: inference cần self-verification và routing theo độ khó.

Ngôn ngữ khi đó chỉ là một biến kiểm tra tính bền của framework, không phải trọng tâm lý thuyết.

---

## 3. Câu hỏi nghiên cứu cốt lõi

### RQ1 — Reward design

Liệu một reward verifier-native dựa trên equivalence class, execution, và step-level agreement có cải thiện program accuracy hơn reward sparse kiểu PCPO hay không?

### RQ2 — Evidence grounding

Liệu việc tách riêng bước grounding evidence theo schema tài chính có giảm lỗi chọn sai metric / period / unit so với Markdown-only prompting hay không?

### RQ3 — Inference selection

Liệu verifier-ranked inference có tốt hơn majority voting theo final answer trên cả accuracy, calibration và cost hay không?

### RQ4 — Structure-aware generalization

Liệu một framework được học trong không gian program + evidence có ổn định hơn khi chuyển từ short table QA sang long-document reasoning và rule-governed reasoning hay không?

### RQ5 — External language portability

Sau khi framework lõi đã được chứng minh trên benchmark chính tiếng Anh, liệu nó có giữ được một phần lợi ích khi áp dụng sang tiếng Việt hoặc ngôn ngữ khác hay không?

RQ5 là **secondary research question**, không còn là headline chính.

---

## 4. Giả thuyết nghiên cứu

- **H1:** ECRL-Fin cải thiện PA mạnh hơn PCPO vì nó reward chương trình theo equivalence class và consistency theo bước, thay vì chỉ hard-gate validity và final answer.
- **H2:** Evidence grounding theo schema tài chính giảm đáng kể lỗi metric/year/unit selection.
- **H3:** Verifier-ranked inference tạo ra accuracy-cost frontier tốt hơn self-consistency vote thuần.
- **H4:** Framework verifier-native giữ lợi thế rõ hơn trên long-document và rule-diagnostic settings so với prompting hoặc RAG thuần.
- **H5:** Nếu framework thực sự học được cấu trúc reasoning thay vì pattern bề mặt, nó sẽ có external portability tốt hơn sang các ngôn ngữ khác.

---

## 5. Phương pháp lõi nên xây

## 5.1. Contribution A — ECRL-Fin

**ECRL-Fin** là đóng góp thuật toán cốt lõi.

### Ý tưởng

Thay vì reward chỉ dựa trên:

- chương trình có hợp lệ không,
- answer cuối có đúng không,

thì ECRL-Fin reward thêm:

- symbolic equivalence,
- agreement của intermediate execution states,
- reward mềm cho validity gần đúng,
- tín hiệu độ ngắn gọn/ổn định của chương trình.

### Giá trị học thuật

Nó giải bài toán credit assignment trong reasoning tốt hơn PCPO và gắn trực tiếp với gap mà FinChain chỉ mới nêu ra ở **evaluation level**, chưa đưa vào **training objective**.

## 5.2. Contribution B — FinGraph-PoT

**FinGraph-PoT** là đóng góp representation + grounding.

### Ý tưởng

Biến context tài chính thành cấu trúc chuẩn hóa quanh các thực thể:

- metric,
- period,
- entity,
- value,
- unit/currency,
- provenance.

### Giá trị học thuật

Nó biến bài toán từ QA text-based thành **structure-governed reasoning**. Đây là bước quan trọng để:

- giảm noise từ bề mặt bảng,
- cho phép evidence-level evaluation,
- và mở đường sang rule verification/auditing tasks.

## 5.3. Contribution C — Verifier-ranked inference

Đây là đóng góp inference-time.

### Ý tưởng

Thay majority vote trên final answer bằng một ranker dựa trên verifier score:

- syntax validity,
- execution stability,
- step-level consistency,
- evidence consistency,
- optional self-repair confidence.

### Giá trị học thuật

Nó biến inference từ “lấy số nào lặp lại nhiều nhất” sang “chọn lời giải kiểm toán được nhất”. Đây là khác biệt mang tính phương pháp, không chỉ heuristic nhỏ.

## 5.4. Contribution D — Structure-aware critic / repair loop

Thay vì multi-agent đầy đủ ngay từ đầu, nên xây một version học thuật gọn hơn:

> Generator -> Executor -> Critic/Repair -> Verifier Ranker

### Giá trị học thuật

Điểm mạnh của mô-đun này là đo được trực tiếp nó sửa loại lỗi nào:

- syntax,
- wrong operator,
- wrong operand,
- wrong period,
- wrong unit.

Điều này giúp paper có error analysis rõ, đúng tinh thần conference paper mạnh.

---

## 6. Benchmark strategy mới

## 6.1. Bộ benchmark chính cho main paper

Core suite nên xoay về benchmark tiếng Anh cùng task family để tránh nhiễu từ trục ngôn ngữ:

1. **FinQA** — benchmark anchor cho program-style financial QA.
2. **TAT-QA** — hybrid table+text reasoning.
3. **ConvFinQA** — compositional / conversational financial reasoning.
4. **DocMath-Eval** — long-document and multi-table reasoning.
5. **FinChain** — verifiable step-level reasoning.

### Vì sao bộ 5 này tốt hơn

- cùng xoay quanh numerical/program reasoning,
- cùng nằm trong ecosystem benchmark mạnh nhất của lĩnh vực,
- đủ bao phủ từ short-table đến long-doc,
- và đủ để kể một câu chuyện thống nhất về verifier-native reasoning.

## 6.2. Benchmark phụ cho external validity

Sau khi đã chứng minh phương pháp trên core suite, mới thêm:

- **ViNumQA** — external portability probe cho tiếng Việt.
- **Optional**: SAHM, KFinEval-Pilot, hoặc benchmark khác nếu task family còn hợp lý.

### Vai trò của ViNumQA lúc này

Không còn là benchmark định vị chính.

Nó trở thành câu hỏi phụ rất có giá trị:

> Một framework được học để reasoning đúng và audit được trên benchmark chính tiếng Anh có cải thiện khi áp dụng sang tiếng Việt hay không?

Đây là external validation mạnh, nhưng không làm lệch trục nghiên cứu.

## 6.3. Benchmark gần kề nhưng không nên đưa vào main table

- **FinRule-Bench** — cực kỳ quan trọng về nghiên cứu, nhưng task là rule reasoning / diagnosis hơn là program QA. Nên dùng làm motivation và future extension.
- **FinDVer** — rất mạnh cho claim verification và long-hybrid docs, phù hợp làm adjacent benchmark hoặc analysis track.
- **FinAuditing** — rất sát auditing, nhưng là bài toán structure-aware multi-document benchmark rộng hơn numerical QA thuần.

---

## 7. Experimental design mới

## 7.1. Group A — Core reasoning results

Chạy trên 5 benchmark chính:

- FinQA
- TAT-QA
- ConvFinQA
- DocMath-Eval
- FinChain

So sánh:

- Direct
- CoT
- PoT/PAL
- Self-consistency
- SFT-only
- PCPO
- ECRL-Fin
- ECRL-Fin + verifier-ranked inference

## 7.2. Group B — Grounding and long-document robustness

Đo riêng trên:

- DocMath-Eval
- subset khó của TAT-QA / ConvFinQA

Metrics:

- evidence F1,
- table grounding accuracy,
- period/unit error rate,
- long-context degradation.

## 7.3. Group C — Structure-aware reasoning extension

Nếu đủ thời gian, thêm một track phụ trên:

- FinRule-Bench hoặc FinDVer

Mục tiêu không phải leaderboard, mà là chứng minh framework lõi chuyển tốt hơn sang:

- rule-governed reasoning,
- explainable verification,
- diagnostic completeness.

## 7.4. Group D — External language portability

Chỉ sau khi đã có kết quả ổn trên core suite:

- evaluate on ViNumQA,
- compare English-trained vs English+Vietnamese-enhanced variants,
- đo xem gain nào còn giữ được.

Phần này nên được viết như:

- `external language portability`,
- `non-English robustness`,
- `secondary extension study`.

Không nên dùng nó để tái dựng toàn bộ framing paper.

---

## 8. Baseline matrix mới

## 8.1. Theo trục method

- Direct Answer
- CoT
- PoT/PAL
- Self-consistency
- SFT-only
- PCPO
- ECRL-Fin
- ECRL-Fin + verifier-ranked inference

## 8.2. Theo trục retrieval/grounding

- No retrieval / raw context
- Standard RAG
- FinAgent-RAG-like retrieval
- FinGraph grounding
- FinGraph + verifier-ranked inference

## 8.3. Theo trục structure-awareness

- No critic
- Critic repair only
- Critic repair + verifier ranker
- FinRule/claim-verification transfer probe

## 8.4. Theo trục external language extension

- English-only core model evaluated on ViNumQA
- English-core + lightweight adaptation evaluated on ViNumQA
- If feasible, English-core + aligned mini-set adaptation on ViNumQA

---

## 9. Vì sao hướng này conference-worthy hơn

### 9.1. Nó bám đúng frontier học thuật

Reviewer sẽ thấy bài này chạm đúng các câu hỏi nóng:

- verifiable reasoning,
- step-level reward,
- financial auditing and diagnostic completeness,
- long-document evidence grounding,
- inference-time verification.

### 9.2. Nó tránh bị xem là “engineering mix”

Nếu headline là ngôn ngữ, người đọc dễ nghĩ bài chỉ là:

- trộn dataset,
- prompt/mix language,
- rồi kiểm tra transfer.

Khi headline là verifier-native reasoning, bài trở thành một contribution có chiều sâu hơn nhiều:

- reward theory,
- structure-aware representation,
- inference selection,
- evaluation science.

### 9.3. Nó vẫn giữ được chỗ cho tiếng Việt

Tiếng Việt không bị bỏ đi. Ngược lại, nó trở thành một external probe có ý nghĩa hơn:

- nếu phương pháp thật sự tốt, nó phải có external portability,
- nếu không, đó là một bằng chứng quan trọng về giới hạn của verifier-native transfer.

---

## 10. Roadmap triển khai chuẩn chỉ

## Phase 1 — Paper framing and benchmark reset

Mục tiêu:

- chốt framing mới,
- chốt core benchmark suite tiếng Anh,
- hạ vai trò trục ngôn ngữ thành extension study.

Deliverables:

- updated blueprint,
- updated survey,
- updated README and strategy docs.

## Phase 2 — Reproducible core suite

Mục tiêu:

- có pipeline chạy reproducibly trên FinQA, TAT-QA, ConvFinQA, DocMath-Eval, FinChain.

Deliverables:

- loaders,
- unified evaluator,
- run registry,
- reproducibility logs.

## Phase 3 — ECRL-Fin ablation

Mục tiêu:

- chứng minh H1,
- lấy contribution thuật toán mạnh nhất sớm nhất.

Deliverables:

- reward component logs,
- PA/EA/valid/step tables,
- ablation figure.

## Phase 4 — Grounding and verifier-ranked inference

Mục tiêu:

- chứng minh H2 và H3.

Deliverables:

- FinGraph prototype,
- evidence F1,
- verifier-rank vs majority frontier,
- critic-repair analysis.

## Phase 5 — Adjacent structured reasoning probe

Mục tiêu:

- thử framework trên FinRule-Bench hoặc FinDVer để xem nó có generalize sang auditing / verification hay không.

Deliverables:

- transfer analysis,
- diagnostic completeness discussion,
- future-work bridge.

## Phase 6 — External language portability

Mục tiêu:

- evaluate whether the validated core method improves non-English performance.

Deliverables:

- ViNumQA portability study,
- optional additional language extension,
- appendix or secondary section in paper.

---

## 11. Gợi ý packaging cho paper

## 11.1. Title candidates

1. **XFinReason: Verifier-Native Financial Numerical Reasoning over Tables and Documents**
2. **Auditable Financial Numerical Reasoning via Evidence-Grounded Program Synthesis**
3. **Equivalence-Class Reinforcement Learning for Verifiable Financial Reasoning**

## 11.2. Abstract structure

1. Financial reasoning requires correctness, transparency, and auditability.
2. Existing benchmarks and methods under-emphasize step-level validity, grounded evidence, and structure-aware selection.
3. We propose a verifier-native framework with ECRL-Fin, evidence grounding, and verifier-ranked inference.
4. We evaluate on a core suite spanning short-table, hybrid, conversational, long-document, and verifiable step-level reasoning tasks.
5. We show gains in PA/EA/step consistency and then probe external portability to non-English data.

## 11.3. Main result tables nên có

### Table 1 — Core suite main results

- FinQA
- TAT-QA
- ConvFinQA
- DocMath-Eval
- FinChain

### Table 2 — Reward ablation

- PCPO vs ECRL full vs no-step vs no-equiv vs no-soft-valid

### Table 3 — Grounding/ranking ablation

- Markdown-only vs FinGraph
- Majority vs verifier-rank

### Table 4 — External portability

- ViNumQA only as extension table

---

## 12. Final strategic recommendation

### Điều nên làm ngay

1. Bỏ hẳn `cross-lingual` và `multilingual` khỏi positioning chính.
2. Chuyển core benchmark suite về tiếng Anh và task-coherent.
3. Giữ tiếng Việt như external portability study.
4. Đẩy mạnh ba trục học thuật chính:
   - ECRL-Fin
   - FinGraph grounding
   - verifier-ranked inference
5. Dùng FinRule-Bench / FinDVer như motivation và extension sang auditing/verification, không nhất thiết ép vào main result table ngay.

### Một câu đóng gói cuối cùng

> This research is not primarily about language. It is about building a verifier-native financial reasoning framework that produces more auditable, evidence-grounded, and step-consistent solutions; non-English performance is then used as an external test of whether the learned reasoning structure is genuinely portable.