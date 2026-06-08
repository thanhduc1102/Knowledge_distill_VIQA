# AAAI-27 Research Proposal & Paper Blueprint

**Chủ đề:** Verifier-Native Financial Numerical Reasoning over Tables and Documents  
**Trục bài toán:** Financial Numerical Reasoning, Program-of-Thought, Evidence Grounding, Verifier-guided RL, Auditable AI  
**Mục tiêu hội nghị:** AAAI-27 Main Technical Track

> **Tài liệu đồng hành:** khảo sát SOTA, lựa chọn benchmark và baseline xem [docs/AAAI27_SOTA_SURVEY_AND_BENCHMARKS.md](AAAI27_SOTA_SURVEY_AND_BENCHMARKS.md).  
> **Tài liệu định vị lõi:** chiến lược nghiên cứu và roadmap tổng thể xem [docs/AAAI27_CORE_RESEARCH_DIRECTION_AND_STRATEGY.md](AAAI27_CORE_RESEARCH_DIRECTION_AND_STRATEGY.md).

---

## 0. Executive Summary

Blueprint này đặt lại bài toán về đúng trục học thuật cốt lõi: **không nghiên cứu ngôn ngữ như mục tiêu chính**, mà nghiên cứu cách xây dựng một framework financial reasoning:

- sinh ra chương trình có thể thực thi,
- ground được evidence từ bảng và tài liệu tài chính,
- được tối ưu bởi verifier thay vì chỉ tối ưu đáp án cuối,
- và cho ra lời giải có thể kiểm toán.

Luận điểm trung tâm là:

> **Financial numerical reasoning nên được học trong không gian verifier-native của executable programs và grounded evidence, thay vì trong không gian câu trả lời bề mặt.**

Vì vậy, main paper nên được kiểm chứng trên một **core benchmark suite** đồng nhất theo task family:

1. **FinQA**
2. **TAT-QA**
3. **ConvFinQA**
4. **DocMath-Eval**
5. **FinChain**

Trong framing này, **ViNumQA** vẫn rất quan trọng, nhưng được đặt ở vai trò đúng hơn: **external language portability study**. Nghĩa là sau khi phương pháp lõi đã được xác nhận trên benchmark chính, ta mới dùng ViNumQA để kiểm tra xem reasoning structure học được có còn hiệu lực khi áp dụng sang tiếng Việt hay không.

Repository hiện đã có nhiều thành phần đủ mạnh để phát triển thành paper thực sự:

- pipeline KD 6 pha trong [pipeline/run.py](../pipeline/run.py),
- DSL executor trong [pipeline/program_executor.py](../pipeline/program_executor.py),
- symbolic evaluation trong [pipeline/evaluate.py](../pipeline/evaluate.py),
- reward ECRL-Fin trong [pipeline/reward.py](../pipeline/reward.py),
- wiring GRPO trong [pipeline/train_grpo.py](../pipeline/train_grpo.py),
- test regression trong [tests/test_reward.py](../tests/test_reward.py).

Điểm cần làm tiếp không còn là “thêm ngôn ngữ”, mà là **chuẩn hóa benchmark, hoàn thiện grounding, xây verifier-ranked inference, và tạo một evaluation story đủ chặt cho AAAI**.

---

## 1. AAAI Positioning

## 1.1. Reviewer AAAI sẽ tìm gì

Reviewer AAAI thường không bị thuyết phục bởi một pipeline challenge nếu nó chỉ cho thấy:

- fine-tuning + RL hoạt động,
- vài ablation nhỏ,
- và điểm tăng trên một benchmark hẹp.

Để đủ mạnh cho main track, paper cần thuyết phục ở ít nhất bốn lớp:

1. **Methodological novelty**: có một đóng góp phương pháp rõ ràng, không chỉ tích hợp hệ thống.
2. **General AI relevance**: vấn đề và phương pháp có ý nghĩa vượt khỏi một dataset đơn lẻ.
3. **Empirical rigor**: benchmark hợp lý, baseline mạnh, ablation sạch, phân tích lỗi đủ sâu.
4. **Insight**: paper giải thích được vì sao mô hình đúng hoặc sai, không chỉ báo bảng điểm.

## 1.2. One-sentence pitch

Pitch phù hợp cho paper này là:

> **We propose a verifier-native framework for financial numerical reasoning in which training, inference, and evaluation operate over executable programs and grounded evidence rather than free-form answer strings.**

Điểm mạnh của pitch này là nó:

- khớp với các benchmark mới như FinChain và FinanceReasoning,
- mở cửa sang auditing, rule reasoning, và long-document reasoning,
- và biến repo từ competition system thành research framework.

## 1.3. Vì sao repo hiện tại có lợi thế tốt

Không phải project nào cũng có sẵn ba thứ cùng lúc:

- **executable DSL**,
- **symbolic PA evaluation**,
- **post-training loop bằng GRPO**.

Repo này đã có cả ba. Do đó, hướng verifier-native không phải là đổi trục hoàn toàn; nó là **nâng cấp trực tiếp trên nền đang có**.

---

## 2. Thesis, Research Questions, and Hypotheses

## 2.1. Thesis trung tâm

> Financial numerical reasoning becomes more reliable when models are trained and selected in a verifier-native space of executable programs and grounded financial evidence.

## 2.2. Research questions

**RQ1. Reward design**  
Liệu một reward dựa trên equivalence class, execution, và step-level agreement có cải thiện program accuracy tốt hơn reward sparse kiểu PCPO hay không?

**RQ2. Evidence grounding**  
Liệu việc chuẩn hóa context thành schema tài chính quanh `metric / period / entity / value / unit / provenance` có giảm lỗi grounding so với Markdown-only prompting hay không?

**RQ3. Inference selection**  
Liệu verifier-ranked inference có tốt hơn majority voting theo final answer về accuracy, calibration và cost hay không?

**RQ4. Structure-aware generalization**  
Liệu framework verifier-native có giữ lợi thế khi chuyển từ short-table QA sang long-document và step-verifiable reasoning hay không?

**RQ5. External portability**  
Sau khi phương pháp lõi được xác nhận trên benchmark chính, liệu lợi ích đó có còn giữ được khi áp dụng sang ViNumQA hay không?

## 2.3. Hypotheses

**H1:** ECRL-Fin cải thiện PA mạnh hơn PCPO vì reward đi trên không gian chương trình và trạng thái trung gian, thay vì dựa chủ yếu vào validity cứng và final answer.  
**H2:** FinGraph-style grounding làm giảm lỗi chọn sai metric, sai period, sai unit.  
**H3:** Verifier-ranked inference tạo ra accuracy-cost frontier tốt hơn self-consistency vote thuần.  
**H4:** Gain của verifier-native training rõ hơn trên long-document và step-verifiable benchmarks so với short-table QA đơn thuần.  
**H5:** Nếu framework thực sự học được reasoning structure, một phần gain sẽ vẫn giữ trên ViNumQA như một external portability probe.

---

## 3. Proposed Framework: XFinReason

## 3.1. Tên framework

Tên làm việc nên giữ là:

> **XFinReason**

Tên này đủ trung tính để không khóa paper vào một benchmark hay một ngôn ngữ cụ thể.

## 3.2. Kiến trúc tổng thể

```text
Question + Financial Context
        |
        v
  Structure-aware Formatting
        |
        v
   Generator (SFT / GRPO)
        |
        +--> Program Candidate
        +--> Optional Evidence Trace
        |
        v
 Executor + Verifier
        |
        +--> Validity
        +--> Execution Result
        +--> Program Equivalence
        +--> Step Consistency
        +--> Evidence Consistency
        |
        v
 Critic / Repair / Ranker
        |
        v
 Auditable Final Output
```

## 3.3. Contribution A: ECRL-Fin

Đây là đóng góp thuật toán trung tâm và cũng là phần đã được cài đặt gần nhất với paper-ready state.

### Mục tiêu

Thay reward hard-gated của PCPO bằng reward mềm hơn nhưng vẫn verifier-native, gồm:

- syntax/validity,
- execution correctness,
- symbolic program equivalence,
- agreement của intermediate steps,
- answer consistency,
- brevity/stability penalty nhỏ.

### Trạng thái hiện tại trong repo

- logic reward: [pipeline/reward.py](../pipeline/reward.py)
- execution theo bước: [pipeline/program_executor.py](../pipeline/program_executor.py)
- symbolic equivalence fallback: [pipeline/evaluate.py](../pipeline/evaluate.py)
- GRPO wiring: [pipeline/train_grpo.py](../pipeline/train_grpo.py)

### Giá trị học thuật

Các paper mới đã cho thấy step-level verification quan trọng, nhưng phần lớn dừng ở **evaluation**. ECRL-Fin là cách biến insight đó thành **training signal**.

## 3.4. Contribution B: FinGraph-PoT

Đây là contribution representation và grounding.

### Ý tưởng

Chuyển context tài chính từ text/table raw sang graph hoặc schema chuẩn hóa quanh các nút:

- `entity`
- `metric`
- `period`
- `value`
- `unit`
- `source/provenance`

### Mục tiêu khoa học

FinGraph-PoT không chỉ giúp retrieval. Nó còn cho phép:

- evidence-level supervision,
- error analysis theo dimension tài chính,
- và mở đường sang rule verification hoặc auditing tasks.

### Lưu ý triển khai

Nếu thời gian hạn chế, version đầu chỉ cần một **structured evidence layer** chứ chưa cần graph neural component. Reviewer sẽ chấm việc representation có tạo insight và measurable gain hay không, không đòi graph phức tạp.

## 3.5. Contribution C: Verifier-ranked inference

Hiện repo dùng majority voting ở [pipeline/inference.py](../pipeline/inference.py). Đây là điểm đủ tốt cho challenge, nhưng chưa đủ mạnh cho paper.

### Hướng cần nâng cấp

Thay vì chọn theo tần suất đáp án, rank candidate bằng verifier score tổng hợp từ:

- syntax validity,
- execution success,
- step consistency,
- program equivalence confidence,
- evidence consistency.

### Lý do học thuật

Majority vote là heuristic yếu vì nó không phân biệt lời giải đúng do reasoning tốt với lời giải trùng số nhưng sai quy trình. Verifier-ranked inference trực tiếp nhắm vào auditability.

## 3.6. Contribution D: Critic / Repair loop

Nếu còn đủ thời gian, nên bổ sung một loop gọn:

> Generator -> Executor -> Critic -> Repair -> Verifier Ranker

Mục tiêu của loop này không phải là dựng multi-agent phức tạp, mà là:

- giảm syntax errors,
- giảm wrong-operator errors,
- giảm wrong-period / wrong-unit errors,
- và tạo error taxonomy đủ rõ trong paper.

---

## 4. Codebase Audit: Những gì đã có và còn thiếu

## 4.1. Những gì đã có

### Pipeline core

- orchestration: [pipeline/run.py](../pipeline/run.py)
- config dataclasses và GPU profiles: [pipeline/config.py](../pipeline/config.py)
- data preparation: [pipeline/data_prep.py](../pipeline/data_prep.py)
- teacher distillation: [pipeline/teacher_distill.py](../pipeline/teacher_distill.py)
- SFT: [pipeline/train_sft.py](../pipeline/train_sft.py)
- GRPO: [pipeline/train_grpo.py](../pipeline/train_grpo.py)
- inference: [pipeline/inference.py](../pipeline/inference.py)
- evaluation: [pipeline/evaluate.py](../pipeline/evaluate.py)

### Paper-oriented components đã có

- ECRL-Fin reward dispatcher trong [pipeline/reward.py](../pipeline/reward.py)
- `execute_program_steps()` trong [pipeline/program_executor.py](../pipeline/program_executor.py)
- commutative symbolic fallback trong [pipeline/evaluate.py](../pipeline/evaluate.py)
- regression tests trong [tests/test_reward.py](../tests/test_reward.py)

## 4.2. Những gì còn thiếu cho AAAI

1. **Unified benchmark runners** cho FinQA, TAT-QA, ConvFinQA, DocMath-Eval, FinChain.
2. **Benchmark-specific evaluators** được chuẩn hóa vào cùng một reporting layer.
3. **Verifier-ranked inference** thay cho majority vote thuần.
4. **Structured evidence extraction** để đo grounding errors một cách có hệ thống.
5. **Experiment registry** lưu seed, cost, GPU profile, checkpoint, và metrics.
6. **Paper-grade plots/tables** cho reward ablation, grounding ablation, cost frontier.

## 4.3. Thông điệp quan trọng

Nền kỹ thuật lõi đã đủ để làm research paper. Việc còn thiếu chủ yếu là:

- packaging khoa học,
- benchmark breadth,
- và evaluation discipline.

---

## 5. Benchmark Strategy

## 5.1. Core benchmark suite cho main paper

| Benchmark | Vai trò | Vì sao cần |
|---|---|---|
| **FinQA** | Anchor benchmark | Program-style financial QA, có `program_re`, rất phù hợp cho equivalence reasoning |
| **TAT-QA** | Hybrid reasoning | Bảng + văn bản, đa phép toán, kiểm tra robustness ngoài FinQA |
| **ConvFinQA** | Compositional reasoning | Multi-turn / compositional structure, tốt cho kiểm tra consistency |
| **DocMath-Eval** | Long-document reasoning | Stress test tài liệu dài, đa bảng, có `python_solution` |
| **FinChain** | Step-verifiable reasoning | Đo trực tiếp lợi ích của step-level reward |

## 5.2. External extension

| Benchmark | Vai trò |
|---|---|
| **ViNumQA** | External language portability probe sau khi framework lõi đã được xác nhận |

## 5.3. Adjacent extensions

Nếu muốn mở rộng chiều auditing / verification trong appendix hoặc follow-up:

- **FinRule-Bench** cho rule-governed reasoning,
- **FinDVer** cho long-hybrid explainable verification,
- **FinAuditing** nếu muốn đẩy mạnh diagnostic completeness.

## 5.4. Vì sao không dùng ViNumQA làm benchmark chính của paper

Vì benchmark chính phải phục vụ câu hỏi khoa học lõi của paper. Trong giai đoạn này, câu hỏi lõi là:

- reward verifier-native có hiệu quả không,
- grounding có giảm lỗi cấu trúc không,
- inference có nên được rank bằng verifier không,
- và framework có giữ được lợi thế trên các benchmark ngày càng khó hơn không.

ViNumQA vẫn rất hữu ích, nhưng hợp lý nhất khi đóng vai trò **external validation**, không phải trục chính của storytelling.

---

## 6. Baseline Matrix

## 6.1. Frontier API baselines

- GPT-4o / GPT-4.1 với CoT và PoT
- OpenAI o1 / o3-mini với PoT
- Claude Sonnet / Opus với CoT và PoT
- Gemini 1.5 / 2.x với CoT và PoT

Vai trò của nhóm này là xác lập:

- khoảng cách tới frontier closed-source,
- kiểu lỗi numerical precision,
- và giới hạn ở long-document / step-verifiable settings.

## 6.2. Prompting and inference baselines

- Direct answer
- CoT
- PoT / PAL
- Self-consistency
- Majority voting over executable candidates

## 6.3. Open-model post-training baselines

- SFT-only
- SFT + PCPO
- SFT + ECRL-Fin
- SFT + ECRL-Fin + verifier-ranked inference

## 6.4. Retrieval / agentic baselines

- FinQA-style retriever-generator
- FinAgent-RAG style retrieval + PoT baseline
- Optional decomposition agent baseline if reproducible

---

## 7. Experimental Design

## 7.1. Group A: Core benchmark results

Chạy toàn bộ phương pháp chính trên 5 benchmark lõi để chứng minh framework không bị khóa vào một dataset riêng lẻ.

### Báo cáo tối thiểu

- main metric của từng benchmark,
- execution validity,
- latency / sample,
- cost nếu dùng API baselines.

## 7.2. Group B: Reward ablation

So sánh:

- PCPO
- ECRL full
- ECRL không `program_equiv`
- ECRL không `step`
- ECRL không `soft_valid`
- ECRL không `brevity`

Mục tiêu là chứng minh chính xác phần nào của reward tạo gain.

## 7.3. Group C: Grounding ablation

So sánh:

- Markdown-only context
- Structured evidence layer
- Structured evidence + verifier-ranked inference

Metrics nên thêm:

- evidence F1,
- period selection accuracy,
- unit selection accuracy,
- row/column grounding accuracy.

## 7.4. Group D: Inference and calibration

So sánh:

- greedy decode,
- self-consistency,
- majority vote,
- verifier-ranked inference,
- verifier-ranked inference + abstain threshold.

Mục tiêu là dựng accuracy-cost frontier và chứng minh verifier dùng tốt hơn token majority.

## 7.5. Group E: External portability study

Chỉ thực hiện sau khi core suite đã ổn.

### Thiết kế tối thiểu

- evaluate English-trained core model trên ViNumQA,
- evaluate English-trained + lightweight adaptation trên ViNumQA,
- compare với Vi-centric baseline hiện có.

### Thông điệp cần giữ

Kết quả này là **external validation** của reasoning structure, không phải headline chính của paper.

---

## 8. Metrics and Error Taxonomy

## 8.1. Metrics chính

- **EA**: execution accuracy
- **PA**: symbolic program accuracy
- **ChainEval** hoặc step-level metric cho FinChain
- **Evidence F1** cho grounding
- **Validity rate**
- **Latency / cost**

## 8.2. Error taxonomy

Paper nên có taxonomy ít nhất gồm:

1. **Syntax error**
2. **Operator error**
3. **Operand selection error**
4. **Period error**
5. **Unit / scale error**
6. **Evidence omission**
7. **Spurious answer agreement**: đáp án đúng nhưng quy trình sai
8. **Long-context failure**

Điểm mạnh của verifier-native framing là taxonomy này có thể đo được khá trực tiếp.

---

## 9. Implementation Roadmap

## Phase 1: Benchmark reset

Mục tiêu:

- chốt core suite,
- chốt protocol đánh giá,
- dọn toàn bộ language-centric framing khỏi paper story.

Deliverables:

- updated docs,
- benchmark registry,
- run plan.

## Phase 2: Reproducible core suite

Mục tiêu:

- có runner và evaluator ổn định cho FinQA, TAT-QA, ConvFinQA, DocMath-Eval, FinChain.

Deliverables:

- scripts/configs cho từng benchmark,
- unified results schema,
- saved artifacts.

## Phase 3: ECRL-Fin paper ablation

Mục tiêu:

- chứng minh H1 thật rõ.

Deliverables:

- reward component ablation,
- PA/EA gains,
- step consistency analysis.

## Phase 4: Grounding + verifier-ranked inference

Mục tiêu:

- chứng minh H2 và H3.

Deliverables:

- structured evidence layer,
- ranking module,
- grounding error analysis.

## Phase 5: Long-document and adjacent transfer

Mục tiêu:

- chứng minh H4 trên DocMath-Eval / FinChain,
- optional probe trên FinRule-Bench hoặc FinDVer.

## Phase 6: External portability study

Mục tiêu:

- kiểm tra H5 trên ViNumQA.

Deliverables:

- extension table,
- discussion về what transfers and what does not.

---

## 10. Paper Packaging

## 10.1. Title candidates

1. **XFinReason: Verifier-Native Financial Numerical Reasoning over Tables and Documents**
2. **Auditable Financial Numerical Reasoning via Evidence-Grounded Program Synthesis**
3. **Equivalence-Class Reinforcement Learning for Verifiable Financial Reasoning**

## 10.2. Abstract skeleton

1. Financial numerical reasoning requires correctness, transparency, and auditability.
2. Existing methods under-emphasize step-level verification, grounded evidence, and inference-time selection.
3. We propose XFinReason, a verifier-native framework with ECRL-Fin, structure-aware grounding, and verifier-ranked inference.
4. We evaluate on a core suite spanning short-table, hybrid, conversational, long-document, and step-verifiable reasoning.
5. We show gains in program accuracy, step consistency, and auditability, then probe external portability on ViNumQA.

## 10.3. Main tables nên có

- **Table 1:** Core suite main results
- **Table 2:** Reward ablation
- **Table 3:** Grounding + ranking ablation
- **Table 4:** Cost / calibration frontier
- **Table 5:** External portability on ViNumQA

---

## 11. Risks and Mitigations

## Risk 1: Reviewer cho rằng paper chỉ là system integration

**Cách giảm rủi ro:** đẩy ECRL-Fin thành contribution thuật toán rõ nhất; không để paper chỉ xoay quanh pipeline.

## Risk 2: Grounding module quá nặng so với tiến độ

**Cách giảm rủi ro:** bắt đầu bằng structured evidence layer tối giản; không cần graph neural hay retrieval phức tạp ở version đầu.

## Risk 3: Benchmark breadth lớn nhưng kết quả chưa đồng đều

**Cách giảm rủi ro:** ưu tiên ba benchmark đầu để ổn định pipeline; DocMath-Eval và FinChain dùng làm stress tests có chọn lọc trước.

## Risk 4: External portability không tăng rõ

**Cách giảm rủi ro:** viết đúng vai trò của ViNumQA là external probe; không khóa novelty của paper vào kết quả đó.

---

## 12. Immediate Repo Actions

1. Chuẩn hóa runners/evaluators cho 5 benchmark lõi.
2. Tạo experiment configs cho `sft`, `pcpo`, `ecrl`, `ecrl+ranker`.
3. Thiết kế results schema chung cho EA, PA, validity, step metrics, cost.
4. Nâng [pipeline/inference.py](../pipeline/inference.py) từ majority vote sang verifier-ranked selection.
5. Bổ sung structured evidence extraction ở data layer trước khi xây full FinGraph.
6. Để ViNumQA ở extension phase sau khi core suite có kết quả đủ chặt.

---

## Final positioning statement

> This research is not primarily about language. It is about building a verifier-native financial reasoning framework that produces more auditable, evidence-grounded, and step-consistent solutions; ViNumQA is then used as an external probe of whether the learned reasoning structure remains portable.