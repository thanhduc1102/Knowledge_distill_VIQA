# AAAI-27 — SOTA Survey, Benchmark & Baseline Selection for Verifier-Native Financial Reasoning

**Tài liệu đồng hành với:** [docs/AAAI27_RESEARCH_BLUEPRINT.md](AAAI27_RESEARCH_BLUEPRINT.md)  
**Tài liệu định vị lõi:** [docs/AAAI27_CORE_RESEARCH_DIRECTION_AND_STRATEGY.md](AAAI27_CORE_RESEARCH_DIRECTION_AND_STRATEGY.md)  
**Phạm vi:** khảo sát học thuật 2024-2026 cho numerical reasoning over financial data, lựa chọn benchmark, baseline, và khoảng trống đủ lớn cho AAAI-27.  
**Trục bài toán:** Financial Numerical Reasoning · Program-of-Thought · Evidence Grounding · Verifier-guided RL · Auditability

---

## 0. Executive Summary

Nếu nhìn vào các paper gần đây, có một kết luận rất rõ: frontier models chưa giải xong financial reasoning. Chúng vẫn yếu ở ít nhất năm trục:

1. **numerical precision**,
2. **step-level correctness**,
3. **long-document evidence handling**,
4. **rule-governed diagnostic completeness**,
5. **verification-aware inference selection**.

Vì vậy, framing mạnh nhất cho paper không nên là một câu chuyện về ngôn ngữ. Framing mạnh nhất là:

> **verifier-native financial numerical reasoning with executable programs, grounded evidence, and auditable inference.**

Từ góc nhìn đó, benchmark strategy hợp lý nhất là:

- **Core suite:** FinQA, TAT-QA, ConvFinQA, DocMath-Eval, FinChain
- **External probe:** ViNumQA
- **Adjacent extensions:** FinRule-Bench, FinDVer, FinAuditing

Và contribution đủ mạnh cho AAAI nên nằm ở giao của bốn thứ:

- reward design,
- evidence grounding,
- inference-time verification,
- structure-aware evaluation.

---

## 1. Landscape 2024-2026

## 1.1. Các cụm nghiên cứu chính

| Nhóm | Đại diện | Insight chính | Liên quan tới paper |
|---|---|---|---|
| **Financial QA / numerical reasoning** | FinQA, TAT-QA, ConvFinQA, MultiHiertt | Program-style reasoning trên bảng + văn bản | Core task family |
| **Long-document reasoning** | DocMath-Eval, FinDVer | LLM degrade mạnh trên tài liệu dài, nhiều bảng, hybrid evidence | Stress test cho grounding |
| **Verifiable reasoning** | FinChain, FinanceReasoning | Final answer là chưa đủ; cần step-level validity và executable traces | Hỗ trợ trực tiếp ECRL-Fin |
| **Rule / audit style reasoning** | FinRule-Bench, FinAuditing | Tài chính cần diagnosis, consistency checking, compliance-style reasoning | Mở rộng sang auditing |
| **Agentic retrieval / post-training** | FinAgent-RAG, Fino1/FinCoT, DianJin-R1, FEVO | Tool use, PoT, GRPO hữu ích nhưng chưa giải được verifier-native selection | Baseline và motivation |

## 1.2. Ý nghĩa của landscape này

Khoảng trống quan trọng nhất không còn nằm ở việc “thêm dữ liệu” hay “thêm prompt”. Nó nằm ở việc:

- đưa verifier vào training signal,
- đưa evidence structure vào representation,
- đưa verification vào inference-time selection,
- và mở rộng evaluation sang step-level cùng diagnostic errors.

---

## 2. Quantitative Findings Worth Using

Các con số dưới đây đủ mạnh để dùng trong motivation của paper:

- **FinanceReasoning (ACL 2025):** OpenAI o1 + Program-of-Thought đạt khoảng **89.1%**, nhưng paper vẫn nhấn mạnh lỗi numerical precision và nhu cầu tool support.
- **FinChain:** thiết kế benchmark để chấm cả final answer lẫn step-level chain validity, cho thấy frontier models vẫn còn khoảng cách rõ ở symbolic financial reasoning.
- **FinAgent-RAG (2026):** báo cáo EA khoảng **76.81** trên FinQA, **78.46** trên ConvFinQA, **74.96** trên TAT-QA, và giảm chi phí API nhờ adaptive router.
- **FinDVer:** ngay cả GPT-4o vẫn còn dưới chuyên gia người ở explainable verification trên long financial documents.
- **Fino1 / FinCoT line of work:** domain-specific post-training và GRPO hữu ích, nhưng long-document vẫn là điểm yếu.
- **VLSP 2025 system results:** PCPO giúp tăng PA đáng kể, cho thấy reward trên program validity là hướng đúng; tuy nhiên current setup vẫn còn thiếu step-level reward và verifier-ranked inference.

### Kết luận từ số liệu

Hai điều gần như đã được cộng đồng chấp nhận:

1. **Program/PoT-native reasoning tốt hơn mental-math reasoning** trong tài chính.
2. **GRPO/RLVR là hướng post-training hợp lý** cho domain này.

Điều chưa được giải quyết tốt là **verifier-native optimization end-to-end**.

---

## 3. Recommended Benchmark Suite

## 3.1. Core suite cho main paper

| Benchmark | Vai trò trong paper | Lý do chọn |
|---|---|---|
| **FinQA** | Anchor benchmark | Program supervision và `program_re` rất phù hợp với equivalence-class reasoning |
| **TAT-QA** | Hybrid text+table reasoning | Kiểm tra tính tổng quát ngoài FinQA |
| **ConvFinQA** | Compositional reasoning | Kiểm tra multi-step consistency ở setting hội thoại |
| **DocMath-Eval** | Long-document stress test | Tài liệu dài, đa bảng, executable solution |
| **FinChain** | Step-verifiable benchmark | Đo trực tiếp lợi ích của step-level reward |

## 3.2. Vì sao bộ 5 này hợp lý

Core suite này có ba ưu điểm rất mạnh:

1. **Task coherence:** đều nằm trong financial reasoning / program reasoning ecosystem.
2. **Difficulty coverage:** bao phủ short-table, hybrid, conversational, long-document, step-verifiable.
3. **Story coherence:** cho phép kể một câu chuyện thống nhất về verifier-native reasoning.

## 3.3. External probe

| Benchmark | Vai trò |
|---|---|
| **ViNumQA** | External language portability study |

ViNumQA nên được viết đúng vai trò: không phải benchmark định vị lõi, mà là phép thử xem reasoning structure học được có giữ được ở một setting khác hay không.

## 3.4. Adjacent extensions

- **FinRule-Bench:** rất mạnh cho rule reasoning và diagnostic completeness.
- **FinDVer:** phù hợp nếu muốn nối paper sang explainable claim verification.
- **FinAuditing:** phù hợp nếu muốn đẩy mạnh trục auditing/compliance.

Các benchmark này nên dùng để:

- củng cố motivation,
- hoặc làm appendix / follow-up,

thay vì nhét vào main table quá sớm.

---

## 4. Baseline Matrix

## 4.1. Frontier API baselines

| Nhóm | Cấu hình tối thiểu cần chạy | Vai trò |
|---|---|---|
| GPT family | GPT-4o / GPT-4.1 với CoT và PoT | Closed-source upper bound |
| OpenAI reasoning | o1 hoặc o3-mini với PoT | Reference point cho reasoning-heavy setting |
| Claude family | Sonnet / Opus với CoT và PoT | Model-family diversity |
| Gemini family | 1.5 / 2.x với CoT và PoT | Additional frontier comparison |

## 4.2. Prompting / inference baselines

| Baseline | Vai trò |
|---|---|
| Direct Answer | Lower bound |
| CoT | Standard reasoning baseline |
| PoT / PAL | Program-first baseline |
| Self-consistency | Standard inference-time boost |
| Majority vote over executable samples | Repo current baseline |

## 4.3. Open-model post-training baselines

| Baseline | Vai trò |
|---|---|
| SFT-only | Foundation post-training baseline |
| SFT + PCPO | Existing verifier-style reward baseline |
| SFT + ECRL-Fin | Proposed training contribution |
| SFT + ECRL-Fin + ranker | Full proposed method |

## 4.4. Retrieval / agentic baselines

| Baseline | Vai trò |
|---|---|
| FinQA-style retriever-generator | Classic financial QA pipeline |
| FinAgent-RAG-like setup | Strong retrieval + PoT reference |
| Optional decomposition agent | If reproducible within scope |

---

## 5. Frontier Model Gap Analysis

| Gap | Evidence from literature | Cơ hội cho paper |
|---|---|---|
| **Numerical precision** | FinanceReasoning vẫn ghi nhận precision issues dù dùng strong reasoning models | ECRL-Fin + executor-based training |
| **Step-level correctness** | FinChain phải thêm ChainEval vì final answer là không đủ | Step reward và verifier-native selection |
| **Long-document failure** | FinDVer và DocMath-style benchmarks cho thấy performance giảm mạnh trên tài liệu dài | Structured evidence + long-doc evaluation |
| **Rule / diagnostic completeness** | FinRule-Bench cho thấy reasoning phải đúng cả về quy tắc và chẩn đoán | Mở rộng sang auditing-style reasoning |
| **Inference calibration** | Self-consistency và answer majority không đảm bảo correct process | Verifier-ranked inference + abstention |

### Tóm tắt

Frontier models rất mạnh ở answer generation, nhưng paper này không cần thắng chúng ở mọi metric. Paper chỉ cần chứng minh rằng:

- verifier-native optimization tạo ra **lời giải đáng tin hơn**,
- **sai theo cách dễ phân tích hơn**,
- và **hiệu quả hơn** trong các setting cần auditability.

---

## 6. Positioning Against Existing Work

| Công trình | Reward-time verifier | Step-level signal | Evidence grounding | Inference-time verifier | Auditability story |
|---|---|---|---|---|---|
| FinQA retriever-generator | No | No | Partial | No | Weak |
| FinanceReasoning | No | Limited | Partial | Partial tool use | Medium |
| FinChain | No | Yes, but eval-only | No | No | Medium |
| FinAgent-RAG | No | No | Retrieval-focused | Partial self-verify | Medium |
| Fino1 / domain GRPO models | Partial correctness reward | No | No | No | Medium |
| **XFinReason** | **Yes** | **Yes** | **Yes** | **Yes** | **Strong** |

### Điểm khác biệt cần nhấn mạnh

Paper không nên tự bán như “một model tài chính nữa”. Nó nên được bán như:

- một reward design mới,
- một representation choice mới,
- một inference-time selection strategy mới,
- và một evaluation story chặt chẽ hơn cho financial reasoning.

---

## 7. Claims the Paper Can Realistically Defend

### Claim 1

**ECRL-Fin improves verifier-native financial reasoning beyond PCPO by adding equivalence-class and step-level training signals.**

### Claim 2

**Structure-aware evidence grounding reduces financial-specific grounding errors such as wrong period, wrong metric, and wrong unit selection.**

### Claim 3

**Verifier-ranked inference yields a better accuracy-cost-auditability trade-off than majority voting over final answers.**

### Claim 4

**The benefits of verifier-native reasoning persist across progressively harder settings, from short-table QA to long-document and step-verifiable reasoning.**

### Claim 5

**After the core method is validated, its reasoning structure retains partial external portability on ViNumQA.**

Claim 5 là câu hỏi phụ, không nên gánh toàn bộ novelty.

---

## 8. Experimental Priorities for This Repo

## Priority 1. Benchmark unification

Việc quan trọng nhất là có một benchmark/evaluation layer thống nhất cho:

- FinQA
- TAT-QA
- ConvFinQA
- DocMath-Eval
- FinChain

## Priority 2. Paper-grade ECRL ablation

Phải sớm có bảng:

- PCPO vs ECRL full
- no-equivalence
- no-step
- no-soft-valid
- no-brevity

## Priority 3. Inference upgrade

Nâng [pipeline/inference.py](../pipeline/inference.py) từ majority vote thành verifier-ranked selection.

## Priority 4. Structured evidence layer

Bắt đầu bằng schema nhẹ trước khi nghĩ tới graph phức tạp:

- identify metric,
- identify period,
- identify unit,
- trace provenance.

## Priority 5. External probe only after core results

Khi core suite đã ổn mới chạy ViNumQA portability study. Làm ngược lại sẽ khiến paper bị kéo lệch khỏi câu hỏi khoa học chính.

---

## 9. Suggested Reading List for the Paper

- **FinQA** — financial numerical reasoning over tables and text.
- **TAT-QA** — table-and-text financial QA.
- **ConvFinQA** — conversational financial reasoning.
- **DocMath-Eval** — long-document mathematical reasoning benchmark.
- **FinanceReasoning** — credible financial reasoning evaluation with program/tool support.
- **FinChain** — step-verifiable chain evaluation for finance.
- **FinDVer** — explainable verification over long financial documents.
- **FinRule-Bench** — rule-governed diagnostic reasoning on financial statements.
- **FinAgent-RAG** — agentic retrieval with PoT for finance.
- **Fino1 / FinCoT / DianJin-R1 / FEVO** — domain-specific post-training and RLVR lines.

---

## Final recommendation

Nếu mục tiêu là một AAAI paper mạnh, repo nên được trình bày như một nỗ lực về **verifier-native financial reasoning**, không phải như một nghiên cứu về ngôn ngữ. Bộ benchmark lõi nên là FinQA, TAT-QA, ConvFinQA, DocMath-Eval, và FinChain; ViNumQA nên được giữ lại như một external portability probe có giá trị, nhưng không chi phối framing của paper.