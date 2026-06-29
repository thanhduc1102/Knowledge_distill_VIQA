# SYSTEM — Bản đồ toàn trình (Retrieval → Generation → Reliability)

> **Tài liệu sống #1.** Mô tả *kiến trúc, phương pháp, kỹ thuật* và *hiện trạng* của toàn hệ thống.
> Cập nhật tài liệu này mỗi khi một module/kỹ thuật thay đổi. Số liệu nằm ở [RESULTS.md](RESULTS.md);
> lý thuyết nền ở [FRAMEWORK_TCEP.md](FRAMEWORK_TCEP.md); rủi ro/đánh giá ở [CONTRIBUTION_AUDIT.md](CONTRIBUTION_AUDIT.md).
> Cập nhật lần cuối: 2026-06-29.

---

## 0. Bài toán & định vị

**Task:** Retrieval-Augmented *numerical* financial QA — (1) truy hồi đúng bằng chứng giữa các bảng
tài chính nhìn giống nhau, (2) sinh ra một con số *kiểm chứng được*. Benchmark chính: **T²-RAGBench**
(FinQA / ConvFinQA / TAT-DQA); OOD: **FinanceBench**. Chi tiết benchmark → [RESULTS.md §1](RESULTS.md).

**Hai trục đóng góp (đã hiệu chỉnh theo bằng chứng):**
- **Trục Retrieval:** truy hồi cấu trúc (MMER) + tái lập SOTA *metadata-aware BM25* (leaderboard-faithful).
- **Trục Reliability (lõi novelty):** CPR (Concept–Period–Role) structure grounding → *cost-efficient
  selective answering*; phân tích khi nào structure thắng/thua model-internal confidence.

---

## 1. Luồng dữ liệu toàn trình

```
                ┌──────────────────────── PHA 1: RETRIEVAL ────────────────────────┐
 query ────────►│  (A) Metadata-aware BM25  ──┐                                     │
 (company/year  │      (leaderboard SOTA)     │   pool (∪ retriever top-k,           │──► top-k docs
  in question)  │  (B) MMER 7-expert fusion ──┘    KHÔNG nhồi gold)  → learned fusion │
                └───────────────────────────────────────────────────────────────────┘
                                                   │
                ┌──────────────────────── PHA 2: LEDGER ────────────────────────────┐
 top-k docs ───►│  Fact Ledger: (concept, entity, period, value, unit, scale, prov)  │──► selected facts
                │  trích orientation-aware từ trường `table` sạch + ontology tài chính │   + structure graph
                └───────────────────────────────────────────────────────────────────┘
                                                   │
                ┌──────────────────────── PHA 3: GENERATION ────────────────────────┐
 facts+context ►│  Generator (Gemini 2.5 Flash / Qwen / extractive) → "Answer: <num>" │──► answer
                └───────────────────────────────────────────────────────────────────┘
                                                   │
                ┌──────────────────────── PHA 4: RELIABILITY ───────────────────────┐
 answer ───────►│  CPR verifier (annotation-free) ⊕ model-internal (SC, verbalized)  │──► confidence
                │  → fusion → SELECTIVE ANSWERING (abstain) [KHÔNG override/re-ask]    │   + abstain
                └───────────────────────────────────────────────────────────────────┘
```

---

## 2. PHA 1 — Retrieval

### 2.1 Hai phương pháp song song (hai *setting* khác nhau, đều hợp lệ)

| Phương pháp | Setting | Metadata? | Kỹ thuật | Kết quả (W.Avg MRR@3) |
|---|---|---|---|---|
| **(A) Metadata-aware BM25** | leaderboard-standard | có (company/year trong query) | Sparse IR + filter/boost theo company+year | **0.747** (provided) / 0.673 (question-derived) |
| **(B) MMER 7-expert fusion** | content-only (honest) | chỉ từ câu hỏi | 7 expert độc lập + fusion học được, 5-fold CV | **0.736** |
| **(C) MMER 8-expert (+meta)** | honest (meta rút từ câu hỏi) | có (từ câu hỏi) | (B) + `meta` expert làm cột fusion thứ 8 | **0.798** (FinQA 0.846 / ConvFinQA 0.862) |
| **(D) MMER 8-expert (+meta-provided)** ⭐ | provided (metadata cấp kèm query) | có (provided) | (C) với `--meta-provided`; sector qua entity/GICS | **0.873** (FinQA 0.914 / ConvFinQA 0.932) |

> **(D) là cấu hình mạnh nhất — VƯỢT leaderboard #1 (~0.82) toàn cục** mà không dùng LLM frontier (đây là setting
> "metadata-aware" chuẩn của benchmark, tái lập SOTA trước đây). **(C)** là setting *honest* khó hơn (chỉ dùng metadata
> rút từ câu hỏi). Cross-encoder rerank generic **thất bại** (§2.5) → giữ MMER fusion. Chi tiết: [RESULTS.md §2](RESULTS.md).

> **Điểm hợp lệ của (A):** company/year **recoverable từ câu hỏi 87–98%** ([RESULTS.md §2](RESULTS.md)),
> nên dùng metadata là *hợp pháp* (nó là một phần truy vấn) — đúng như leaderboard #1 "Metadata-aware BM25".
> File: [scripts/research/metadata_aware_bm25.py](../scripts/research/metadata_aware_bm25.py).

### 2.2 MMER — 7 expert độc lập + đầu fusion học được

| # | Expert | Kỹ thuật | Tín hiệu | File |
|---|---|---|---|---|
| 1 | lexical | BM25 + abbr sentinel | sparse term match | [experts/lexical.py](../src/gsr_cacl/experts/lexical.py) |
| 2 | dense | bi-encoder e5-large-instruct | cosine | [experts/dense.py](../src/gsr_cacl/experts/dense.py) |
| 3 | lateint | late interaction (ColBERT-style, 1 vector/fact) | `max_f cos(q,f)` | [experts/late_interaction.py](../src/gsr_cacl/experts/late_interaction.py) |
| 4 | entity | **GICS ontology + SupCon** | cos(e_Q,e_D) | [experts/entity.py](../src/gsr_cacl/experts/entity.py) |
| 5 | concept | **ontology kế toán (42 concept + 7 identity)** | concept⊕period coverage | [experts/concept.py](../src/gsr_cacl/experts/concept.py) |
| 6 | cell | Fact Ledger (row-label,period) | overlap×period | [experts/cell.py](../src/gsr_cacl/experts/cell.py) |
| 7 | graph | structure KG (HierFinRAG-style) | structural-satisfaction | [experts/graph.py](../src/gsr_cacl/experts/graph.py) |

**Fusion head** ([experts/fusion.py](../src/gsr_cacl/experts/fusion.py)): linear / MLP / gate-MoE, huấn luyện
**listwise InfoNCE**, đánh giá **5-fold CV** (mỗi query chấm bởi head không train trên fold của nó). Trọng số học
được tự thích nghi: lexical+entity thống trị, concept/graph nhỏ (~0.05–0.09).

```
# Thuật toán MMER honest retrieval
1. meta ← extract(company/year) CHỈ từ câu hỏi              # honest contract
2. pool ← ∪ top-50(lexical, dense, lateint)                 # KHÔNG nhồi gold → recall ceiling thật
3. for expert i: s_i[d] ← expert_i(Q,d); minmax-normalize trong pool
4. F[pool×7] ← stack(s_i)
5. for fold in 5: head ← train_listwise_InfoNCE(F_train); score[fold] ← head(F_fold)
6. return argsort(score)[:k]
```

---

## 3. PHA 2 — Fact Ledger (cầu nối retrieve↔generate↔verify)

Trích từ trường `table` sạch (orientation-aware: row-major/col-major) → mỗi *fact* =
`(concept, entity, period, value, unit, scale, provenance)`. Module: [ledger/](../src/gsr_cacl/ledger/)
(`extract.py`, `select.py`, `fact.py`, `numeric.py`). Đồ thị cấu trúc: [kg/structure_graph.py](../src/gsr_cacl/kg/structure_graph.py).
- **Query-aware fact selection** ([ledger/select.py](../src/gsr_cacl/ledger/select.py)) → đưa *đúng cell* cho generator.
- **calculation_plan** gán vai operand (old/new, part/total, siblings) — *kiểm chứng đối xứng với generation*.
  ⚠ Heuristic này yếu khi đứng một mình (precision 11–24%, operand-F1 ~0.5 vs oracle) → xem [RESULTS.md §6](RESULTS.md).

---

## 4. PHA 3 — Generation

Interface chung `generate(query, evidence_block, meta, facts) → "Answer: <num>"`. Ba backend:
| Backend | File | Dùng cho |
|---|---|---|
| **Gemini 2.5 Flash** | [utils/gemini_client.py](../src/gsr_cacl/utils/gemini_client.py) | generator MẠNH (cache/retry/throttle, thinking=0) |
| HF (Qwen…) | [generation/generator.py](../src/gsr_cacl/generation/generator.py) | generator yếu/cục bộ |
| Extractive | nt | sàn không-GPU |

Number-Match (tolerance + scale-drift) theo chuẩn leaderboard ([ledger/numeric.py](../src/gsr_cacl/ledger/numeric.py)).

---

## 5. PHA 4 — Reliability (lõi novelty)

### 5.1 CPR verifier — [research/cpr_verifier.py](../src/gsr_cacl/research/cpr_verifier.py)
Đáp án được "supported" chỉ khi fact hỗ trợ đồng thời **Concept-consistent × Period-consistent × Role-consistent**
trên đồ thị cấu trúc. `conf = max(grounded, derivable, value_floor, raw_floor)`; period dùng partial-credit;
3-operand fallback ở confidence thấp. Annotation-free, model-free.

### 5.2 Tín hiệu reliability so sánh (trên generator MẠNH)
- **value-only** (legacy) — số có mặt/derivable bất kỳ.
- **CPR** (ours) — structure typed grounding.
- **self-consistency** — đồng thuận k=5 mẫu (chi phí 6×).
- **verbalized** — model tự chấm 0–1 (chi phí 2×).
- **fusion** — logistic CV trên các tín hiệu (tốt nhất).

### 5.3 Chính sách triển khai (đã chốt bằng thực nghiệm)
- **Selective answering (abstain khi conf thấp): DÙNG.** False-flag chỉ tốn coverage.
- **Verify-then-reask / answer-override: KHÔNG (trên model mạnh).** Net-negative — false-flag *hủy* đáp án đúng
  (xem [RESULTS.md §5](RESULTS.md)). CPR là tín hiệu *ranking/abstention*, không phải *arbitration*.

---

## 6. Tri thức tài chính được nhúng ở đâu (financial domain knowledge)

| Tri thức | Hiện thực | Dùng tại |
|---|---|---|
| **GICS 11 sectors** | [ontology/gics.py](../src/gsr_cacl/ontology/gics.py) | entity expert (E1), cross-company generalization |
| **Canonical concepts + alias** (Revenue/GrossProfit/…) | [ontology/concepts.py](../src/gsr_cacl/ontology/concepts.py) | concept expert (C3), CPR concept-consistency |
| **7 accounting identities** (Revenue−COGS=GrossProfit, OCF+ICF+FCF=ΔCash…) | `concepts.IDENTITIES` | concept-coverage scoring, structure graph, verifier |
| **Calculation templates** | [templates/library.py](../src/gsr_cacl/templates/library.py) | KG construction, intent→formula |
| **Period/scale/unit** | [ledger/numeric.py](../src/gsr_cacl/ledger/numeric.py) | ledger normalization, CPR period gate, NM scale-drift |
| **Company alias/acronym** | [ontology/aliases.py](../src/gsr_cacl/ontology/aliases.py) | metadata-aware retrieval, entity channel |

→ Tri thức tài chính **đã được tận dụng** ở cả retrieval (entity/concept/graph) và verification (CPR/identities).
Khoảng trống: ontology concept mới phủ ~14% → đòn bẩy "learned concept typing" ([RESULTS.md §6](RESULTS.md)).

---

## 7. Hiện trạng hệ thống (trạng thái từng cấu phần)

| Cấu phần | Trạng thái | Ghi chú |
|---|---|---|
| Metadata-aware BM25 (SOTA repro) | ✅ đo full set | W.Avg 0.747 (provided), leaderboard-faithful |
| MMER 7-expert fusion (honest) | ✅ đo full set, 5-fold CV | W.Avg 0.736, leak-free (đã sửa rò rỉ) |
| **MMER 8-expert (+meta, honest)** | ✅ đo full set | **W.Avg 0.798** — vượt #1 trên ConvFinQA |
| **MMER 8-expert (+meta-provided)** | ✅ đo full set | **W.Avg 0.873** — VƯỢT #1 toàn cục (FinQA 0.914) |
| Cross-encoder rerank (generic) | ✅ đo (negative) | phá thứ hạng → cần fine-tune in-domain |
| Fact Ledger + selection | ✅ chạy 3 dataset | trần trích xuất 0.45–0.80 (cap hệ thống) |
| **Learned operand attribution** | ✅ train+eval | deriv-hit 2–3×; soft-CPR trung tính (§6.4) |
| **Ontology + learned concept encoder** | ✅ 42→80 + encoder | coverage 14%→22–31% (exact) → ~98% (semantic, τ=.55) |
| **Retrieval→NM linkage** | ✅ đo | lift +0.34..+0.54 NM khi gold@rank1 vs vắng |
| Generation (Gemini mạnh) | ✅ 300/ds × 3 | acc 0.35–0.60 (vượt xa Qwen) |
| CPR reliability + fusion | ✅ đo + CI + cost-Pareto | cpr+verb (2×) > self-consistency (6×) |
| Selective answering | ✅ đo | coverage@acc, AURC |
| Benchmarks OOD | ✅ FinanceBench + **DocFinQA (long-doc)** | mẫu hình tổng quát hóa |
| Generation full-set / multi-generator | ⏳ một phần | củng cố C3 cross-model |

Next actions ưu tiên: xem [RESULTS.md §8](RESULTS.md) và [CONTRIBUTION_AUDIT.md](CONTRIBUTION_AUDIT.md).
