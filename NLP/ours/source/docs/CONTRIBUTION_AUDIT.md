# Critical Contribution Audit (honest, per-aspect)

> ⚠ **UPDATE 2026-06-29 (đọc [RESULTS.md](RESULTS.md) §4 trước).** Phần C1 dưới đây ("CPR beats self-consistency")
> dựa trên *generator yếu* (Qwen). Trên generator **mạnh** (Gemini 2.5 Flash), self-consistency & verbalized
> confidence **vượt** CPR standalone; đóng góp được tái định vị thành **cost-efficient reliability** (CPR+verbalized
> 2× ≥ self-consistency 6×) + bắt confident-hallucination trực giao + phổ generator-strength. Retrieval integrity
> đã sửa (leaky → honest 0.736; metadata-aware BM25 SOTA 0.747). Các con số tuyệt đối C0–C3 dưới đây là bối cảnh
> lịch sử; nguồn sự thật hiện tại là [RESULTS.md](RESULTS.md).

Goal: name exactly where each contribution is strong, **thin**, or **does not stand out**, what
criterion is under-measured, and what to add. Scored S (strong / submission-ready), M (medium /
needs work), W (weak / risk). Grounded in executed numbers.

## Problem positioning & benchmarks (must be explicit in the paper)

- **Task:** retrieval-augmented *numerical* financial QA — retrieve evidence among look-alike
  tables, then produce a *verifiable* number.
- **Benchmarks & role:**
  - **T²-RAGBench** (FinQA / ConvFinQA / TAT-DQA): primary; retrieval (MRR@3) **and** end-to-end
    Number-Match. This is where we keep the **SOTA retrieval** claim.
  - **FinanceBench** (real SEC 10-Ks): OOD generalization of the reliability signal.
  - **DocFinQA**: long-document setting — *planned* (cache currently broken).
- **Why these:** they isolate the two failures we target — evidence selection (retrieval) and
  answer auditability (verification/selective answering).

---

## C0 — Structure-aware retrieval (the SOTA evidence) — **S/M**

- **Strength (keep it):** MMER 7-expert fusion, honest 5-fold CV, leak-controlled, **W.Avg
  MRR@3 0.722** (FinQA 0.768 / ConvFinQA 0.782 / TAT-DQA 0.495) — beats every *non-oracle*
  leaderboard system (~0.40) and approaches #1 (GPT-5.4 meta-BM25 ~0.82) which uses gold metadata.
  This is a concrete "we solve the retrieval problem" demonstration.
- **Thin / does-not-stand-out:**
  1. It is **fusion-of-experts engineering**, not a deep novel mechanism — a reviewer can call it
     incremental. The *novelty* must be carried by C1 (CPR), with C0 as strong empirical support.
  2. **The structure/KG experts are the weakest contributors** (graph 0.02–0.09, concept
     0.02–0.08 standalone; fusion weights ~0.05–0.09). Lexical + entity dominate. So "structure
     helps retrieval" is a **weak claim** — be careful: structure's real payoff is in
     *verification* (C1), not retrieval.
  3. Retrieval is MRR-level; the leaderboard headline is metadata-driven (near-oracle), so beating
     it outright is not the goal — **frame metadata as an artifact**, claim the honest gap.
- **Add:** fine-tune the late-interaction encoder (currently pretrained) on same-company±1yr hard
  negatives to lift recall; report retrieval→NM correlation so retrieval gains are shown to matter.
- **⚠ INTEGRITY FLAG (must resolve before using any retrieval number):** the *honest* MMER result
  (5-fold CV, no gold metadata) is W.Avg **0.722** (FinQA 0.768 / ConvFinQA 0.782 / TAT-DQA 0.495,
  `docs/retrieval/07_research_report.md`). But the on-disk `outputs/modular/*/modular.json` now
  shows fusion MRR@3 FinQA **0.90** / TAT-DQA **0.70** with **pool_recall = 1.0** — i.e. a
  *leakier* pool/run than the documented honest one. **Use only the honest 0.722 numbers**; re-run
  the modular eval under the honest contract and overwrite the on-disk file, or the paper risks a
  leakage rejection. This is the same leaky-vs-honest trap that bit earlier retrieval arms.

## C1 — CPR structure grounding (the core novelty) — **S**

- **Strength:** best annotation-free reliability signal — AUROC beats value-only **and**
  self-consistency on all 3 (FinQA .576/.521/**.653**, ConvFinQA .638/.675/**.751**,
  TAT-DQA .662/.543/**.680**); significant on FinQA/ConvFinQA (paired bootstrap CI excludes 0);
  removes 60–85% of false "grounded" flags; transfers to OOD FinanceBench.
- **Thin / under-measured (fix before submission):**
  1. ✅ **Done — component ablation (`cpr_ablation.py`):** Role carries the gain (R-alone ≈ full),
     Concept is complementary (inert alone, C+R>R), **Period was the weak/harmful criterion**.
     Fixed with partial-credit period (modulate, not zero) → period now neutral-to-positive, full
     CPR ≥ CR. Reportable as a clean ablation. Remaining: the role check leans on a heuristic plan
     (see #2).
  2. **Role depends on the heuristic `calculation_plan`** — if operand roles are mis-assigned,
     role-consistency is wrong. We do **not** yet measure role-assignment accuracy. *Thin criterion.*
  3. **Concept ontology covers only ~14%** of canonical concepts → concept-consistency mostly uses
     crude token overlap. Expanding the ontology is low-risk upside.
  4. **Absolute AUROC 0.66–0.76**, not 0.9 — a useful but not strong classifier; the *selective*
     framing (C2) is what makes it compelling. Do not oversell it as a detector.
  5. **TAT-DQA gain is non-significant** (CI includes 0) — honest limitation (table-parse noise).

## C2 — Selective answering / AURC (deployability) — **M/S**

- **Strength:** deployable abstention — ConvFinQA safe coverage at ≥70% acc **7%→36%** (~5×); AURC
  improves on all 3 (e.g. ConvFinQA 0.442→0.375).
- **Thin:**
  1. **Collapses when base accuracy is low** — FinQA at 28% acc cannot reach a 60–70% bar at any
     coverage, so coverage@risk ≈ 0. The story **needs a stronger generator** (→ the Qwen2.5-3B
     full-set run) to demonstrate selective NM cleanly.
  2. **CV calibrator overfits at n=400** (helps at full-set scale). Report raw CPR conf as default.
  3. Selective *Number-Match* (not just selective grounded-accuracy) should be the headline metric.

## C3 — OOD generalization (FinanceBench) — **M**

- **Strength:** same pattern on real 10-Ks (AUROC 0.732→0.756, acc-when-supported 0.253→0.406,
  grounded_wrong −73%).
- **Thin:**
  1. **n=126, gold-evidence setting** (verification isolated from retrieval) → a reviewer will say
     "evidence-level, not document-level." Add a *retrieval* FinanceBench setting.
  2. **Single OOD benchmark**; no long-document (DocFinQA cache broken). Add ≥1 more.
  3. No bootstrap CI on FinanceBench yet.

## Extraction ceiling (the real bottleneck) — measured

`fact_extraction_recall.py`, gold-document ledger (isolates extraction from retrieval): fraction
of gold answers that are grounded-or-**derivable** from the auto-ledger = the *auditable ceiling*.

| Dataset | grounded | derivable | **certifiable (ceiling)** |
|---|---|---|---|
| FinQA | 0.046 | 0.424 | **0.47** |
| ConvFinQA | 0.300 | 0.394 | **0.694** |
| TAT-DQA | 0.154 | 0.458 | **0.612** |

**Implication:** even with the gold doc, ~47–69% of answers are certifiable — this caps verifier
AUROC and explains why absolute numbers are not higher. FinQA is lowest because its answers are
multi-operand computations (ratio/percent-change) the **2-operand** `is_derivable` cannot
reconstruct. *Biggest single lever to raise the whole system = extend derivation to ≥3 operands
and improve cell/operand extraction.* This is honest and points the next work precisely.

## Negatives (reported honestly — strengthen, don't hide)

- **Answer router fails** (pairwise raw-vs-KG arbitration by CPR conf does not beat raw NM despite
  +0.09–0.13 oracle headroom). This is informative: CPR is for *global ranking / abstention*, not
  *pairwise arbitration*. Keep as a finding; it sharpens the scope.
- **End-to-end Number-Match is modest** (small generators). The paper is reliability/selective,
  not SOTA-accuracy — but the **C0 retrieval SOTA** keeps a concrete "we solve part of the problem"
  anchor.

---

## Gap-filling status (this round)

- ✅ **Provenance precision audit** (`provenance_audit.py`, 100 inspectable samples/dataset):
  cited-cell concept+period precision = ConvFinQA **0.956** / TAT-DQA 0.760 / FinQA 0.702; "precision
  when well-cited" = ConvFinQA 0.636 / TAT 0.507 / FinQA 0.273 (low FinQA confirms grounding is
  coincidental for *computed* answers — consistent with TCEP Law 3).
- ✅ **Retrieval integrity diagnosed:** the on-disk leaky `modular.json` is a `company_pool`,
  n=150, 5-expert variant (pool = company-complete ⇒ recall 1.0 ⇒ near-oracle). Honest MMER
  (7-expert, pool=BM25∪dense, full test, 5-fold CV) = **0.722** (07_research_report). Procedural fix:
  re-run honest config (`--experts lexical,dense,entity,concept,cell,graph,lateint --cv 5`, no
  company_pool) to overwrite — queued behind GPU generation.
- ⏳ **Strong-generator full-set headline:** Qwen3-4B-Instruct-2507 full-set running.
- ⏳ **Self-consistency k≥5 baseline:** queued (needs GPU).
- ⚠ **DocFinQA blocked** (HF `DatasetGenerationError` + heavy 10-K files time out) → substituting a
  controlled *buried-evidence long-context* robustness test on existing data (`long_context_robustness.py`).

## Priority fixes (by ROI)

| Priority | Item | Addresses | Cost |
|---|---|---|---|
| 1 | CPR C/P/R ablation | "which criterion is thin" (C1) | running |
| 2 | Non-VLM full/large-set generation (Qwen2.5-3B/7B) | C2 selective NM, clean headline | GPU (running) |
| 3 | ✅ Fact-extraction recall (`fact_extraction_recall.py`) | reviewer-required, underpins all | done |
| 4 | Role-assignment accuracy probe | C1 thin criterion | low |
| 5 | Retrieval→NM correlation; encoder fine-tune | C0 "does retrieval matter" | medium |
| 6 | Ontology expansion beyond 14% | C1 concept signal | medium |
| 7 | DocFinQA repair + FinanceBench retrieval setting + bootstrap CI | C3 generalization breadth | medium |
