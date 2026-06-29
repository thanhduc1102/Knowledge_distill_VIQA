# RESULTS — Nguồn sự thật duy nhất cho mọi kết quả

> **Tài liệu sống #2.** Mọi số đều **thực thi thật** trong repo, ghi rõ artifact + lệnh tái lập. Không bịa.
> Phân biệt rõ *honest/content-only* vs *metadata-aware* vs *oracle*. Kiến trúc → [SYSTEM.md](SYSTEM.md).
> Cập nhật lần cuối: 2026-06-29. Phần cũ (gen-1/gen-2) lưu ở [archive/](archive/).

---

## 1. Benchmark & độ phủ

| Benchmark | Vai trò | #query | #corpus | Trạng thái |
|---|---|---|---|---|
| T²-RAGBench / FinQA | chính | 1147 | 2789 | ✅ full (retrieval + generation + reliability) |
| T²-RAGBench / ConvFinQA | chính | 3458 | 1806 | ✅ full |
| T²-RAGBench / TAT-DQA | chính | 1144 | 2723 | ✅ full |
| FinanceBench | OOD (generalization) | 150 (retr) / 126 (gen) | 172 | ✅ evidence-level |
| DocFinQA | long-document (123K-word 10-K) | 120/200 slice | full filing/doc | ✅ **đã tải (streaming)** + chunk→BM25→Gemini→CPR (§7b) |

**Hiện trạng độ phủ:** T²-RAGBench đầy đủ + FinanceBench OOD (evidence) + **DocFinQA long-document** (đã khắc phục lỗi
loader bằng streaming, `fetch_docfinqa.py`). Đa benchmark hơn rõ rệt so với vòng trước.

---

## 2. PHA RETRIEVAL

### 2.1 Metadata-aware BM25 — tái lập SOTA leaderboard (`metadata_aware_bm25.py`)
> Artifact: `outputs/research/metadata_bm25/report.json`. Full test set.

**Recoverability metadata từ câu hỏi** (luận cứ hợp lệ): company **0.87–0.90**, year **0.96–0.98**.

| Setting | FinQA | ConvFinQA | TAT-DQA | **W.Avg** |
|---|---:|---:|---:|---:|
| content-only (pure BM25 + abbr) | 0.665 | 0.641 | 0.417 | 0.601 |
| meta — **question-derived** (honest+meta) | 0.743 | 0.742 | 0.397 | 0.673 |
| meta — **provided** (= leaderboard standard) | **0.793** | **0.793** | **0.564** | **0.747** |

- meta-question ≈ meta-provided trên FinQA/ConvFinQA (company recoverable ~90%) → phần lớn giá trị metadata
  *truy cập được hợp pháp*. TAT-DQA: question-derived *kém hơn* content-only (year không phân biệt + detect nhiễu).

### 2.2 MMER fusion — honest (`modular_retrieval.py`, 5-fold CV)
> Artifact: `outputs/modular/` (7-expert) · `outputs/modular_meta/` (8-expert). **Đã sửa rò rỉ** (`*_LEAKY_companypool.json`).

| Cấu hình | FinQA | ConvFinQA | TAT-DQA | **W.Avg** |
|---|---:|---:|---:|---:|
| best standalone (lexical, in-pool) | 0.665 | 0.641 | 0.418 | 0.601 |
| **MMER 7-expert (content+structure)** | 0.795 | 0.782 | 0.538 | 0.736 |
| **MMER 8-expert (+meta, question-derived)** ⭐ | **0.846** | **0.862** | **0.554** | **0.798** |

> 8-expert thêm `meta` expert (company+year **rút từ câu hỏi**, honest) → vừa nâng pool-recall vừa là 1 cột fusion.
> **Beats leaderboard #1 trên ConvFinQA (0.862 > 0.845)**; W.Avg 0.798 tiệm cận #1 (~0.82) mà KHÔNG dùng LLM frontier.

### 2.3 Bối cảnh leaderboard & hai setting (T²-RAGBench)

> **Hai setting tách bạch — đặt lên bàn cân, không trộn:** (i) *honest/content-only* (company/year chỉ rút
> từ câu hỏi) là đóng góp khó & tổng quát; (ii) *metadata-aware/provided* (dùng company/year cấp kèm query —
> đúng quy ước benchmark, là setting của leaderboard #1) là "khai thác đúng chỗ", không phải đóng góp toàn cục.

| Hệ | setting | FinQA | ConvFinQA | TAT-DQA | **W.Avg** |
|---|---|---:|---:|---:|---:|
| non-oracle BM25/Hybrid (leaderboard) | content | ~0.40 | ~0.44 | ~0.29 | ~0.40 |
| MMER 7-expert (ours) | content-only | 0.795 | 0.782 | 0.538 | 0.736 |
| **MMER 8-expert +meta (ours)** | content+meta-from-Q | 0.846 | 0.862 | 0.554 | 0.798 |
| Metadata-aware BM25 (ours) | provided | 0.793 | 0.793 | 0.564 | 0.747 |
| **MMER 8-expert +meta-provided (ours)** ⭐ | provided | **0.914** | **0.932** | **0.653** | **0.873** |
| #1 GPT-5.4 + Metadata-aware BM25 (LLM rerank) | provided | 0.903 | 0.845 | 0.679 | ~0.82 |

**Đọc (trung thực):**
- **Setting honest:** MMER content-only **0.736**, +meta-từ-câu-hỏi **0.798** (vượt #1 ở ConvFinQA) — vượt xa mọi hệ non-oracle.
- **Setting provided (tái lập SOTA cũ của bạn):** MMER 8-expert +meta-provided đạt **W.Avg 0.873**, **VƯỢT #1 toàn cục**
  (FinQA 0.914 > 0.903; ConvFinQA 0.932 > 0.845; TAT 0.653 < 0.679) mà **không dùng GPT-5.4**. Đây là "khai thác metadata
  đúng chỗ" + bộ experts nội-pool mạnh; chỉ thua #1 ở TAT (bảng đa-cấp khó).

### 2.4 Ablation metadata 3-trường — giá trị biên của `company_sector` (`metadata_aware_bm25.py`)
> Artifact: `outputs/research/metadata_bm25/report_3field.json`. Recoverability từ câu hỏi: company 0.87–0.90, year 0.96–0.98.

| Setting (BM25) | FinQA | ConvFinQA | TAT-DQA |
|---|---:|---:|---:|
| content-only | 0.665 | 0.641 | 0.417 |
| **sector_only** (BM25 + sector prior) | 0.700 | 0.668 | **0.513** |
| meta_provided (company+year) | 0.793 | 0.793 | 0.564 |
| meta_provided_3field (+sector) | 0.793 | 0.793 | 0.564 |

**Phân tích sâu (3 trường):** (1) `company_sector` *một mình* đã hữu ích — vượt content-only, đặc biệt **TAT +0.10**
(nơi year không phân biệt); (2) **nhưng khi đã có `company_name` chính xác, sector trở nên thừa** (3field ≡ provided)
vì company đã khoanh vùng mịn hơn sector (12 sector vs 136 company). → Kết luận: dùng cả 3 là *đúng* nhưng đóng góp
thực của sector nằm ở vai trò *prior thô / dự phòng khi thiếu company*, không cộng thêm khi company đã đủ. Trong MMER,
sector vẫn được khai thác qua **entity expert (GICS)** — nên "dùng cả 3" được hiện thực trọn vẹn ở cấu hình 0.873.

### 2.5 Cross-encoder rerank — NEGATIVE mạnh (generic CE không transfer) — `cross_encoder_rerank.py`
> CE = `cross-encoder/ms-marco-MiniLM-L-6-v2` (huấn luyện trên web passage). Artifact: `outputs/research/cross_encoder/report.json`.

| Pool | FinQA | ConvFinQA | TAT-DQA |
|---|---:|---:|---:|
| bm25 | 0.652 | 0.632 | 0.411 |
| **bm25 + CE rerank** | 0.282 | 0.302 | 0.178 |
| meta_pool (company+year) | 0.878 | 0.882 | 0.559 |
| **meta_pool + CE rerank** | 0.369 | 0.419 | 0.370 |

**Phát hiện (trung thực):** cross-encoder generic **phá huỷ** thứ hạng (bm25 0.65→0.28; meta_pool 0.88→0.37) — vì văn bản
là **bảng/số** rất khác phân phối MS-MARCO; CE tính relevance theo ngữ nghĩa web nên xếp sai. ⇒ **Đề xuất "CE rerank đẩy
toward #1" thất bại với CE sẵn có.** Hướng đúng (đã xác định): **fine-tune cross-encoder trên cặp (query, doc) của
T²-RAGBench** (in-domain), hoặc giữ MMER fusion (vốn học in-domain) làm bộ xếp hạng — MMER fusion (0.80–0.87) >> generic CE.
Bài học khớp với lý do MMER tồn tại: tín hiệu phải *học in-domain*, biểu diễn đơn lẻ/generic không trị được bảng tài chính.

---

## 3. PHA GENERATION (Gemini 2.5 Flash, n=300/dataset)
> Artifact: `outputs/research/gemini_gen/{ds}_predictions.jsonl`. Number-Match (raw, T=0).

| Generator | FinQA | ConvFinQA | TAT-DQA |
|---|---:|---:|---:|
| Qwen-3B (cũ) | 0.117 | — | — |
| Qwen3.5-4B (cũ) | 0.278 | 0.440 | 0.273 |
| **Gemini 2.5 Flash** | **0.510** | **0.597** | **0.353** |

→ Generator mạnh nâng accuracy 2–4× → chế độ mà selective answering KHÔNG sụp (vá Blind Spot #3).

### 3b. Liên kết Retrieval → Number-Match (retrieval là đòn bẩy của output) — `retrieval_nm_linkage.py`
> Artifact: `outputs/research/retrieval_nm_linkage/report.json`. NM của Gemini phân tầng theo *rank của doc vàng*.

| Dataset | NM khi gold @rank-1 | gold @rank 2–3 | gold vắng top-3 | **lift (rank1 − vắng)** |
|---|---:|---:|---:|---:|
| FinQA | 0.586 (n=198) | 0.523 (n=65) | 0.081 (n=37) | **+0.505** |
| ConvFinQA | 0.630 (n=227) | 0.667 (n=51) | 0.091 (n=22) | **+0.539** |
| TAT-DQA | 0.489 (n=92) | 0.455 (n=99) | 0.147 (n=109) | **+0.342** |

**Bằng chứng nhân quả:** đưa được doc vàng lên rank-1 nâng Number-Match **+0.34 … +0.54** so với khi doc vàng vắng
top-3. Đây là minh chứng định lượng *retrieval tốt ⇒ output (NM) tốt* — biện minh trực tiếp cho việc đẩy retrieval lên
SOTA (§2). Đồng thời lý giải vì sao TAT-DQA NM thấp: 109/300 query có doc vàng vắng top-3 (retrieval TAT yếu nhất).

---

## 4. PHA RELIABILITY — vai trò & headline

> **Reliability làm GÌ (giải thích rõ):** T²-RAGBench chấm hai thứ — *retrieval* (MRR@3) và *output* (Number-Match).
> Tầng reliability **KHÔNG thay đổi NM**; nó là tầng phủ lên trên, gán cho mỗi đáp án một **độ tin cậy** để quyết định
> *câu nào nên trả lời / câu nào nên từ chối (abstain)*. Đầu ra triển khai: "trả lời iff confidence ≥ τ" → đổi *coverage*
> lấy *accuracy* (selective answering). Độ đo: **AUROC** (phân tách đúng/sai), **AURC / coverage@accuracy** (giá trị triển
> khai). Nói cách khác: retrieval+NM là *bài toán chính*; reliability là *lớp an toàn/khả kiểm chứng* để đưa hệ thống vào
> thực tế (tài chính cần biết *khi nào KHÔNG nên tin máy*). Đây là trục "trustworthy ML", bổ trợ chứ không thay thế NM.

### 4.0 HEADLINE (Gemini, n=300/ds, AUROC, paired-bootstrap CI)
> Artifact: `outputs/research/strong_reliability/report.json`. (`strong_reliability_eval.py`)

| Tín hiệu | chi phí | FinQA | ConvFinQA | TAT-DQA | **TB** |
|---|---:|---:|---:|---:|---:|
| value-only | 1× | 0.480 | 0.550 | 0.582 | 0.537 |
| **CPR (ours)** | 1× | 0.645 | 0.690 | 0.618 | 0.651 |
| self-consistency (k=5) | 6× | 0.767 | 0.708 | 0.773 | 0.749 |
| verbalized | 2× | 0.783 | 0.768 | 0.854 | 0.802 |
| **cpr ⊕ verbalized** | 2× | 0.805 | 0.810 | 0.852 | **0.822** |
| sc ⊕ verbalized | 7× | 0.839 | 0.815 | 0.884 | 0.846 |
| **fusion (learned, all)** | 7× | 0.823 | 0.822 | 0.868 | 0.838 |

**Kiểm định (paired bootstrap 2000):**
- CPR > value-only: FinQA P=1.0, ConvFinQA P=0.9995, TAT ns (P=0.82).
- fusion(all) > từng tín hiệu đơn: vs verbalized P=0.985/0.997/0.887; vs self-consistency P≥0.996.
- structure cộng thêm (fusion-all > fusion-internal): **chỉ ConvFinQA P=0.993**; FinQA P=0.53; TAT P=0.26.

### 4.1 Đường biên hiệu quả–chi phí (ĐÓNG GÓP CHÍNH)
**`cpr+verbalized` (2× chi phí, AUROC TB 0.822) Pareto-dominate `self-consistency` (6×, 0.749) trên CẢ 3 dataset.**
Self-consistency — chuẩn vàng đắt đỏ — bị structure thay thế. Ở 1× chi phí, CPR (0.651) >> value-only (0.537).

### 4.2 Cơ chế (vì sao fusion thắng) — `error_disjointness.py`
Ở ngân sách abstain 40%, CPR bắt thêm **+9.5 / +16.5 / +14.4%** số lỗi mà verbalized BỎ SÓT (confident-but-
ungrounded hallucinations). Union recall 0.77 / 0.83 / 0.74. → structure có **giá trị trực giao**.

### 4.3 Phổ generator-strength (C3)
| Generator | FinQA AUROC value-only→CPR | nhận định |
|---|---|---|
| Qwen-3B (yếu, acc ~0.12) | 0.531 → **0.637** (CPR thắng SC) | structure THIẾT YẾU |
| Gemini-2.5 (mạnh, acc ~0.51) | model-internal > CPR; cpr+verb cost-dominant | structure BỔ TRỢ hiệu quả-chi phí |

---

## 5. Chính sách inference — Verify-then-reask là NET-NEGATIVE trên model mạnh
> Artifact: `outputs/research/gemini_verify_reask/report.json`. (`gemini_verify_reask.py`)

| Dataset | raw NM | verify-then-reask NM | rescued | broke |
|---|---:|---:|---:|---:|
| FinQA | 0.510 | **0.447** | 7 | 26 |
| ConvFinQA | 0.597 | **0.540** | 2 | 19 |
| TAT-DQA | 0.353 | **0.297** | 9 | 26 |

→ Re-ask phá nhiều hơn cứu (vì trần trích xuất thấp ⇒ CPR false-flag nhiều đáp án raw-đúng; KG-filter cắt thông
tin). **Hệ quả:** dùng CPR cho **abstention**, KHÔNG cho **override**. (Trên Qwen yếu, verify-then-reask từng *dương*
nhẹ 0.278→0.295 → bằng chứng phổ generator-strength.)

---

## 6. Kiểm toán heuristic & tối ưu CPR (iterate-to-plateau)

### 6.1 Trần trích xuất (auditable ceiling, gold-doc) — `fact_extraction_recall.py`
| Dataset | grounded | certifiable 2-op | **certifiable 3-op** |
|---|---:|---:|---:|
| FinQA | 0.034 | 0.453 | **0.717** |
| ConvFinQA | 0.295 | 0.676 | **0.792** |
| TAT-DQA | 0.147 | 0.618 | **0.802** |
→ trần chặn trên toàn hệ; mở rộng ≥3-op là đòn bẩy lớn nhất (FinQA +0.26).

### 6.2 Heuristic gán vai (role) — `role_assignment_probe.py`
| Dataset | plan fire-rate | plan precision (standalone) | oracle operand-F1 | operation-acc |
|---|---:|---:|---:|---:|
| FinQA | 0.84 | 0.143 | 0.525 | 0.637 |
| ConvFinQA | 0.85 | 0.235 | 0.522 | 0.593 |
| TAT-DQA | 0.80 | 0.113 | 0.399 | 0.741 |
→ operand selection chỉ ~F1 0.5 vs oracle mạnh → **learned operand attribution** là đòn bẩy lõi tiếp theo.

### 6.3 Tối ưu CPR — `cpr_optimize.py`
- Component sweep: `full` (concept+period+role+3op) tối ưu cả 3.
- Học lại trọng số CPR: **không** cải thiện standalone (TAT −0.042 P=0.012) → giới hạn ở extraction/typing.
- Complementarity sạch: có ý nghĩa **chỉ ConvFinQA** (+0.044).

### 6.4 Learned operand attribution (đòn bẩy #1 — ĐÃ HIỆN THỰC & THẮNG) — `learned_operand_attribution.py`
> Distant supervision từ gold answer (operand set tái dựng gold); model = bge-small + logistic; 5-fold CV trên gold-doc ledger.
> Artifact: `outputs/research/learned_operand/report.json`.

| Dataset | operand-F1 learned | operand-F1 heuristic | **deriv-hit top6 learned** | deriv-hit heuristic |
|---|---:|---:|---:|---:|
| FinQA | 0.274 | 0.213 | **0.501** | 0.158 |
| ConvFinQA | 0.492 | 0.422 | **0.851** | 0.557 |
| TAT-DQA | 0.176 | 0.223 | **0.584** | 0.250 |

→ **Derivation hit-rate (top-6 operand đã học) gấp 2–3× heuristic** trên cả 3 → giải đúng "ambiguous multi-operand
attribution" (open problem TCEP §5). operand-F1 thắng FinQA/ConvFinQA, thua TAT (bảng nhiễu) nhưng deriv-hit vẫn thắng đậm.

**Cắm vào CPR (hard-restrict top-8) — NEGATIVE, có ích về hướng đi** (`cpr_learned_integration.py`, 5-fold CV leak-safe):

| Dataset | CPR full ledger | CPR top-8 learned (hard) | value-only |
|---|---:|---:|---:|
| FinQA | **0.645** | 0.606 | 0.452 |
| ConvFinQA | **0.690** | 0.639 | 0.542 |
| TAT-DQA | **0.621** | 0.506 | 0.570 |

→ Hard-restrict ledger **làm GIẢM** CPR AUROC (cả train-on-union vẫn giảm) → bản chất là **cắt cứng làm mất grounding cell thật**.

**Cắm vào CPR (soft-weight, ĐÃ HIỆN THỰC ĐÚNG) — `cpr_verifier.fact_weight_fn`** (down-weight, không loại; `report_soft.json`):

| Dataset | CPR full (baseline) | **CPR + soft learned-weight** | CPR + hard top-8 | value-only |
|---|---:|---:|---:|---:|
| FinQA | 0.645 | **0.643** | 0.616 | 0.452 |
| ConvFinQA | 0.690 | **0.676** | 0.643 | 0.542 |
| TAT-DQA | 0.621 | **0.628** | 0.498 | 0.570 |

**Kết luận (trung thực, đã hội tụ):** soft-weight **khắc phục thảm hoạ của hard-restrict** (kéo về ≈ full, TAT +0.007),
nhưng **không vượt** baseline CPR → tín hiệu operand-attribution **trùng lặp** với chính phép kiểm concept/period/role
của CPR cho mục đích *xếp hạng reliability*. ⇒ **Giá trị thực của learned-operand nằm ở chế độ derivation/ceiling**
(deriv-hit 2–3×, §6.4), KHÔNG ở reliability ranking. Đây là điểm dừng tối ưu của nhánh này (đã thử hard/soft/train-src).

### 6.5 Ontology expansion + Learned concept encoder
**(a) Mở rộng thủ công 42 → 80 concept** (`ontology/concepts.py`; nguồn XBRL US-GAAP/FASB). Coverage nhãn line-item
(khớp alias chính xác): FinQA **0.225** / ConvFinQA **0.246** / TAT-DQA **0.307** (≈ gấp đôi mốc 14% cũ).

**(b) Learned concept encoder (annotation-free)** — `concept_encoder.py` (bge-small, anchor = mean embedding alias mỗi concept):
- **Intrinsic** (5-fold, giữ lại alias, dự đoán concept từ anchor phần còn lại): **top-1 = 0.688** trên 80 lớp → encoder
  *tổng quát hoá* sang cách diễn đạt line-item chưa thấy (token-overlap không làm được).
- **Semantic coverage** (gán concept cho line-item *không* khớp alias, cos ≥ τ=0.55): exact 0.22–0.31 **+ semantic ~0.70–0.75
  = combined ~0.98–0.999**. → mở rộng coverage gần như toàn bộ, đánh đổi precision (~0.69 theo intrinsic). τ điều chỉnh
  precision/coverage. *Hướng sâu hơn:* fine-tune contrastive head trên cặp alias↔concept để nâng precision (annotation-free).

---

## 7. Benchmark OOD — FinanceBench
> Artifact: `outputs/research/external_financebench/`, `outputs/research/financebench_cpr/`.

| Retrieval (n=150) | MRR@3 |  | Reliability (n=126, Qwen-7B) | value-only → CPR |
|---|---:|---|---|---|
| BM25 | 0.321 |  | AUROC | 0.732 → **0.756** |
| BM25 + cross-encoder rerank | 0.456 |  | acc-when-supported | 0.253 → **0.406** |
| company loclex | 0.687 |  | supported-but-wrong | 71 → **19** (−73%) |
| **company-year loclex** | **0.814** |  | | |

→ Mẫu hình lặp lại trên 10-K thật (OOD) → tín hiệu mang tính cấu trúc, không phải artifact T²-RAGBench.

### 7b. Benchmark long-document — DocFinQA (123K-word 10-K) — `docfinqa_eval.py`
> Đã khắc phục lỗi loader bằng **streaming** (`fetch_docfinqa.py`). Pipeline: chunk filing → BM25 top-12 →
> Gemini answer → CPR. n=120. Artifact: `outputs/research/docfinqa/report.json`.

| Metric | Giá trị |  | Reliability AUROC | value-only | cpr | verbalized | cpr+verb |
|---|---:|---|---|---:|---:|---:|---:|
| base accuracy (Gemini) | 0.425 |  | | 0.427 | **0.528** | **0.645** | 0.631 |
| evidence recall@12 (BM25 trong doc) | 0.721 |  | | | | | |

→ **Cùng mẫu hình với T²-RAGBench/FinanceBench** trên long-document thật: CPR > value-only (0.528>0.427);
model-internal (verbalized 0.645) > CPR; complementarity giảm (chunk-extraction nhiễu hơn). Khẳng định
finding "phổ generator-strength + cost-efficient structure" *tổng quát hóa* sang chế độ tài liệu dài.

---

## 7c. Tổng hợp 3 đề xuất (round-3) — bảng kết luận khách quan

| Đề xuất | Kết quả | Phán quyết | Hướng tiếp |
|---|---|---|---|
| **Metadata 3-trường (provided) ⊕ MMER** | W.Avg **0.873** (FinQA 0.914/Conv 0.932/TAT 0.653) — **vượt #1** | ✅ **THẮNG** (tái lập+vượt SOTA) | giữ làm headline setting "provided"; sector qua entity/GICS |
| **Learned operand attribution** | deriv-hit **2–3×** heuristic (§6.4) | ✅ **THẮNG** ở derivation/ceiling | dùng cho nâng trần certifiable |
| → cắm vào CPR (soft-weight) | ≈ baseline CPR (không vượt) | ⚪ **TRUNG TÍNH** (trùng tín hiệu typing) | giá trị ở ceiling, không ở reliability ranking |
| **Cross-encoder rerank (generic)** | bm25 0.65→0.28; meta 0.88→0.37 | ❌ **THUA** (không transfer) | cần CE **fine-tune in-domain** |
| **Learned concept encoder** | intrinsic 0.688; coverage 22%→~98% (τ=.55) | 🟡 **MỘT PHẦN** (coverage↑, precision~0.69) | fine-tune contrastive nâng precision |

**Bài học xuyên suốt (đã kiểm chứng nhiều vòng):** tín hiệu phải **học in-domain** — MMER fusion (học) thắng generic CE;
metadata khai thác đúng chỗ cho SOTA; còn các tín hiệu *học thêm* (operand/concept) giúp ở đúng pha của chúng
(ceiling/coverage) chứ không nhất thiết cải thiện mọi pha (reliability ranking đã bão hoà bởi CPR typing).

---

## 8. Reproduce & Next Actions

```bash
cd ours/source && export PYTHONPATH=src           # GOOGLE_API_KEY/HF_TOKEN trong .env
# RETRIEVAL
python scripts/research/metadata_aware_bm25.py                                   # SOTA repro (CPU)
python scripts/modular_retrieval.py --dataset FinQA --cv 5 \
  --experts lexical,dense,entity,concept,cell,graph,lateint,meta --device cuda:1 --out outputs/modular_meta  # MMER 8-expert (W.Avg 0.798)
# RETRIEVAL — metadata-aware (provided) SOTA  W.Avg 0.873
python scripts/modular_retrieval.py --dataset FinQA --cv 5 \
  --experts lexical,dense,entity,concept,cell,graph,lateint,meta --meta-provided --device cuda:1 --out outputs/modular_meta_provided
python scripts/research/metadata_aware_bm25.py                                   # 3-field ablation (sector)
python scripts/research/cross_encoder_rerank.py --device cuda:0                  # CE rerank (NEGATIVE)
# OPERAND ATTRIBUTION + CONCEPT + LONG-DOC
python scripts/research/learned_operand_attribution.py --device cuda:0           # deriv-hit 2-3x heuristic
python scripts/research/cpr_learned_integration.py --device cuda:0 --train-src union  # soft vs hard CPR
python scripts/research/concept_encoder.py --device cuda:0                       # learned concept encoder
python scripts/research/fetch_docfinqa.py --n 200 --splits test                 # acquire DocFinQA (streaming)
python scripts/research/docfinqa_eval.py --n 120                                 # long-document e2e + CPR
# GENERATION + RELIABILITY + LINKAGE
for d in finqa convfinqa tatqa; do python scripts/research/gemini_generate.py --dataset $d --sample 300 --k-sc 5; done
python scripts/research/strong_reliability_eval.py      # headline + cost frontier
python scripts/research/retrieval_nm_linkage.py         # retrieval -> NM causal linkage
python scripts/research/cpr_optimize.py; python scripts/research/error_disjointness.py
python scripts/research/fact_extraction_recall.py --gold-doc-only --multi-op
python scripts/research/role_assignment_probe.py --oracle --sample 100
```

| # | Next action | Trạng thái | ROI |
|---|---|---|---|
| 1 | Learned operand attribution (deriv-hit 2–3×) | ✅ DONE (§6.4) | cao |
| 2 | Metadata ⊕ MMER (honest 0.798 + provided **0.873** vượt #1) | ✅ DONE (§2.3) | — |
| 3 | Long-document DocFinQA | ✅ DONE (§7b) | — |
| 4 | Ontology + learned concept encoder | ✅ DONE (§6.5) | — |
| 5 | Soft-weight learned-operand vào CPR | ✅ DONE (§6.4): trung tính (trùng typing) | — |
| 6 | Cross-encoder rerank | ✅ DONE (§2.5): generic THUA → cần fine-tune in-domain | — |
| 7 | **Fine-tune cross-encoder in-domain** (T²-RAGBench query–doc) → đẩy FinQA/TAT lên #1 | ⏳ next | cao |
| 8 | Fine-tune contrastive concept encoder (nâng precision 0.69→) | ⏳ | TB |
| 9 | Full-set generation + generator mạnh thứ 2 (Claude/GPT) → C3 cross-model | ⏳ | TB |
| 10 | Conformal selective answering trên fusion → đảm bảo coverage | ⏳ | thấp |
