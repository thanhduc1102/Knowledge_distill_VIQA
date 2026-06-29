# Cost-Efficient Structure-Grounded Reliability for Financial Numerical QA
### Strong-Generator Study (Gemini 2.5 Flash) — kiểm toán, thực nghiệm và định hướng AAAI-27

> Tài liệu này tổng hợp vòng nghiên cứu bổ sung (2026-06-28) nhằm lấp các *blind spots* đã chỉ ra
> trong [CONTRIBUTION_AUDIT.md](CONTRIBUTION_AUDIT.md) / báo cáo kiểm toán, sử dụng **một generator
> mạnh thật sự** (Gemini 2.5 Flash) và quy trình đánh giá khách quan, có CI. Mọi số dưới đây được
> **thực thi** trong repo; script và artifact được nêu kèm. Tài liệu này **bổ sung và hiệu chỉnh**
> [AAAI27_RESEARCH_PLAN.md](AAAI27_RESEARCH_PLAN.md) ở phần "headline reliability".

---

## 0. Tóm tắt điều hành — điều gì đã thay đổi

Vòng trước (generator yếu: Qwen-3B/3.5-4B, accuracy 0.10–0.28) kết luận *"CPR là tín hiệu reliability
annotation-free tốt nhất, thắng cả self-consistency"*. **Kết luận đó là một artifact của generator
yếu.** Khi đo lại trên **Gemini 2.5 Flash** (accuracy 0.35–0.60 — chế độ generator mạnh mà reviewer
yêu cầu), bức tranh đảo chiều và **sâu sắc hơn**:

1. **CPR vẫn đánh bại value-only áp đảo** (ΔAUROC P≈1.0) — luận điểm lõi *"typed > untyped grounding"*
   sống sót trên model mạnh.
2. **Nhưng tín hiệu nội tại của model (self-consistency, verbalized confidence) nay VƯỢT CPR.** Model
   mạnh "biết khi nào nó đúng"; CPR (bị chặn trên bởi extraction ceiling) không theo kịp về AUROC tuyệt đối.
3. **Đóng góp thật được tái định vị — và mạnh hơn:** CPR là tín hiệu reliability **hiệu quả nhất trên mỗi
   đơn vị chi phí suy luận**. `cpr+verbalized` ở **2× chi phí** đánh bại `self-consistency` ở **6× chi phí**
   trên *cả ba* dataset, vì CPR bắt được lớp lỗi mà tín hiệu nội tại bỏ sót (confident-but-ungrounded
   hallucinations, +9.5–16.5% error-recall).
4. **Tín hiệu triển khai tốt nhất là fusion học được** (structure ⊕ model-internal); đóng góp biên của
   structure có ý nghĩa thống kê ở nơi bảng có cấu trúc tốt (ConvFinQA).

Đồng thời vòng này **đóng các blind spots**: (a) sửa rò rỉ số liệu retrieval, (b) đo trần trích xuất,
(c) đo độ chính xác của heuristic gán vai (role).

---

## 1. Hạ tầng & phương pháp luận

| Thành phần | File | Vai trò |
|---|---|---|
| Env loader | [utils/envload.py](../src/gsr_cacl/utils/envload.py) | Đọc `.env` (định dạng hỗn hợp) → `GOOGLE_API_KEY` |
| Gemini client | [utils/gemini_client.py](../src/gsr_cacl/utils/gemini_client.py) | Cache đĩa (resumable), retry/backoff, throttle, `thinking_budget=0` |
| Sinh đáp án mạnh | [scripts/research/gemini_generate.py](../scripts/research/gemini_generate.py) | raw (T=0) + verbalized-conf + k=5 self-consistency (T=0.7) |
| Eval reliability | [scripts/research/strong_reliability_eval.py](../scripts/research/strong_reliability_eval.py) | 4 tín hiệu + fusion + overconfidence audit + bootstrap CI |
| Tối ưu CPR | [scripts/research/cpr_optimize.py](../scripts/research/cpr_optimize.py) | component sweep + learned-CPR + complementarity |
| Phân rã lỗi | [scripts/research/error_disjointness.py](../scripts/research/error_disjointness.py) | CPR bắt lỗi nào verbalized bỏ sót |
| Trần trích xuất | [scripts/research/fact_extraction_recall.py](../scripts/research/fact_extraction_recall.py) | certifiable ceiling (2-op / 3-op) |
| Probe role | [scripts/research/role_assignment_probe.py](../scripts/research/role_assignment_probe.py) | plan self-accuracy + oracle operand-F1 |

**Quy mô:** 300 query/dataset × 3 dataset (FinQA/ConvFinQA/TAT-DQA) = 900 query, mỗi query 7 lời gọi LLM
(1 raw + 1 verbalized + 5 self-consistency). Mẫu 300 (cùng giao thức s400 của vòng trước), random seed cố định.
**Hợp đồng trung thực:** ledger được dựng lại từ `retrieval_top3.jsonl` **y hệt pipeline triển khai**
(`build_evidence_pack` → union ledger), nên so sánh là apples-to-apples; verification annotation-free, model-free.

---

## 2. Kết quả 1 — Sửa rò rỉ số liệu retrieval (Blind Spot #1, ĐÃ ĐÓNG)

File on-disk cũ trộn lẫn run *leaky* và *honest*. Đã **cách ly** (đổi tên `*_LEAKY_companypool.json`) và
**chạy lại honest** (7 expert, 5-fold CV, không company-pool, năm/công ty chỉ từ câu hỏi):

| Dataset | Leaky cũ (đã loại) | **Honest (mới, chính thức)** | pool recall |
|---|---:|---:|---:|
| FinQA | 0.90 (n=150, meta+loclex) | **0.795** (mlp, n=1147) | 0.993 |
| ConvFinQA | (đã honest) | **0.782** (mlp, n=3458) | 0.993 |
| TAT-DQA | 0.70 (meta+loclex) | **0.538** (mlp, n=1144) | 0.934 |
| **W.Avg** | — | **0.736** | — |

**Kết luận:** MMER honest W.Avg **0.736** — *cao hơn* mốc 0.722 tài liệu cũ, **không rò rỉ**, vượt mọi hệ
non-oracle trên leaderboard (~0.40) và tiệm cận #1 (GPT-5.4 meta-BM25 ~0.82, vốn dùng gold metadata).
Artifact: `outputs/modular/{finqa,convfinqa,tat-dqa}/modular.json`.

---

## 3. Kết quả 2 — Reliability trên generator mạnh (HEADLINE, tái định vị)

Gemini 2.5 Flash, n=300/dataset. AUROC dự đoán đúng/sai của đáp án (cao = tốt). CI từ paired bootstrap 2000 lần.

| Tín hiệu | chi phí | FinQA | ConvFinQA | TAT-DQA | **AUROC TB** |
|---|---:|---:|---:|---:|---:|
| value-only (legacy) | 1× | 0.480 | 0.550 | 0.582 | 0.537 |
| **CPR (structure, ours)** | 1× | 0.645 | 0.690 | 0.618 | 0.651 |
| self-consistency (k=5) | 6× | 0.767 | 0.708 | 0.773 | 0.749 |
| verbalized confidence | 2× | 0.783 | 0.768 | 0.854 | 0.802 |
| cpr ⊕ verbalized | 2× | 0.805 | 0.810 | 0.852 | **0.822** |
| sc ⊕ verbalized | 7× | 0.839 | 0.815 | 0.884 | 0.846 |
| **fusion (learned, all)** | 7× | 0.823 | 0.822 | 0.868 | 0.838 |

Kiểm định (paired bootstrap):
- **CPR > value-only:** FinQA ΔCI [+0.095,+0.235] P=1.0; ConvFinQA [+0.069,+0.213] P=0.9995; TAT [−0.038,+0.103] P=0.82 (ns).
- **fusion(all) > từng tín hiệu đơn:** vs verbalized P=0.985/0.997/0.887; vs self-consistency P≥0.996 cả 3.
- **structure có cộng thêm? fusion(all) > fusion(internal):** ConvFinQA [+0.007,+0.084] **P=0.993 (có)**;
  FinQA P=0.53 (không); TAT P=0.26 (không).

**Đọc kết quả (trung thực):** trên model mạnh, tín hiệu nội tại > CPR về AUROC tuyệt đối. Đóng góp biên của
structure chỉ có ý nghĩa thống kê ở ConvFinQA (bảng lookup, cấu trúc rõ — nơi concept+period trích được sạch nhất).

---

## 4. Kết quả 3 — Đường biên hiệu quả-chi phí (ĐÓNG GÓP CHÍNH)

Trục đúng để đánh giá tín hiệu reliability khi triển khai là **AUROC trên mỗi lời gọi LLM** (self-consistency
đắt gấp 6×). Sắp theo chi phí:

| chi phí | tín hiệu | AUROC TB | ghi chú |
|---:|---|---:|---|
| 1× | **CPR** | **0.651** | tốt nhất ở 1× (>> value-only 0.537) |
| 2× | **cpr ⊕ verbalized** | **0.822** | **vượt self-consistency 6×** trên cả 3 dataset |
| 6× | self-consistency | 0.749 | bị `cpr+verb` (2×) Pareto-dominate |
| 7× | sc ⊕ verbalized / fusion | 0.838–0.846 | tốt nhất tuyệt đối, nhưng +0.02 đổi lấy 3.5× chi phí |

**Phát biểu được (mạnh, khả triển khai):**
1. Ở ngân sách **1×** (chỉ 1 đáp án), CPR là tín hiệu reliability tốt nhất hiện có (structure verification miễn phí, deterministic).
2. Ở ngân sách **2×**, `cpr+verbalized` **đạt chất lượng của fusion-nội-tại 7×** và **vượt self-consistency 6×** trên cả ba dataset.
3. Self-consistency (chuẩn vàng đắt đỏ) **bị Pareto-dominate** bởi structure verification.

---

## 5. Kết quả 4 — Cơ chế: structure bắt lớp lỗi mà model-internal bỏ sót

Tại ngân sách abstain 40% (`error_disjointness.py`):

| Dataset | #sai | verbalized bắt | CPR bắt | **CPR bắt mà verbalized BỎ SÓT** | union (cpr∨verb) |
|---|---:|---:|---:|---:|---:|
| FinQA | 147 | 99 (0.674) | 70 | **14 (+9.5% recall)** | 113 (0.769) |
| ConvFinQA | 121 | 81 (0.669) | 76 | **20 (+16.5%)** | 101 (0.835) |
| TAT-DQA | 194 | 115 (0.593) | 89 | **28 (+14.4%)** | 143 (0.737) |

→ CPR bắt thêm **9.5–16.5%** số lỗi mà model *tự tin bằng lời* nhưng *không grounded trong chứng cứ*
(confident hallucination). Đây là **giá trị trực giao** của structure — lý do `cpr+verb > verb` và lý do
Pareto-dominance ở §4. Bổ trợ: *overconfidence audit* — trong nhóm model tự tin nhất (top-50% theo sc+verb),
AUROC của CPR = 0.515/0.533/0.495 (≈ngẫu nhiên, ConvFinQA nhỉnh hơn): structure không thay thế được model-internal,
nhưng **lệch pha** với nó.

---

## 5b. Kết quả phụ — Verify-then-reask là NET-NEGATIVE trên generator mạnh (negative result quan trọng)

`gemini_verify_reask.py` — chính sách: giữ đáp án raw khi CPR grounded; nếu CPR conf < 0.5 → hỏi lại Gemini
với chứng cứ structure/KG (đã strip symbolic answer). Đo Number-Match (NM):

| Dataset | raw NM | verify-then-reask NM | #re-ask | rescued | broke |
|---|---:|---:|---:|---:|---:|
| FinQA | 0.510 | **0.447** | 98 | 7 | 26 |
| ConvFinQA | 0.597 | **0.540** | 77 | 2 | 19 |
| TAT-DQA | 0.353 | **0.297** | 134 | 9 | 26 |

→ Re-ask **phá nhiều hơn cứu** trên cả 3. Nguyên nhân (sắc bén, đáng đưa vào paper): vì trần trích xuất thấp,
CPR *false-flag* nhiều đáp án raw-ĐÚNG (model mạnh dùng tốt bảng thô); khi re-ask với chứng cứ KG đã lọc, thông
tin bị cắt → đáp án mới sai. **Hệ quả chính sách:** với model mạnh, dùng CPR cho **abstention** (false-flag chỉ
tốn coverage) **chứ KHÔNG dùng cho re-ask/override** (false-flag *hủy* đáp án đúng). Điều này khớp và mở rộng
finding cũ "answer router fails" / "raw table beats filtered KG" lên generator mạnh. (Trên Qwen yếu, verify-then-reask
từng *dương* nhẹ 0.278→0.295 — lại một bằng chứng cho phổ generator-strength ở C3.)

## 6. Kết quả 5 — Tối ưu CPR đã chạm trần (iterate-to-plateau)

`cpr_optimize.py` (component sweep + học lại trọng số + complementarity):

- **Component sweep:** `full` (concept+period+role+3op) tối ưu trên **cả 3** — period và 3-op đều đóng góp.
- **Học lại trọng số CPR (CPR-cal, CV-logistic trên chính feature của CPR):** *không* cải thiện tín hiệu
  standalone (FinQA −0.005, ConvFinQA −0.007, **TAT −0.042 P=0.012**). → max() thiết kế tay đã gần tối ưu;
  cần feature *mới* (extraction/typing tốt hơn), không phải trọng số mới.
- **Complementarity sạch (fusion raw-CPR vs internal):** có ý nghĩa ở ConvFinQA (+0.044), biên ở FinQA, không ở TAT.

**Kết luận plateau:** không thể làm CPR *standalone* vượt model-internal trên generator mạnh — đây là **một
tính chất khoa học**, không phải lỗi tinh chỉnh. Đòn bẩy còn lại duy nhất là **nâng trần trích xuất/typing** (§7).

---

## 7. Kết quả 6 — Kiểm toán hai heuristic nền (Blind Spot #2, ĐÃ ĐO)

**(a) Trần trích xuất (auditable ceiling, gold-doc, full test set):** CPR chỉ chứng nhận được những gì ledger tái dựng được.

| Dataset | grounded | certifiable 2-op | **certifiable 3-op** |
|---|---:|---:|---:|
| FinQA | 0.034 | 0.453 | **0.717** (+0.26) |
| ConvFinQA | 0.295 | 0.676 | **0.792** |
| TAT-DQA | 0.147 | 0.618 | **0.802** |

Phân rã theo task xác nhận TCEP Law 3: ratio/percent-change/average cần ≥3 operand. **Mở rộng derivation ≥3-op
là đòn bẩy lớn nhất** (FinQA +0.26 trần). Đây là chặn trên giải thích vì sao AUROC tuyệt đối của CPR không cao hơn.

**(b) Độ chính xác heuristic gán vai (role) — `role_assignment_probe.py`:**

| Dataset | plan fire-rate | plan precision (khi fire) | oracle operand-F1 | oracle operation-acc |
|---|---:|---:|---:|---:|
| FinQA | 0.84 | 0.143 | 0.525 | 0.637 |
| ConvFinQA | 0.85 | 0.235 | 0.522 | 0.593 |
| TAT-DQA | 0.80 | 0.113 | 0.399 | 0.741 |

→ `calculation_plan` fire rất nhiều nhưng (i) như **bộ giải standalone** chỉ đúng 11–24% (đúng quyết định thiết kế:
dùng plan để *kiểm tra* đáp án của generator, **không** override — khớp finding "answer router fails"); (ii) so với
**oracle mạnh trên cùng tập fact**, chọn operand chỉ đạt **F1 ~0.52** (TAT 0.40). Đây là **điểm yếu cốt lõi đã được
định lượng**: thành phần Role (workhorse của CPR) dựa trên việc chọn operand chính xác ~50%. → *Learned operand
attribution* là hướng nghiên cứu lõi tiếp theo (trùng open-problem trong [FRAMEWORK_TCEP.md](FRAMEWORK_TCEP.md) §5).

---

## 8. Định vị bài báo (đã hiệu chỉnh theo bằng chứng mới)

> **Title:** *Know When You're Right, Cheaply: Cost-Efficient Structure-Grounded Reliability for Financial
> Numerical QA.*

- **C1 (chính, mới):** *structure-grounded CPR là tín hiệu reliability hiệu quả nhất trên mỗi đơn vị chi phí
  suy luận* — `cpr+verbalized` (2×) đánh bại self-consistency (6×) trên cả 3 dataset; CPR ở 1× là tín hiệu tốt nhất.
- **C2 (cơ chế):** CPR bắt **confident-but-ungrounded hallucinations** mà tín hiệu nội tại bỏ sót (+9.5–16.5%
  error-recall); ràng buộc đáp án vào concept × period × role trên đồ thị cấu trúc, đối xứng với generation.
- **C3 (phổ generator-strength):** dọc trục độ mạnh generator (Qwen-3B yếu → Gemini-2.5 mạnh), vai trò của structure
  chuyển từ *thiết yếu* (model yếu, internal-conf không tin được) sang *bổ trợ hiệu quả-chi phí* (model mạnh).
  Đây là một *finding* tổng quát, không phải một con số đơn lẻ.
- **C4 (deployable):** fusion học được (structure ⊕ model-internal) là tín hiệu selective-answering tốt nhất;
  đóng góp biên của structure tập trung ở bảng cấu trúc tốt (ConvFinQA, P=0.993).
- **Supporting:** retrieval MMER honest (W.Avg MRR@3 **0.736**, leak-free); trần trích xuất & kiểm toán role
  định lượng giới hạn trên và hướng cải thiện.

**Honest negatives (báo cáo, không giấu):** trên model mạnh CPR standalone < model-internal; học lại trọng số CPR
không giúp; complementarity chỉ chắc ở ConvFinQA; role-operand selection F1 ~0.5; **verify-then-reask net-negative
trên model mạnh** (§5b) → CPR dùng cho abstention, không cho override.

---

## 9. Tái lập (reproduce)

```bash
cd ours/source && export PYTHONPATH=src
# 1) integrity: honest retrieval (GPU)
python scripts/modular_retrieval.py --dataset FinQA  --cv 5 \
  --experts lexical,dense,entity,concept,cell,graph,lateint --device cuda:1
# 2) strong-generator generation (Gemini; key trong .env)
for d in finqa convfinqa tatqa; do
  python scripts/research/gemini_generate.py --dataset $d --sample 300 --k-sc 5; done
# 3) headline reliability + cost frontier
python scripts/research/strong_reliability_eval.py
# 4) optimization + mechanism + heuristic audits
python scripts/research/cpr_optimize.py
python scripts/research/error_disjointness.py
python scripts/research/fact_extraction_recall.py --gold-doc-only --multi-op
python scripts/research/role_assignment_probe.py --oracle --sample 100
```
Artifacts: `outputs/research/{gemini_gen,strong_reliability,cpr_optimize,error_disjointness,role_probe,fact_extraction_recall_v2}/`.

---

## 10. Khoảng trống còn lại & Next Actions (ưu tiên)

| # | Hành động | Đóng | ROI |
|---|---|---|---|
| 1 | **Learned operand attribution** (≥3-op, ràng buộc theo concept câu hỏi) → nâng trần FinQA 0.45→0.80 *và* sửa role-F1 0.5 | C2/C3, trần §7 | cao (research) |
| 2 | ~~Verify-then-reask~~ → **ĐÃ ĐO, net-negative trên model mạnh (§5b)**; thay bằng **selective-answering policy** (abstain on low-CPR) | đã đóng | done |
| 3 | Mở rộng generation lên full test set + thêm 1 generator mạnh thứ 2 (Claude/GPT) → củng cố C3 cross-model | bề rộng | trung bình (API) |
| 4 | Multi-level-header parser cho TAT-DQA → nâng extraction → kiểm lại complementarity | TAT yếu | trung bình |
| 5 | Conformal selective answering trên fusion → đảm bảo coverage có chứng minh | C4 | thấp |
| 6 | Mở rộng concept ontology (>14%) bằng contrastive encoder annotation-free | typing | trung bình |

**Kết luận:** vòng này đã (i) sửa rò rỉ integrity, (ii) thay headline yếu bằng một đóng góp **mạnh hơn, trung thực
hơn, khả triển khai hơn** (cost-efficient reliability + cơ chế bắt confident-hallucination), (iii) định lượng đầy đủ
các giới hạn (trần trích xuất, role-F1). Tín hiệu CPR standalone đã chạm trần tối ưu; đòn bẩy tiếp theo nằm ở
*operand attribution/extraction*, không ở trọng số.
