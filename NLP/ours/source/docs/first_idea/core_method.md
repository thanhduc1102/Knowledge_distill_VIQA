# LEDGER: Disentangled Fact-Level Retrieval for Financial Documents

> **Target:** AAAI Main Track. **Benchmark:** T²-RAGBench (EACL 2026) — FinQA, ConvFinQA, TAT-DQA.
> **Bản này thay thế `core_method.md` cũ và bản v1 (6 contribution dàn trải).** Đã gom còn **1 research gap → 2 contribution lồng nhau**, và xử lý trực tiếp `review_1.md`, `review_2.md`. Changelog ở §11.

---

## 0. Một câu chuyện, một gap, hai contribution

**Research gap (thống nhất).** Trong truy hồi tài liệu tài chính (text + bảng), tín hiệu quyết định *relevance* là **mịn** — bộ ba *(thực thể, kỳ báo cáo, đại lượng/độ lớn)* — nhưng (i) bị **ngữ nghĩa dùng-chung nhấn chìm** trong một vector tổng (*granularity dilemma* [Xu et al., Findings EMNLP 2025]) và (ii) bị **tokenization phá huỷ độ lớn số** (*numeracy gap* [Findings EACL 2026]). Hệ quả đo được trên T²-RAGBench: tài liệu *sai* được chấm cao hơn tài liệu *đúng* ở 59.5% truy vấn; khoảng cách tương đồng (đúng−sai) **âm** (−0.027); same-company tương đồng **2.82×** inter-company. **Chưa phương pháp nào biến các khóa mịn này thành *first-class* đồng thời ở cả inference lẫn training** — dense giữ single-vector, BM25/ColBERT coi số như token, *Numbers Matter!* đòi điều kiện số tường minh trong query.

**Thesis.** Đưa bộ khóa mịn lên *first-class* ở **hai cấp**:

- **C1 — Disentangled fact-grounded retrieval với định tuyến entity–time (inference-time).**
  Tách biểu diễn thành *concept ⟂ magnitude ⟂ khóa rời rạc*, đối sánh ở **cấp fact** bằng **gated typed late interaction**, và — mấu chốt để khả thi & không "giòn" — **hợp nhất lọc rời rạc vào chính chỉ mục vector** (metadata-routed masked search) kèm **soft fallback**.
  *Giải:* granularity dilemma + numeracy gap *tại thời điểm truy hồi*, một cách **scalable và robust**.

- **C2 — Ledger-grounded provably-true contrastive learning (training-time).**
  Nhiễu loạn có cấu trúc trên ledger của tài liệu vàng để sinh hard negative **chắc chắn đúng (provably-true)**: *period / entity / **metric** / scale / value-identity swap*. Mỗi loại huấn luyện đúng một kênh/cổng của C1.
  *Giải:* hard-negative mining cho corpus near-duplicate (theo thời gian/thực thể) **không bị nhiễu false-negative** — điều mà mining cổ điển phải lọc tốn kém.

**Vì sao hai contribution gắn chặt (yêu cầu cốt lõi).**
C2 định nghĩa negative *trên chính fact ledger mà C1 dùng*; và **mỗi loại negative của C2 cấp gradient cho đúng một thành phần của C1**:

| Negative (C2) | Huấn luyện thành phần nào của C1 |
|---|---|
| metric-swap | kênh concept $\sigma_m$ |
| scale-break | kênh magnitude $\rho$ |
| entity-swap / period-swap | cổng $\gamma_e,\gamma_t$ |
| value-identity break | kênh magnitude + cầu nối sang verifier (§6) |

Ngược lại, **cổng tách-kênh của C1 chính là thứ khiến negative của C2 vừa cực khó (gần như trùng lặp từ vựng) vừa học được (cổng cung cấp trục phân biệt)**. Tháo một cái, cái còn lại mất ý nghĩa. Đây là một phương pháp *thống nhất*, không phải hai mẹo rời rạc.

Các phần *không* phải contribution lõi (theo review): phân tích lý thuyết = **motivation** (§2), sinh có kiểm chứng = **extension ngắn** (§6), dữ liệu tiếng Việt = **case study** trong thực nghiệm (§7).

---

## 1. Biểu diễn nền: Financial Fact Ledger (substrate dùng chung)

Cả C1 và C2 đứng trên một biểu diễn trung gian symbolic.

**Định nghĩa (Fact).** $f=(m,e,t,v,u)$: concept $m$ (line item canonical, embedding $\mathbf c_f\in\mathbb R^{d}$), entity $e$, period $t$, value $v\in\mathbb R$, unit/scale $u$ → magnitude $\mu_f=\operatorname{sign}(v)\log_{10}(1+|v|/s_u)$.
**Ledger tài liệu** $\mathcal L(d)=\{f_i\}$: mỗi cell bảng → một fact với $m$=nhãn **hàng**, $t$=nhãn **cột** (sửa lỗi hướng-bảng của code cũ), $e$=thực thể; cộng fact số trích từ narrative.
**Query intent** $\mathcal Q(q)=\{(\tilde m_j,\mathrm{op}_j,\tilde v_j)\}_{j=1}^k\cup(\tilde e,\tilde t)$.

> Chất lượng ledger là *điều kiện sống còn* (cả hai review nhấn mạnh). Cách xây, scope, đo F1, fallback: §3.4 và §7.

---

## 2. Motivation: vì sao single-vector *nhất thiết* thua (đã hạ cấp từ "định lý" → phân tích có đo đạc)

> **Phản hồi review 1.1 & 2.§2:** không tuyên bố đây là đóng góp lý thuyết phổ quát. Đây là *mô hình giải thích lý tưởng hoá* (assumption minh bạch) **đi kèm thí nghiệm xác nhận**.

**Mô hình cộng tính (idealized, A1).** Trong một lân cận near-duplicate, ta *xấp xỉ* embedding bằng $\mathbf h_d\approx\mathbf u_{\text{top}}(\text{sector,metric-vocab})+\mathbf u_{\text{id}}(e,t)+\boldsymbol\varepsilon$. **A1 là xấp xỉ bậc nhất, không phải tính chất đúng tuyệt đối** của encoder phi tuyến — ta nêu rõ và chỉ dùng nó để rút trực giác; tính đúng đắn cuối cùng dựa vào *đo đạc* bên dưới.

**Quan sát.** Với $d^\star=(e,t)$ và near-duplicate $d^-=(e,t')$: hiệu điểm dense
$\Delta=\langle\mathbf u_{\text{id}}(\tilde e,\tilde t),\,\mathbf u_{\text{id}}(e,t)-\mathbf u_{\text{id}}(e,t')\rangle+\text{nhiễu}$, vì phần $\mathbf u_{\text{top}}$ triệt tiêu (cùng công ty/loại báo cáo). Nếu $\Sigma_{\text{top}}\!\gg\!\Sigma_{\text{id}}$ thì $\mathrm{SNR}\!\lesssim\!\Sigma_{\text{id}}/\Sigma_{\text{top}}\!\to\!0$ → bộ phân biệt single-vector tiệm cận ngẫu nhiên (khớp gap âm trong EDA).

**Thí nghiệm xác nhận (bắt buộc trong paper — phản hồi review 1.1, 2.§2):**
1. **Linear probe:** từ embedding tài liệu, probe dự đoán *(entity, year)* vs *(sector/topic)*. Dự đoán: acc(topic) cao ≫ acc(entity,year) → bằng chứng *trực tiếp* cho $\Sigma_{\text{top}}\gg\Sigma_{\text{id}}$ (không chỉ gián tiếp qua 2.82×).
2. **Ước lượng phương sai theo trục:** chiếu embedding lên không gian topic vs identity (dùng nhãn sector vs (company,year)), đo $\widehat\Sigma_{\text{top}},\widehat\Sigma_{\text{id}}$ thực trên T²-RAGBench.

Kết luận §2: ta gọi đây là **"diagnostic analysis"**, dùng để *thúc đẩy thiết kế cổng nhân ở §3.2*, không phải để khoe định lý. Cổng nhân biến tín hiệu phân biệt từ *số hạng cộng (bị nhấn chìm)* thành *thừa số (không thể bị nhấn)* — đó là nội dung kỹ thuật thực sự.

---

## 3. Contribution 1 — Disentangled fact-grounded retrieval + entity–time routing

### 3.1. Tách kênh biểu diễn (giải numeracy gap, granularity dilemma ở mức biểu diễn)
- **Concept** $\mathbf c_f=\mathrm{Enc}_\theta(m_f)$: encode *chỉ chuỗi concept ngắn* (không nhồi cả bảng/số thô) → tránh numeracy gap ở input.
- **Magnitude** $\mu_f$ giữ **tách biệt** dạng scalar (không học embedding số nặng — tối giản tham số, phản hồi "anti-over-parameterization").
- **Khóa rời rạc** $e_f,t_f$ chuẩn hoá.

### 3.2. Gated typed late interaction (lõi đối sánh)
$$
\boxed{\,S(q,d)=\sum_{j=1}^{k}\max_{f\in\mathcal L(d)}\big[\sigma_m(\tilde{\mathbf c}_j,\mathbf c_f)\cdot\gamma_e(\tilde e,e_f)\cdot\gamma_t(\tilde t,t_f)\cdot\rho_j(\mu_f)\big]\,}
$$
$\sigma_m=\mathrm{ReLU}(\cos)^{1/\tau_m}$; $\gamma_t(\tilde t,t)=\exp(-|\tilde t-t|/\sigma_t)$ (**kernel mềm**, không veto 0/1 — quan trọng cho §3.3); $\gamma_e$ khớp canonical/fuzzy; $\rho_j\equiv1$ nếu query không nêu số. Multi-vector ở cấp fact ⇒ 1 doc phục vụ nhiều câu hỏi (giải context-sharing). MaxSim ⇒ mỗi slot "bầu" cho fact ủng hộ mạnh nhất.

### 3.3. Hợp nhất lọc rời rạc vào chỉ mục — *Masked Vector Search* (giải two-stage paradox & brittleness)

> **Đây là phần tái thiết kế theo review 2 (§1.1, §3).** Bỏ kiến trúc "FAISS thô → lọc lại" (gây nghịch lý: concept "Total Equity" trả về hàng trăm nghìn fact mọi công ty/năm).

**Single-Stage Filtered ANN.** Metadata trở thành **mặt nạ** can thiệp trực tiếp vào duyệt đồ thị HNSW:
1. **Fuzzy mask (Roaring Bitmap):** inverted index trả tập fact-ID của $e\!\simeq\!\tilde e$ **giao** với năm $\in[\tilde t-1,\tilde t,\tilde t+1]$ (nới lỏng để giữ cơ chế soft-veto của $\gamma_t$, không lọc cứng).
2. **Masked HNSW traversal:** tại mỗi node, kiểm tra bitset trước khi tính cosine; node ngoài mask bị bỏ qua (vẫn đi tiếp theo cạnh). HNSW *buộc* trả top-K concept **trong đúng vùng metadata**.
3. **Late interaction trên K ứng viên đã sạch:** áp dụng công thức §3.2; $\gamma_t$ giờ làm đúng việc (phạt nhẹ năm lân cận, không phải lọc thô).

*Độ phức tạp:* không tuyên bố $\mathcal O(1)$ (review 2 đã chỉnh: ANN bị chặn $\Omega(\log N)$); mục tiêu là **bounded latency** — chi phí late-interaction chỉ chạy trên $K$ (vd 100) ứng viên, không bùng nổ tuyến tính theo corpus. Routing bằng metadata = "Metadata định tuyến, Concept định khoảng cách, Magnitude/Time định cỡ".

**Soft fallback (chống "giòn" — review 2 §1.2, §3.2):** đếm cardinality bitmap ($\mathcal O(1)$). Nếu `count < ngưỡng` (parser sai/quá gắt) → tự nới: bỏ mask thực thể, giữ mask năm; nếu vẫn rỗng → **flatten** toàn bộ mask, lui về truy hồi concept thuần, để cổng $\gamma$ ở late-interaction tự phạt. ⇒ Lỗi parser gây **suy giảm từ từ**, không sụp về Recall 0. (Đo bằng ablation tiêm nhiễu parser, §7.)

### 3.4. Fact extraction: scope, fallback, đo lường (giải review 1.2, 2.§1.3)
- **Scope minh bạch:** LEDGER nhắm bảng có **cấu trúc lưới cơ bản**; với multi-level header/merged cells, dùng (a) một bộ chuẩn hoá header phân cấp đơn giản, (b) **fallback**: nếu parse thất bại → coi mỗi dòng số là fact text-level với concept = câu mô tả gần nhất.
- **Canonicalization:** từ điển đồng nghĩa + viết tắt (giải abbreviation mismatch) *kết hợp* alignment bằng embedding (không thuần rule).
- **Đo lường bắt buộc:** (i) **Fact-extraction F1** trên tập dev gán tay; (ii) **Extraction-to-Retrieval correlation**; (iii) **ablation oracle-fact vs auto-fact** để định upper-bound & độ nhạy. Lập luận robustness: MaxSim chỉ cần *fact đúng tồn tại*; nhưng ta thừa nhận nếu nó *không được trích* thì mất doc — nên fallback text-level là lưới an toàn.

---

## 4. Contribution 2 — Ledger-grounded provably-true contrastive learning

### 4.1. Negative cấu trúc, sinh trên ledger của $d^+$
- **period-swap** $d^-_t$: đổi $t\to t'$ — near-duplicate khác kỳ (thủ phạm EDA #1).
- **entity-swap** $d^-_e$: đổi $e\to e'$.
- **metric-swap** $d^-_m$ *(thêm theo review 1.5)*: **cùng company, cùng năm, đổi giá trị giữa hai metric** (vd gán giá trị của *Revenue* cho *Total Equity*) → buộc $\sigma_m$ học tương đồng concept thật, chống "học lười sai-năm".
- **scale-break** $d^-_s$: nhân cột ×$10^3$ → kênh magnitude.
- **value-identity break** $d^-_v$: phá giá trị tham gia đồng nhất thức kế toán → cầu nối verifier (§6).

### 4.2. Mục tiêu & ánh xạ gradient (chính là chỗ C1–C2 lồng nhau)
$$
\mathcal L_{\text{ret}}=-\log\frac{e^{S(q,d^+)/\tau}}{e^{S(q,d^+)/\tau}+\sum_{d^-\in\mathcal N(q)}e^{S(q,d^-)/\tau}},\quad \mathcal N(q)=\text{in-batch}\cup\{d^-_t,d^-_e,d^-_m,d^-_s\}.
$$
Vì mỗi $d^-$ chỉ khác $d^+$ ở **một** khía cạnh, gradient InfoNCE dồn vào **đúng** thành phần tương ứng của $S$ (bảng ở §0) → mỗi kênh/cổng được giám sát riêng biệt, sạch.

### 4.3. Tính chất provably-true (đóng góp được cả 2 review đánh giá cao nhất)
**Mệnh đề.** Gọi $f^\star\in\mathcal L(d^+)$ là fact trả lời $q$. Phép nhiễu loạn $\pi$ thay đổi *chính khóa/giá trị* của $f^\star$ (năm, thực thể, concept, hoặc value), nên đáp án vàng không còn truy được từ $\pi(d^+)$; do đó $\pi(d^+)$ **chắc chắn không liên quan** đến $q$. ⇒ $\mathcal N(q)$ **không chứa false negative theo cấu trúc**. $\square$

**Vì sao đây là gap thật:** mining hard-negative cổ điển (RocketQA, NV-Retriever) phải dùng cross-encoder/LLM lọc false negative (tới ~70% top-similar trên MS-MARCO là positive thật). Trên corpus tài chính, same-company-other-year *vừa cực hard vừa rủi ro false-negative cao nhất* — đúng nơi mining cổ điển nguy hiểm nhất, thì perturbation-trên-ledger lại **đảm bảo đúng**. Ta tổng quát hoá thành nguyên tắc: *khi có một "view" symbolic của tài liệu, hãy perturb phần-tử-mang-đáp-án để tổng hợp hard negative true-by-construction* — áp dụng cho mọi corpus versioned/structured, không riêng tài chính.

---

## 5. Mối quan hệ C1 ⇄ C2 (làm rõ, vì đây là tiêu chí bạn nhấn mạnh)
1. **Chung substrate:** negative của C2 là phép biến đổi trên ledger §1 mà C1 chấm điểm.
2. **Bổ trợ gradient:** mỗi negative type ↦ một thành phần của $S$ (bảng §0). Không có cổng tách-kênh (C1) thì không có "trục" để negative (C2) dạy; không có negative theo từng khía cạnh (C2) thì cổng (C1) không được calibrate.
3. **Cùng giải một gap:** C1 đưa khóa mịn thành first-class *lúc suy luận*; C2 đưa khóa mịn thành first-class *lúc học*. Hai mặt của cùng một nguyên lý disentanglement.

---

## 6. Extension (không phải contribution lõi): tái dùng ledger cho sinh có kiểm chứng
Ngắn gọn, để cho thấy *tính tái dụng của ledger*: generator nhận top-K + fact đã trích; **verifier** dùng *cùng* ledger kiểm tra (i) value grounding, (ii) đồng nhất thức kế toán $|f_c(\mathbf v)|>\tau_c$ *khi áp dụng được*, (iii) khớp phép toán; **tiered DPO** (accuracy ưu tiên tuyệt đối) tinh chỉnh. Đây là nơi đồng nhất thức kế toán (bản cũ) thực sự hữu ích — như *verifier*, không phải tín hiệu xếp hạng. Trình bày 1 mục, không chiếm trọng tâm.

---

## 7. Thực nghiệm

**Dữ liệu:** T²-RAGBench (FinQA 8,281 / ConvFinQA 3,458 / TAT-DQA 11,349; 7,317 context, unknown-context). **VAS tiếng Việt** = *case study* nhỏ về chuyển giao ngôn ngữ của concept canonicalization (không phải contribution riêng — review 1.6).

**Baselines (sắc theo review 1.3):** BM25; **Hybrid BM25** (SOTA hiện tại); dense (BGE-M3, e5-large, text-embedding-3-large); **ColBERTv2**; reranker; HyDE; **Numbers Matter!**; và **baseline mạnh bắt buộc: dense + hard metadata-filter (company/year) + rerank** — để chứng minh *cổng mềm + masked-soft-fallback thắng filter cứng khi parse không hoàn hảo*.

**Metrics:** MRR@3, Recall@1/3/5, nDCG@3; (gen) NumAcc, Constraint-Violation Rate; **latency/throughput** so ColBERT & hybrid (review 1.4, 2).

**Thí nghiệm chẩn đoán (kể "câu chuyện thắng" — review 1.3, 2):**
- **Probe + variance** (§2) xác nhận cơ chế.
- **Similarity-gap flip:** đo (đúng−sai) kỳ vọng lật từ **âm → dương**.
- **Phân khúc khó:** *same-company-multi-year* và *high-numerical-density* — nơi gain phải đậm nhất (chuẩn bị tinh thần gain tổng khiêm tốn, thắng ở phân khúc).
- **Parser-noise ablation:** tiêm nhiễu vào intent parser, đo độ suy giảm (chứng minh soft fallback chống brittleness).
- **Fact-extraction:** F1, oracle-vs-auto, extraction-to-retrieval correlation.

**Ablation (mỗi dòng ⇄ một thành phần):** − cổng E–T; − tách kênh magnitude; − fact-level→doc single-vector; − masked-index (về two-stage); − soft fallback; − từng loại negative (đặc biệt − metric-swap để lộ "học lười"); − reasoning-aware.

---

## 8. Định vị so với prior work
| | Khóa E–T first-class | Số theo *độ lớn* | Negative true-by-construction | Scalable filtered index | Retriever (không phải reader) |
|---|---|---|---|---|---|
| Dense (BGE/e5/te-3) | ✗ | ✗ | ✗ | n/a | ✓ |
| Hybrid BM25 (SOTA) | ✗ (token) | ✗ | ✗ | ✓ | ✓ |
| ColBERTv2 | ✗ | ✗ | ✗ | ✓ | ✓ |
| Numbers Matter! (EMNLP'24) | ✗ | ✓ (điều kiện *trong query*) | ✗ | một phần | ✓ |
| SoarGraph (WWW'23) | – | ✓ | ✗ | ✗ | ✗ (reader) |
| **LEDGER (ours)** | **✓ (cổng + routing)** | **✓ (kênh tách)** | **✓** | **✓ (masked HNSW)** | **✓** |

---

## 9. Tự phản biện & giới hạn (thẳng thắn theo cả 2 review)
- **Theory là idealized:** A1 cộng tính chỉ là xấp xỉ; ta chống đỡ bằng probe/variance thực nghiệm, gọi đúng tên "diagnostic", không phải định lý phổ quát.
- **Phụ thuộc fact extraction & parser:** thừa nhận; giảm thiểu bằng *scope rõ*, *fallback text-level*, *soft fallback ở index*, và đo *F1/correlation/parser-noise*.
- **Gain tổng có thể khiêm tốn** so với hybrid BM25 (rất mạnh trên text): chiến lược kể chuyện qua phân khúc khó + cơ chế (gap-flip), không chỉ con số tổng.
- **Chi phí:** bounded, không $\mathcal O(1)$; báo latency thật.

---

## 10. Lộ trình
1. Probe/variance §2 (biến lý thuyết thành bằng chứng) — *làm sớm, rẻ, thuyết phục reviewer*.
2. Implement ledger extraction (đúng hướng-bảng) + đo F1 + fallback.
3. C1: masked HNSW + gated late interaction; chạy FinQA; đo **gap-flip**.
4. C2: 5 loại negative (gồm metric-swap); ablation từng loại.
5. Parser-noise + oracle-fact ablations; latency.
6. Extension §6 + VAS case study.

---

## 11. Changelog so với bản v1 (đáp ứng review)
- **6→2 contribution** lồng nhau, có bảng ánh xạ negative↦thành phần (review: "không trọng tâm").
- **Two-stage paradox** → masked single-stage filtered ANN (review 2 §1.1, §3).
- **Brittleness** → soft fallback + cổng mềm + ablation nhiễu parser (review 2 §1.2).
- **Theory** hạ cấp "định lý"→"diagnostic" + thêm probe/variance (review 1.1, 2 §2).
- **Fact extraction** scope + F1 + oracle-vs-auto + fallback (review 1.2, 2 §1.3).
- **+ metric-swap negative** (review 1.5).
- **metadata-filter+dense** thành baseline bắt buộc + phân tích phân khúc (review 1.3).
- **VAS → case study; generation → extension** (review 1.6).
- **Title rút gọn**; bỏ $\mathcal O(1)$, dùng "bounded latency" (review 2).

## Tham chiếu chính
T²-RAGBench (EACL'26) · Granularity Dilemma (Findings EMNLP'25) · Numeracy Gap (Findings EACL'26) · Numbers Matter! (Findings EMNLP'24) · ColBERT/ColBERTv2 · SoarGraph (WWW'23) · RocketQA, NV-Retriever (hard-negative/false-negative) · xVal · Information Bottleneck · FinQA/TAT-QA/ConvFinQA · DPO.
