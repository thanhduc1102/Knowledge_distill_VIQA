# ĐỊNH HƯỚNG NGHIÊN CỨU AAAI-27
## LEDGER: Fact-Level Indexing cho Truy hồi & Suy luận Tài chính Toàn trình

> Tài liệu này (i) đánh giá lại **trung thực** hiện trạng hệ thống, tách phần "thật" khỏi phần "khai thác artifact"; (ii) khảo sát **literature SOTA 2025–2026** (đã truy cập thực tế) để biết khoảng trống nào còn mở; (iii) đề xuất **5 đóng góp** gắn kết quanh **một** cấu trúc tri thức duy nhất — *Financial Fact Graph / Fact Ledger* — phủ toàn trình retrieval → huấn luyện → sinh → kiểm chứng; (iv) định vị từng đóng góp so với từng đối thủ; (v) kế hoạch thực nghiệm và lộ trình.
>
> **Nguyên tắc chủ đạo:** chúng ta KHÔNG lấy "metadata để vượt SOTA" làm đóng góp — vì nó đã được publish và bản thân nó *trivialize* benchmark. Chúng ta lấy **phần khó còn lại sau khi đã dùng metadata** làm bài toán nghiên cứu thật.

---

## CẬP NHẬT BẮT BUỘC 2026-06-19 — C4 KHÔNG CÒN LÀ GRPO Ở GIAI ĐOẠN HIỆN TẠI

Theo yêu cầu hiện tại, **không huấn luyện / RL generator ở pha này**. Mọi đoạn cũ xem GRPO, answer-only RLVR, hoặc process-reward training là bước thực thi chính được hạ xuống **future optional**. Đóng góp C4 hiện tại được định nghĩa lại là:

> **Ledger-Grounded Inference-Time Generation Support:** KG/Fact Ledger nhận top-3 retrieval nhiễu, chọn/focus tài liệu đúng, trích toán hạng, tính toán ký hiệu khi đủ tự tin, đưa provenance fact→cell/question, phát hiện xung đột và render context để generator Qwen frozen chỉ kiểm tra/copy kết quả thay vì tự làm toàn bộ extraction + arithmetic + explanation.

Kết quả đã chạy trực tiếp sau khi triển khai `rules4`:

| Thành phần | FinQA | ConvFinQA | TAT-DQA |
|---|---:|---:|---:|
| Original top-1 trong top-3 | 0.6417 | 0.7279 | 0.3260 |
| KG arbitration top-1 | 0.6469 | 0.7287 | 0.3628 |
| Best KG policy top-1 | **0.6600** (`margin≥0.15`) | **0.7420** (`margin≥0.25`) | **0.3820** (`rankprior=2.0`) |
| Symbolic coverage, high-confidence | 0.5937 | 0.5604 | 0.6154 |
| Symbolic NM when available | 0.2349 | 0.3782 | 0.1491 |

End-to-end frozen Qwen3-4B sample:

| Dataset | rules2 | rules3 | rules4 |
|---|---:|---:|---:|
| FinQA, 100 mẫu | 0.21 | 0.21 | **0.22** |
| TAT-DQA, 100 mẫu | 0.18 | 0.19 | **0.20** |
| TAT grounded fraction | 0.32 | 0.36 | **0.43** |

Kết luận: hướng hiện tại không còn là "metadata để vượt SOTA" và cũng không phải "RL generator"; nó là **typed financial fact graph như một lớp điều phối, tính toán, kiểm chứng và giải thích tại inference**. Đây là câu chuyện phù hợp nhất với ba yêu cầu KG: focus top-3, giảm tải LLM, và minh bạch.

---

## 1. Luận đề (Thesis)

Trên tài liệu tài chính, *độ liên quan của một tài liệu* và *tính đúng của một bước suy luận* đều được quyết định ở **cấp từng dữ kiện cụ thể** (concept × entity × period × value), KHÔNG phải ở cấp toàn tài liệu. Một biểu diễn **đồ thị dữ kiện có kiểu (typed Financial Fact Graph)** vì thế có thể đồng thời đóng bốn vai trò mà các hệ thống hiện nay tách rời:

1. **Biểu diễn truy hồi** — gated late-interaction ở cấp fact (C2).
2. **Nguồn sinh mẫu âm khó** có bảo chứng hợp lệ (C3).
3. **Bộ điều phối + tính toán + kiểm chứng tất định tại inference** cho generator frozen (C4); RL chỉ là hướng mở sau này.
4. **Bằng chứng + provenance** cho sinh có chọn lọc, minh bạch (C5).

Và một chẩn đoán (C1) chứng minh **vì sao** các cách tiếp cận cấp-tài-liệu (kể cả metadata) là không đủ. Sự **thống nhất trên một substrate ký hiệu duy nhất, xuyên suốt retrieval→training→generation→verification** chính là điểm mới hệ thống mà không đối thủ đơn lẻ nào đạt được.

---

## 2. Đánh giá lại trung thực hiện trạng (cái gì THẬT, cái gì là ARTIFACT)

### 2.1 Kết quả hiện có
Hệ thống (MMER + ledger + company-pool + pool-local IDF) báo cáo: TAT-DQA **0.704**, FinQA **0.901**, ConvFinQA **0.909** MRR@3 — vượt/xấp xỉ leaderboard #1 (GPT-5.4 + Metadata-BM25: 67.9 / 90.3 / 84.5) mà **không dùng LLM ở pha retrieval**.

### 2.2 Vì sao phần lớn con số này KHÔNG phải đóng góp nghiên cứu
- **Company-scoping cho recall = 1.0 một cách tầm thường.** Câu hỏi T²-RAGBench là *context-independent*, 100% chứa prefix `company:`, và gold *luôn* cùng company. Lọc theo company → pool ~5–47 chunk, gold chắc chắn trong đó. Đây là **khai thác hợp lệ nhưng không mới**: đã được publish dưới tên *Metadata-driven RAG* (arXiv 2510.24402) — chính là phương pháp leaderboard #1.
- **Difficulty decomposition tự tố cáo artifact:** độ khó *content-only* (mask company khỏi câu hỏi) chỉ **0.31 / 0.53 / 0.51**; company-scoping thổi lên 0.68/0.82/0.82. Khoảng cách = độ lớn artifact.
- **Pool-local IDF (`loclex`)** là kỹ thuật đúng nhưng *first-order* — bản thân nó là "BM25 tính lại IDF trong cụm". Là engineering tốt, **chưa đủ tầm một đóng góp AAAI** nếu đứng một mình.

### 2.3 Phần THẬT đáng giữ
- **Reframe bài toán:** "within-entity-cluster disambiguation" — chọn đúng *chunk/fact* trong ~5–47 tài liệu **cùng công ty**, nơi metadata vô dụng và embedding dense thất bại. Đây mới là bài toán khó, tổng quát.
- **Phát hiện phản trực giác:** `loclex` THUA BM25 ở full-corpus (0.345 < 0.418) nhưng THẮNG trong company-pool (0.681 > 0.565); dense/ColBERT thua ở **cả hai** regime. → *Conditional salience chỉ mở khoá khi đã cụm hoá; và neural toàn cục không bắt được phân biệt within-filing trên bảng số.*
- **Fact Ledger** (concept-entity-period-value, orientation-aware, có accounting identities) — substrate đã có, đúng hướng.

> **Kết luận mục 2:** Ta giữ *reframe* và *substrate*, vứt bỏ tuyên bố "vượt SOTA nhờ metadata". Đóng góp phải nằm ở **phần residual khó** và ở **sự thống nhất toàn trình**.

---

## 3. Khảo sát SOTA 2025–2026 — khoảng trống còn mở ở đâu?

### 3.1 Cái đã bị "chiếm" (không lấy làm core novelty)

| Hướng | Bài tiêu biểu (đã truy cập) | Đã làm gì | Hệ quả cho ta |
|---|---|---|---|
| Metadata-aware retrieval | **Metadata-driven RAG for Financial QA** — arXiv [2510.24402](https://arxiv.org/pdf/2510.24402) | Tiêm metadata (company/date/form) → QA 50–60%→72–75% | Metadata KHÔNG còn là đóng góp |
| Within-document disambiguation | **Decomposing Retrieval Failures in RAG for Long-Document Financial QA** — arXiv [2602.17981](https://arxiv.org/abs/2602.17981) | Decompose doc→page→chunk; *learned page scorer* (BGE-M3 fine-tune). Chỉ FinanceBench (150 Q). | Khung within-doc đã có; nhưng **chỉ single-doc**, để mở: cross-filing, RL, structure-metadata |
| KG cho numeric reasoning tài chính | **Structure First, Reason Next** — arXiv [2601.07754](https://arxiv.org/abs/2601.07754) | KG schema rút từ doc + Llama-3.1-8B, +12% execution acc trên FinQA (reading-comprehension) | "KG giúp LLM suy luận" đã có; nhưng **frozen reasoning, oracle context, không RL, không retrieval** |
| Symbolic routing cho số học | **HierFinRAG** (MDPI Informatics 13(2):30, 2026) — [link](https://www.mdpi.com/2227-9709/13/2/30) | TTGNN (table-text GNN) + *Symbolic–Neural Fusion* route giữa generator và calculator. SOTA >7% (FinQA, FinanceBench) | "Gọi calculator cho số học" đã có; khoảng trống còn lại là **KG arbitration trên top-3 nhiễu + provenance fact-level + bridge end-to-end cho generator frozen** |
| Hierarchical evidence curation | **Hierarchical Retrieval w/ Evidence Curation** — arXiv [2505.20368](https://arxiv.org/pdf/2505.20368) | Truy hồi phân cấp + lọc bằng chứng trên tài liệu chuẩn hoá | Củng cố động lực, không trùng cơ chế |
| KG nhẹ > GNN nặng | **SubgraphRAG**; **Less is More: Denoising KGs for RAG** ([2510.14271](https://arxiv.org/html/2510.14271v1)); **When to use Graphs in RAG** ([2506.05690](https://arxiv.org/html/2506.05690v3)) | MLP + đặc trưng đồ thị nhẹ ≥ GNN phức tạp; KG generic nhiều nhiễu | **Bằng chứng phản đối KG entity-relation generic** → ta dùng *typed fact graph* |

### 3.2 Các bài "động lực" (cite để biện minh, không trùng đóng góp)
- **Dense Retrievers Can Fail on Simple Queries: The Granularity Dilemma** — arXiv [2506.08592](https://arxiv.org/pdf/2506.08592) (EMNLP'25 Findings): một vector không thể vừa giữ chủ đề vừa giữ chi tiết phân biệt → biện minh **biểu diễn cấp-fact đa vector**.
- **Revealing the Numeracy Gap** — arXiv [2509.05691](https://arxiv.org/pdf/2509.05691) (EACL'26 Findings): embedding không phân biệt độ lớn số → biện minh **luồng magnitude tách riêng**.
- **Numbers Matter! Quantity-aware Retrieval** (EMNLP'24): nhưng giả định *điều kiện số nằm trong câu hỏi*; bài ta thì số nằm trong *tài liệu* → khác biệt.
- **xVal** (continuous numeric tokenization): cảm hứng mã hoá magnitude.

### 3.3 Khoảng trống thật sự còn mở (không ai làm)
1. **Một substrate ký hiệu duy nhất** dùng *đồng thời* cho retrieval-scoring, sinh-negative, symbolic planning, generator grounding, verification và provenance. (Mỗi đối thủ chỉ làm 1 lát.)
2. **Inference-time process verification + symbolic answer planning**, ký hiệu, miễn-gán-nhãn, chuyên numeric, lấy ngay từ fact ledger:
   - vs **answer-only LLM**: giảm việc LLM phải tự tìm số, tự tính toán và tự giải thích.
   - vs **learned PRM / RLVR**: không cần nhãn bước, không cần huấn luyện generator trong giai đoạn hiện tại.
   - vs **text-faithfulness reward** (*Beyond Correctness* 2025; *CRAFT* arXiv [2602.01348](https://arxiv.org/pdf/2602.01348)): kiểm tra *văn bản* bám nguồn nhưng không kiểm trực tiếp *giá trị số*, *period*, *unit* và *phép tính*.
   → **Đây là khoảng trống lớn nhất cho mục tiêu hiện tại: KG giảm tải generator ngay tại inference.**
3. **Within-filing numeric disambiguation tổng quát hoá** lên *entity-clustered numeric corpora* (finance/legal/medical), kèm *artifact-control* mà 2602.17981 thiếu.
4. **Channel-aligned hard negatives có bảo chứng hợp lệ** (tránh false-negative filtering của RocketQA/NV-Retriever) — recipe tổng quát cho corpus *có phiên bản/cấu trúc*.

---

## 4. Phán quyết về KNOWLEDGE GRAPH (trả lời trực tiếp yêu cầu nghiên cứu KG)

**Không dùng KG entity-relation generic kiểu GraphRAG/HippoRAG/KAG/LightRAG.** Lý do, có bằng chứng:
- Bảng tài chính là *numeric-dense*; trích entity-relation generic rất giòn (GSR gốc của ta đã thất bại đúng vì điều này — template khớp sai chiều, CS=1.0 cho mọi doc).
- Literature 2025 cho thấy **KG nhẹ + đặc trưng ≥ GNN nặng** (SubgraphRAG), **KG generic nhiều nhiễu cần denoise** (Less is More), và **graph không phải lúc nào cũng nên dùng** (When to use Graphs in RAG).
- GraphRAG community-summary tối ưu cho *global sensemaking*, không cho *point-fact numeric lookup* — sai bài toán.

**Dùng một *typed Financial Fact Graph* (chính là Fact Ledger nâng cấp):**
- **Nút** = fact `(m concept, e entity, t period, v value, u unit/scale)` + vector concept `c_f`, magnitude `μ_f`.
- **Cạnh** = (i) *accounting identities* (Revenue−COGS=GrossProfit, Assets=Liab+Equity, …) — vai trò **kiểm chứng**, KHÔNG phải tín hiệu xếp hạng (đây là bài học từ GSR: identities làm verifier, đừng làm ranking); (ii) *temporal links* cùng concept khác kỳ (phục vụ câu hỏi % thay đổi); (iii) *co-statement links* (cùng báo cáo).
- Đây vừa là KG (có schema, có quan hệ, có suy luận đồ thị nhẹ), vừa né được mọi điểm yếu của KG generic, vừa phục vụ **cả ba mục tiêu KG bạn nêu**: (1) giúp retrieval chọn đúng trong top-3; (2) giảm tải LLM bằng cách *pre-extract* facts; (3) minh bạch nhờ provenance fact→cell.

---

## 5. NĂM ĐÓNG GÓP (gắn kết quanh Fact Graph)

### C1 — Artifact-Controlled Difficulty Decomposition *(đóng góp phương pháp/benchmark)*
**Cái mới:** một quy trình **mask-định-danh** tách độ khó retrieval thành 3 regime: *corpus-level* | *company-scoped* | *within-(company,year)*. Chứng minh định lượng phần gain do metadata là *artifact*, cô lập **residual content-only difficulty** — bài toán thật. Tổng quát hoá khung của *Decomposing Retrieval Failures* (vốn chỉ single-doc, FinanceBench, **không** có artifact-control) sang **multi-doc entity-clustered** và bổ sung kiểm soát artifact.
**Vì sao reviewer thích:** reframe benchmark + "killer diagnostic" — biến điểm yếu (artifact) thành đóng góp.

### C2 — Fact-Level Gated Late-Interaction Retrieval + Conditional Salience *(đóng góp retrieval)*
**Cơ chế:** thay 1 vector/doc bằng *tập fact*; điểm liên quan là tổng theo "ô cần điền" của đóng góp fact khớp nhất:
$$S(q,d)=\sum_j \max_{f\in\mathcal L(d)}\big[\ \sigma_m(\tilde c_j,c_f)\cdot \gamma_e(\tilde e,e_f)\cdot \gamma_t(\tilde t,t_f)\cdot \rho_j(\mu_f)\ \big]$$
- Luồng tách rời: **concept-embedding ⟂ magnitude-scalar ⟂ identity-key** (đáp Granularity Dilemma + Numeracy Gap).
- **Cổng nhân** entity/time → sai company/sai năm bị *triệt tiêu điểm*, không phải bị "trừ điểm" (khác additive fusion hiện tại).
- **Conditional salience** (`loclex` tổng quát hoá, học được): "độ phân biệt" định nghĩa *tương đối với cụm đã truy hồi*, không phải toàn corpus.
**Killer experiment (đã có tín hiệu sơ bộ):** conditional salience THUA ở corpus-level, THẮNG within-cluster; dense/ColBERT thua **cả hai** → kết luận phản trực giác *"global-neural không bắt được within-filing numeric discrimination"*.

### C3 — Ledger-Derived Channel-Aligned Hard Negatives *(đóng góp huấn luyện retriever)*
**Cơ chế:** sinh negative bằng cách *biến đổi đúng fact chứa đáp án* theo **đúng một kênh** (period/entity/concept/scale/value). Dưới giả định phát biểu rõ, các negative này **chắc chắn làm đáp án không còn đúng** ⇒ không phải positive ⇒ **không cần lọc false-negative** (khác RocketQA/NV-Retriever). Mỗi loại negative giám sát đúng một cổng/luồng của C2.
**Tổng quát:** recipe cho mọi corpus *versioned/structured*.

### C4 — Ledger-Grounded Inference-Time Generation Support *(đóng góp lớn nhất — sinh, không huấn luyện generator)*
**Cơ chế:** Fact Ledger = bộ điều phối *ký hiệu, tất định, miễn gán nhãn* nằm giữa retriever và generator frozen.

Đầu vào là top-3 retrieval, thường có ≥2 mẫu nhiễu. KG tạo một `EvidencePack` gồm:
- `KG_SELECTED_DOC`: tài liệu được arbitration/focus dựa trên fact-level support, intent số học và rank prior.
- `KG_SYMBOLIC_ANSWER`: đáp án ký hiệu khi confidence đủ cao, ví dụ lookup, difference, percent change, ratio, question-literal ratio, balance roll-forward, factor bridge.
- `KG_TRACE`: phép tính và toán hạng đã dùng.
- `KG_PROVENANCE`: nguồn từng toán hạng ở cell/table/text/question, gồm concept, period, value.
- `KG_CONFLICTS`: các fact cạnh tranh/xung đột để LLM không nhầm nhiễu là bằng chứng.

Generator Qwen3-4B/9B vẫn **frozen**: prompt yêu cầu copy `KG_SYMBOLIC_ANSWER` khi có, hoặc dùng trace/provenance để trả lời ngắn, kiểm chứng được. `ExtractiveGenerator` cũng đã được sửa để copy symbolic answer, giúp đo trần deterministic.

**Định vị:** vs HierFinRAG chỉ route calculator, C4 của ta còn giải bài toán top-3 nhiễu của retrieval, gắn provenance fact-level và kiểm soát hallucination tại generator. vs RL/GRPO, C4 hiện tại không cần training, phù hợp hạ tầng hiện tại và mục tiêu xây KG hữu ích trước.

### C5 — Selective, Ledger-Grounded Generation *(đóng góp toàn trình + minh bạch)*
Vì MRR@3 trao generator top-3 (≥2 nhiễu "giống thật"):
- **Calibrated confidence margin** route: high-margin → tin top-1 kèm *fact evidence đã định vị* (loclex chỉ thẳng cell phân biệt cho LLM); low-margin → trình đa-doc kèm *cờ xung đột fact*.
- **Provenance fact→(concept,entity,period,cell)** cho mỗi con số trong đáp án → audit được.
Đáp trọn 3 mục tiêu KG-cho-generator bạn nêu: (1) focus đúng mẫu trong top-3; (2) pre-extract giảm tải LLM; (3) minh bạch giải thích.

---

## 6. Mối gắn kết — "Fact-Level Indexing" (câu chuyện AAAI)
Một Fact Graph, bốn vai trò: **C2** dùng nó để chấm điểm, **C3** biến đổi nó để sinh negative, **C4** dùng nó chọn tài liệu đúng + lập kế hoạch tính toán + kiểm chứng tại inference, **C5** dùng nó làm bằng chứng/provenance; **C1** chứng minh vì sao cấp-tài-liệu (kể cả metadata) không đủ. Không đối thủ nào (Structure-First / HierFinRAG / Decomposing-Failures / Metadata-RAG) phủ quá một lát. *Sự cần-nhau ba-bốn chiều* là bằng chứng đây là lời giải hệ thống, không phải ghép kỹ thuật.

---

## 7. Kế hoạch đánh giá
- **Benchmark chính:** T²-RAGBench (FinQA, ConvFinQA, TAT-DQA).
- **Mở rộng (cho tính tổng quát AAAI):** thêm ≥1 trong **DocFinQA** (full 10-K, 100K+ token — arXiv [2401.06915](https://arxiv.org/html/2401.06915v3)), **MultiHiertt** (multi hierarchical tables), **FinanceBench** (để so trực tiếp với Decomposing-Failures). Ưu tiên DocFinQA vì đẩy đúng vào "within-filing" thật.
- **Retrieval:** MRR@3, R@{1,3,5}, nDCG@3 — **báo cáo theo từng regime của C1** (đây là điểm trung thực).
- **Generation:** numeric accuracy, constraint-violation rate, và *arithmetic-faithfulness* (tỉ lệ bước grounded + consistent), + tỉ lệ "đúng đáp án/sai lập luận" giảm bao nhiêu.
- **Generator:** Qwen3 4B & 9B frozen (theo yêu cầu hiện tại), so few-shot / PoT / retrieval-only context / **KG-grounded C4 context**. RL/GRPO chỉ là future optional.
- **Baselines retrieval:** BM25, Hybrid-BM25, BGE-M3, e5-large, ColBERTv2, reranker, HyDE, metadata-filter+rerank.
- **Ablation:** từng cổng (entity/time), luồng magnitude, fact-level vs single-vector, conditional salience, từng loại negative, từng số hạng reward + ràng buộc $\lambda_g+\lambda_a<1$.

---

## 8. Lộ trình thực thi
**Phase 0 — Dựng lại code đã mất** (đã xác định thiếu): `experts/local_lexical.py` (loclex), `experts/schema.py`, company-pool trong `meta_retriever.py` + `scripts/modular_retrieval.py`, `generation/retrieval_bridge.py`, `domain/financial.py`, `scripts/research/{difficulty_decomposition,neural_baselines,selective_generation}.py`.
**Phase 1 — C1 diagnostic** (chạy thật, xác nhận artifact 0.31/0.53/0.51).
**Phase 2 — C2** fact-level gated late-interaction (nâng từ MMER additive → multiplicative gated, học được).
**Phase 3 — C3** channel-aligned negatives + InfoNCE huấn luyện C2.
**Phase 4 — C4** inference-time KG bridge: top-3 arbitration, symbolic plan, provenance, conflict detection, frozen Qwen support. *(Đã triển khai v4.)*
**Phase 5 — C5** selective generation + end-to-end NM + failure-driven rule iteration. *(Đã có vòng rules2→rules4; tiếp tục mở rộng.)*
**Phase 6 — Mở rộng benchmark + significance tests + viết bài.**

## 9. Rủi ro & khả thi
- **Khả thi hạ tầng:** hướng hiện tại không cần train generator; Qwen3-4B chạy được để đánh giá sample, Qwen3-9B là bước mở rộng nếu đủ VRAM/quantization.
- **Rủi ro extraction:** chất lượng fact extraction là trần của cả C2/C4 → cần đo *tỉ lệ lấy được đúng fact chứa đáp án*, fallback row/text-level, và guardrail confidence để không ép symbolic sai vào generator.
- **Rủi ro novelty bị "scoop":** within-doc (2602.17981), KG-numeric (2601.07754), HierFinRAG rất gần → ta phải nhấn **(a)** KG arbitration trên top-3 nhiễu, **(b)** symbolic plan/provenance cho generator frozen, và **(c)** thống nhất 4-vai-substrate; tránh khung "chỉ retrieval" hoặc "chỉ calculator".

---

## 10. KẾT QUẢ THỰC NGHIỆM ĐÃ CHẠY (cập nhật trực tiếp, dữ liệu thật)

### 10.1 C1 — Difficulty Decomposition (đã chạy full 3 datasets, sparse-only, recall=1.0)

| Regime | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| A corpus BM25 | 0.666 | 0.641 | 0.418 |
| A' corpus BM25 (company-masked) | 0.600 | 0.590 | 0.394 |
| B company-pool + global BM25 | 0.719 | 0.692 | 0.565 |
| **C company-pool + loclex** | **0.819** | **0.825** | **0.681** |
| D within-(company,year) + loclex | 0.915 | 0.919 | 0.681 |

company-in-Q = **100%** cả 3; avg company-pool = 47.6/32.6/23.3; avg (company,year)-pool = 9.6/6.2/23.3.

**Đính chính trung thực quan trọng:** "artifact metadata" NHỎ HƠN báo cáo cũ tuyên bố. Mask company khỏi câu hỏi chỉ làm BM25 giảm ~0.05–0.07 (A→A'), KHÔNG sụt về 0.31. → BM25 hầu như không dựa vào tên công ty về mặt từ vựng; **đòn bẩy thật là `loclex` (within-cluster IDF)**: C−B = +0.10/+0.13/+0.12. TAT-DQA là canary: year cho 0 phân biệt (D=C=0.681) → đây là **sàn khó thật**.

### 10.2 C2 — FactLevel zero-shot: NEGATIVE RESULT (đã chạy controlled, 5-fold CV)

Company-pool fusion (5-fold CV), TAT-DQA, **có vs không** `factlevel`:

| Config | linear | mlp | gate | factlevel weight |
|---|---|---|---|---|
| control: meta+loclex+concept+cell | 0.6968 | 0.7015 | 0.7008 | — |
| +factlevel (zero-shot bge-small) | 0.6957 | 0.7043 | 0.6983 | 0.054 |

→ **Chênh trong khoảng nhiễu (±0.005).** `factlevel` standalone chỉ 0.213; weight fusion 0.054. FinQA cũng tương tự (fusion 0.901, factlevel weight 0.068).

**Chẩn đoán (đã kiểm chứng):** extraction KHÔNG phải trần — gold docs có ledger ≥1 fact 94.7–97.7%, ~13–14 facts/gold. Vấn đề là **σ_m semantic zero-shot không phân biệt within-company tốt hơn loclex lexical**. 

**Kết luận redirect:** *fact-level retrieval zero-shot ≈ loclex.* Muốn C2 vượt loclex, σ_m (luồng concept) phải được **HUẤN LUYỆN** bằng channel-aligned hard negatives (C3) — C2 và C3 **không tách rời** (đúng §5.8). Bước tiếp theo: xây C3 negative generator + contrastive training cho disentangled encoder, mục tiêu vượt loclex 0.819/0.825/0.681 (đặc biệt TAT-DQA residual).

### 10.3 C4/C5 — KG bridge cho generator frozen (đã triển khai v4)

**Thay đổi chính trong code:**
- `generation/retrieval_bridge.py`: xây `EvidencePack`, KG arbitration, symbolic answer, trace, provenance, conflict list; sửa bug extraction khi context TAT có nhiều bảng/text để không mất period.
- `ledger/select.py`: thêm intent/rule cho percent change, ratio, question-literal ratio, comparison year, balance roll-forward, factor bridge; thêm confidence guardrail để giảm symbolic sai.
- `generation/prompts.py`: prompt yêu cầu generator copy/check `KG_SYMBOLIC_ANSWER` khi có.
- `generation/generator.py`: `ExtractiveGenerator` copy symbolic answer để đo trần deterministic.
- `generation/verifier.py`: kiểm grounding/arithmetic/process fractions cho output.
- `tests/test_ledger_rag.py`: 16 smoke tests, gồm các ca TAT thực tế: chọn năm đúng, ratio dùng số trong question, direct percent từ narrative, copy symbolic.

**KG arbitration / symbolic plan (`rules4`):**

| Dataset | Original top-1 | KG top-1 | Best policy | Coverage | NM when symbolic |
|---|---:|---:|---:|---:|---:|
| FinQA | 0.6417 | 0.6469 | **0.6600** | 0.5937 | 0.2349 |
| ConvFinQA | 0.7279 | 0.7287 | **0.7420** | 0.5604 | 0.3782 |
| TAT-DQA | 0.3260 | 0.3628 | **0.3820** | 0.6154 | 0.1491 |

**Frozen Qwen3-4B sample:**

| Dataset | rules2 | rules3 | rules4 | Nhận xét |
|---|---:|---:|---:|---|
| FinQA 100 mẫu | 0.21 | 0.21 | **0.22** | guardrail cứu lookup/symbolic sai |
| TAT-DQA 100 mẫu | 0.18 | 0.19 | **0.20** | grounded fraction tăng 0.32→0.43 |

**Full deterministic extractive-symbolic:**

| Dataset | rules2 NM | rules4 NM | rules4 grounded |
|---|---:|---:|---:|
| FinQA | 0.1229 | **0.1526** | 0.1325 |
| ConvFinQA | **0.2296** | 0.2282 | 0.3673 |
| TAT-DQA | **0.1206** | 0.1180 | **0.4554** |

Đọc kết quả: `rules4` không tối đa coverage mù quáng như `rules3`; nó giảm coverage để tăng precision/focus. Với Qwen frozen, đây là cấu hình tốt nhất hiện tại trên cả FinQA và TAT sample.

### 10.4 Trạng thái code (sau phiên này)
- ✅ Dựng lại & verify: `experts/local_lexical.py` (loclex), company-pool trong `meta_retriever.py` + `--company-pool/--meta-max-add` trong `modular_retrieval.py`. Số khớp chính xác báo cáo.
- ✅ Mới: `scripts/research/difficulty_decomposition.py` (C1), `experts/fact_level.py` (C2 v1, zero-shot).
- ✅ Mới/đã nâng cấp: `generation/retrieval_bridge.py`, `ledger/select.py`, `generation/generator.py`, `generation/prompts.py`, `generation/verifier.py`, `scripts/research/kg_bridge_eval.py`, `tests/test_ledger_rag.py`.
- ⏭️ Chưa làm: C3 channel-aligned training cho σ_m; mở rộng benchmark ngoài T²-RAGBench; Qwen3-9B frozen; significance tests. GRPO/process-reward training chỉ là future optional, không thuộc hướng hiện tại.

---
*Nguồn đã truy cập:* T²-RAGBench [2506.12071](https://arxiv.org/pdf/2506.12071) · Metadata-RAG [2510.24402](https://arxiv.org/pdf/2510.24402) · Decomposing Failures [2602.17981](https://arxiv.org/abs/2602.17981) · Structure-First [2601.07754](https://arxiv.org/abs/2601.07754) · HierFinRAG [MDPI](https://www.mdpi.com/2227-9709/13/2/30) · Granularity Dilemma [2506.08592](https://arxiv.org/pdf/2506.08592) · Numeracy Gap [2509.05691](https://arxiv.org/pdf/2509.05691) · SubgraphRAG/Less-is-More [2510.14271](https://arxiv.org/html/2510.14271v1) · When-to-use-Graphs [2506.05690](https://arxiv.org/html/2506.05690v3) · CRAFT [2602.01348](https://arxiv.org/pdf/2602.01348) · DocFinQA [2401.06915](https://arxiv.org/html/2401.06915v3).
