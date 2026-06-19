# ĐÁNH GIÁ TOÀN DIỆN & KHÁCH QUAN — Tất cả Thế hệ, Triển khai, Phương pháp

> Tài liệu này đánh giá **trung thực, có số liệu thật** mọi thế hệ của hệ thống (từ GSR gốc → LEDGER-RAG v2 → MMER → company-pool/loclex → KG-for-generator → coordinate/multipath → KG enrichment), chỉ rõ **cái gì hoạt động, cái gì không, và vấn đề nằm ở đâu**, để định hướng phát triển tiếp. Mọi con số dưới đây là đo thật trong phiên này (2× Tesla T4, dữ liệu `G4KMU/t2-ragbench`).

---

## 0. Kết luận điều hành (đọc trước)

1. **Retrieval đã được giải tốt** bằng tín hiệu fact-level/metadata: trained company-pool fusion đạt **MRR@3 = 0.929 / 0.940 / 0.702** (FinQA/ConvFinQA/TAT-DQA). Nhưng **phần lớn là "regime switch" hợp lệ nhưng không mới** (company+year). Residual khó thật = **TAT-DQA 0.702** (within-company, year vô dụng).
2. **KG-for-generator KHÔNG cải thiện độ chính xác (NM) với model đủ mạnh.** Với Qwen-3B, KG-evidence giúp nhẹ FinQA (+0.017); nhưng với **Qwen-7B, raw table VƯỢT KG-evidence** (FinQA +0.06, ConvFinQA +0.09). Symbolic-override và hybrid **luôn ≤ LLM**. → Giá trị NM của KG ở pha sinh **bốc hơi khi model mạnh lên**.
3. **Giá trị KG đáng bảo vệ KHÔNG phải accuracy mà là: retrieval (within-cluster) + provenance/auditability + grounding-check faithfulness.** Đây là đóng góp đúng cho miền tài chính, không phải đua NM với LLM lớn.
4. **Accounting-identity verifier không khả thi trên benchmark này** (snippet 1 bảng, hiếm đủ operand; semantic-matching tạo identity giả). Đừng dựa vào identities.

---

## 1. Bảng các thế hệ & đánh giá khách quan

| Thế hệ | Ý tưởng | Kết quả đo | Đánh giá trung thực |
|---|---|---|---|
| **GSR gốc** (GAT + constraint-KG) | KG cột-template + GAT + constraint score | đóng góp **≈0** | Template khớp sai chiều (row-major), CS=1.0 mọi doc, encoder train≠eval. **Chết.** |
| **LEDGER-RAG v2** (E1/E2 ontology + C2/C3 + CACL2) | entity ontology + concept coverage + InfoNCE | FULL+C3 MRR@3 0.743/0.818/0.455 | Thật, nhưng lợi ích chính từ **company filter**, không phải coverage. |
| **MMER + FactGate** | mixture-of-experts + fact-level gating | +0.025–0.05 | FactGate đóng góp biên (+1.4pp); fusion hữu ích. |
| **MetadataRetriever (company-pool)** | pool = toàn bộ chunk cùng công ty | recall→1.0 | **Hợp lệ nhưng artifact** (company trong câu hỏi). Trivialize corpus retrieval. |
| **loclex (pool-local IDF)** | BM25 IDF tính lại trong cụm công ty | +0.10–0.13 standalone | **Đòn bẩy thật** (within-cluster). First-order nhưng đúng. |
| **Trained company-pool fusion** | loclex+concept+cell+meta, 5-fold CV | **0.929/0.940/0.702** | **Retrieval mạnh nhất hiện có.** TAT 0.702 = residual khó thật. |
| **FactLevel (C2 zero-shot)** | disentangled concept/magnitude/identity | KHÔNG vượt loclex | Negative result; σ_m cần train (C3). |
| **KG-for-generator bridge** | arbitrate top-3 + evidence + provenance | xem §3 | Provenance tốt; NM **không vượt LLM mạnh**. |
| **Coordinate grounding (2D)** | ô = row(concept)×col(period) | coord<heuristic, **either>cả hai** | Bổ trợ heuristic; nền cho agreement. |
| **Multipath agreement** | 3 đường bỏ phiếu | votes→NM đơn điệu (3-vote 0.79/0.65) | **Calibrated abstention thật**, nhưng override ≤ LLM end-to-end. |
| **KG enrichment (semantic canon)** | embedding nearest-concept | coverage ×3 nhưng identity giả | Coverage↑ nhưng **identity-verifier vẫn không khả thi**. |

---

## 2. Retrieval — phân tích chi tiết

### 2.1 Difficulty decomposition (C1, chống-artifact)
| Regime | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| corpus BM25 | 0.666 | 0.641 | 0.418 |
| corpus BM25, company-masked | 0.600 | 0.590 | 0.394 |
| company-pool + BM25 | 0.719 | 0.692 | 0.565 |
| **company-pool + loclex** | **0.819** | **0.825** | **0.681** |
| within-(company,year) + loclex | 0.915 | 0.919 | 0.681 |
| **trained fusion (meta+loclex+concept+cell)** | **0.929** | **0.940** | **0.702** |

**Vấn đề/nhận định:** (i) company-masking chỉ giảm BM25 ~0.06 → "artifact metadata" nhỏ hơn tuyên bố cũ (0.31); đòn bẩy thật là **loclex within-cluster**. (ii) TAT-DQA toàn FY2019 → year vô dụng (D=C) → **0.702 là sàn khó thật**, nơi nghiên cứu còn dư địa. (iii) dense/ColBERT thua sparse ở cả 2 regime (bảng số phá embedding).

### 2.2 Vấn đề retrieval còn mở
- FinQA/ConvFinQA gần bão hòa nhờ meta=year (weight 1.5+) → **không phải đóng góp khoa học**.
- TAT-DQA residual cần tín hiệu fact-level **được huấn luyện** (C3) — zero-shot đã thất bại. Đây là hướng retrieval thật sự cần đột phá.

---

## 3. Generation — phân tích chi tiết (phần "vấn đề ở đâu" rõ nhất)

### 3.1 Phân rã lỗi symbolic answer (trên retrieval mạnh, gold-in-top3)
| Loại | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| exact | 17.5% | 25.1% | 15.9% |
| format-recoverable (%/scale/sign) | ~2% | ~2% | ~2% |
| wrong-operand (value có trong facts) | 1.7% | 15.5% | 7.3% |
| **value absent (grounding/compute sai)** | **78.8%** | **57.5%** | **74.4%** |

→ Nút thắt = **cell-grounding**, độc lập retrieval (số gần như y hệt retrieval yếu↔mạnh). Quy luật: exact-rate tỉ lệ với mức bảng ràng buộc 2D (difference/%change 40–48% ≫ lookup 8–34% ≫ sum 1–4%).

### 3.2 Multipath agreement calibration (trên gold doc)
| votes | FinQA | ConvFinQA | TAT-DQA |
|---|---|---|---|
| =1 | 0.18 | 0.14 | 0.07 |
| =2 | 0.29 | 0.50 | 0.18 |
| =3 | **0.79** | **0.65** | 0.19 |

→ Đồng thuận = tín hiệu precision tất định, minh bạch. **Nhưng precision tối đa ~0.65–0.79 và coverage thấp.**

### 3.3 End-to-end NM (Qwen frozen) — KẾT QUẢ QUAN TRỌNG NHẤT
| Policy | FinQA-3B | FinQA-7B | ConvFinQA-3B | ConvFinQA-7B |
|---|---|---|---|---|
| raw top-3 | 0.125 | **0.250** | 0.383 | **0.510** |
| KG-evidence | **0.142** | 0.190 | 0.383 | 0.420 |
| hybrid@votes≥2 | 0.133 | 0.170 | 0.317 | 0.350 |
| hybrid@votes≥3 | 0.142 | 0.190 | 0.375 | 0.410 |
| kg+verify | 0.133 | 0.170 | 0.367 | 0.400 |

**VẤN ĐỀ CỐT LÕI (đánh giá khách quan):**
- **KG-evidence chỉ giúp model YẾU (3B FinQA +0.017); với model MẠNH (7B) nó HẠI** (FinQA −0.06, ConvFinQA −0.09). Lý do: filtering evidence đôi khi bỏ ô cần thiết / focus sai, trong khi model mạnh đọc bảng thô tốt hơn.
- **Symbolic override / hybrid LUÔN ≤ LLM.** LLM grounding ô tốt hơn bộ symbolic tất định.
- **NM tuyệt đối bị chặn bởi model** (leaderboard 72–76 NM dùng GPT-5.4/QwQ-32B/LLaMA-70B). Trong phạm vi Qwen-4B/7B, NM ~0.25–0.51.

→ **Giá trị KG ở pha sinh KHÔNG phải accuracy.** Nó là: (a) **provenance** (cell-level, audit được — bắt 17–22% đáp án LLM ungrounded), (b) **calibrated abstention** (multipath votes), (c) giúp regime model-yếu. Headline bài báo phải là **faithfulness + auditability + retrieval within-cluster**, KHÔNG phải "KG tăng NM".

---

## 4. Xây dựng KG — chẩn đoán & vấn đề

| Thành phần | Coverage/firing | Đánh giá |
|---|---|---|
| 2D grid (row,col) | **100%** | Trụ cột vững — nền coordinate grounding. |
| Temporal edges | 37–58% (~6–8/doc) | Tốt cho Δ/%change. |
| Canonical concept | **21–25%** | **Thấp** → identities chết. |
| Identity edges firing | **0–2.4%** | **Verifier kế toán không hoạt động.** |
| Semantic enrich | coverage→58–63% nhưng identity-satisfied **1.0→0.17** | Tạo **identity giả** → không dùng cho verify. |

**Vấn đề KG:** đẳng thức kế toán **không khả thi** trên T²-RAGBench (snippet 1 bảng hiếm đủ operand; matching lỏng tạo false positive). **Kết luận xây-KG:** giữ 2D grid + temporal + provenance + grounding-check; **bỏ identity-verifier như tín hiệu chính**; semantic-canon chỉ làm soft-signal.

---

## 5. VẤN ĐỀ TỔNG HỢP — ở đâu?
1. **Retrieval FinQA/ConvFinQA = artifact (metadata/year)**, không phải khoa học. Chỉ TAT-DQA residual là thật.
2. **KG-for-generator không nâng NM với model mạnh** — đây là vấn đề lớn nhất cho luận điểm "KG giảm tải LLM".
3. **Symbolic answering trần thấp** (~0.3–0.5 ngay cả gold doc) do cell-grounding khó; LLM làm tốt hơn.
4. **Identity verifier chết** trên benchmark này.
5. **NM tuyệt đối bị chặn bởi kích thước model** (Qwen-4B/7B « LLM leaderboard).

---

## 6. ĐỊNH HƯỚNG PHÁT TRIỂN (ưu tiên theo bằng chứng)

**A. Định vị lại đóng góp (quan trọng nhất):** chuyển từ "KG tăng accuracy" sang **"KG cho retrieval within-cluster + faithfulness/auditability"**. Đo và báo cáo: provenance accuracy, hallucination-catch rate, calibrated abstention (votes→precision), grounded-number rate. Đây là đóng góp đứng vững được, đúng miền tài chính.

**B. Retrieval (đột phá thật):** train fact-level σ_m bằng channel-aligned negatives (C3) cho **TAT-DQA residual** (zero-shot đã thất bại) — đây là chỗ retrieval còn dư địa khoa học.

**C. Generation đúng cách dùng KG (không filter cứng):** thay vì thay/lọc, dùng KG làm **lớp xác minh sau sinh** (post-hoc verify số LLM có khớp ô; nếu không → re-ask chỉ thẳng ô) — không bỏ thông tin bảng thô. Thử nghiệm: raw + KG-verify-and-reask vs raw.

**D. Model lớn hơn** (Qwen-9B/14B/32B) là đòn bẩy NM trực tiếp — nhưng phải kết hợp KG ở chế độ "verify/augment" chứ không "filter".

**E. Mở rộng benchmark** (DocFinQA/FinanceBench) để chứng minh tính tổng quát của retrieval within-filing + faithfulness, đặc biệt nơi tài liệu dài (KG-evidence focusing CÓ THỂ giúp lại khi context quá dài cho LLM).

---

## 7. Hiện vật (code/scripts/outputs)
- **Retrieval:** `experts/{local_lexical,meta_retriever,fact_level}.py`, `scripts/modular_retrieval.py --company-pool`.
- **KG:** `kg/fact_graph.py`, `ledger/{coordinate,multipath,semantic_concepts}.py`.
- **Bridge/gen:** `generation/retrieval_bridge.py`, `ledger/select.py`, `generation/generator.py`.
- **Eval scripts:** `scripts/research/{difficulty_decomposition, build_strong_top3, symbolic_calibration, symbolic_error_decomp, coordinate_eval, multipath_eval, generation_e2e, kg_construction_diag, kg_enrich_eval, factlevel_contrastive}.py`.
- **Outputs:** `outputs/{cp_fusion, strong_retrieval, research/*}`.
