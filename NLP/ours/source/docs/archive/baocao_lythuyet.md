BÁO CÁO KIỂM TOÁN KHOA HỌC — DỰ ÁN STRUCTURE-GROUNDED FINANCIAL NUMERICAL QA (AAAI-27)
Phạm vi kiểm toán: mã nguồn ours/source/, pipeline dữ liệu T²-RAGBench (FinQA/ConvFinQA/TAT-DQA) + FinanceBench, các artifact đã thực thi trong outputs/, và tài liệu thiết kế.
Vai trò kép: (A) AAAI-27 Area Chair — đánh giá quyết định accept/reject; (B) Principal Research Scientist — đề xuất chiến lược nâng cấp.
Nguyên tắc: chỉ dùng số đã thực thi và ghi nhận trong repo; phân biệt rõ honest vs leaky.

PHẦN 1 — ĐÁNH GIÁ TÍNH SẴN SÀNG CHO AAAI-27
1.1. Novelty — Tính mới về lý thuyết/kiến trúc
Phải tách bạch ba "lớp" đóng góp trong repo, vì chúng có mức độ mới rất khác nhau:

Lớp	Thành phần	Mức độ mới (đánh giá AC)
L1 — Retrieval	MMER 7-expert fusion (modular_retrieval.py, experts/)	Thấp–Trung bình. Đây là fusion-of-experts engineering. Listwise InfoNCE trên 7 retriever + mixture-of-experts gating là kỹ thuật đã biết. Reviewer sẽ gọi là "incremental".
L2 — Verifier/Reliability	CPR (Concept–Period–Role) grounding (cpr_verifier.py)	Trung bình–Cao. Đây là lõi novelty thật sự. Một verifier annotation-free, model-free ràng buộc đáp án vào (khái niệm × kỳ × vai trò toán tử) trên đồ thị cấu trúc, đối xứng với generation.
L3 — Framework	TCEP — Type-Constrained Evidence Paths (FRAMEWORK_TCEP.md)	Cao nếu được củng cố. Tổng quát hoá grounding/CPR/multi-operand thành lý thuyết "typed evidence path" với 3 định luật thực nghiệm có thể falsify (và đã có 1 giả thuyết bị bác bỏ → tinh chỉnh). Đây là thứ nâng bài từ "một verifier tốt hơn" lên "một framework".
Kết luận về novelty: Nếu định vị bài theo L1 (retrieval), đây là application/engineering paper — gần như chắc chắn bị reject ở AAAI vì "không có cơ chế mới". Nhưng repo đã tự nhận ra điều này (CONTRIBUTION_AUDIT.md: "novelty must be carried by C1 (CPR)") và đã chuyển trục sang L2+L3. Với trục CPR/TCEP, bài có một luận điểm khoa học chân chính: path existence không phải tín hiệu tin cậy; path type-consistency mới là (TCEP Law 1 & 2). Đây là một phát hiện phản trực giác, đo được, có ablation, và tổng quát ngoài tài chính — đủ ngưỡng novelty cho AAAI, nhưng chưa đủ độ chín (xem 1.3).

Phán quyết novelty: Borderline-positive, với điều kiện bài được viết hoàn toàn quanh TCEP/CPR (reliability + selective answering), KHÔNG bán retrieval như đóng góp chính.

1.2. Empirical Rigor — Độ chín thực nghiệm
Điểm mạnh (vượt chuẩn nhiều submission AAAI):

Quy trình đo trung thực rất tốt: honest contract (bóc prefix metadata, chỉ rút năm/công ty từ câu hỏi), 5-fold CV cho fusion head, paired bootstrap CI (2000 resamples), kiểm tra leakage chủ động (validity_check.py). Đây là mức kỷ luật mà đa số reviewer sẽ khen.
Bao phủ thực nghiệm rộng: 3 dataset T²-RAGBench full test set (1147/3458/1144), OOD FinanceBench, cross-generator (Qwen2.5-3B, Qwen3-4B, 7B), ablation thành phần C/P/R, "auditable ceiling" analysis, long-context dilution thay thế DocFinQA, head-to-head vs self-consistency.
Tính trung thực về kết quả âm: answer-routing thất bại được báo cáo công khai; cấu trúc-expert đóng góp gần 0 trong retrieval được thừa nhận; giả thuyết "typed 3-op" làm hại AUROC được ghi nhận và revert.
Điểm yếu thực nghiệm:

Generator yếu (3B–7B). Number-Match tuyệt đối thấp (FinQA 0.10–0.28). Reviewer sẽ hỏi: tín hiệu CPR có còn giá trị với GPT-4o/Claude/Qwen-72B — nơi base accuracy cao hơn nhiều? Chưa có bằng chứng.
AUROC tuyệt đối khiêm tốn (0.66–0.76). Đây là classifier "hữu ích nhưng không mạnh".
DocFinQA hỏng (HF loader lỗi) → không có thí nghiệm long-document thật, chỉ có proxy dilution. Đây là một lỗ hổng mà reviewer finance-RAG sẽ nhắm tới.
FinanceBench n=126, gold-evidence setting → cô lập verification khỏi retrieval; reviewer sẽ nói "evidence-level, không phải document-level".
Provenance precision audit mới ở mức 100-sample proxy, chưa có human audit hoàn chỉnh.
Phán quyết empirical: Phần "Experiments" đã đủ dày về số lượng và vượt chuẩn về tính trung thực, nhưng thiếu một headline thuyết phục (generator mạnh + một benchmark long-document thật). Hiện trạng: đủ cho một bài solid, chưa đủ cho một bài strong-accept.

1.3. Ba điểm yếu chí mạng (Blind spots → khả năng Reject cao)
BLIND SPOT #1 — Khủng hoảng tính toàn vẹn của số liệu retrieval (leakage integrity).
Repo tồn tại HAI bộ số mâu thuẫn cho cùng module: số honest MMER (5-fold CV, không gold metadata) là W.Avg MRR@3 0.722 (07_research_report.md), nhưng file on-disk outputs/modular/*/modular.json lại cho FinQA 0.90 / TAT-DQA 0.70 với pool_recall = 1.0 — tức một run leakier (company-complete pool ⇒ recall=1.0 ⇒ near-oracle). CONTRIBUTION_AUDIT.md đã gắn cờ ⚠ INTEGRITY FLAG cho chính vấn đề này. Nếu một con số leaky lọt vào bản nộp, đây là lý do reject tức thì (data leakage là "tử huyệt" tại AAAI). Bắt buộc xoá/ghi đè file leaky và chỉ dùng 0.722.

BLIND SPOT #2 — Trần trích xuất (extraction ceiling) bóp nghẹn toàn hệ thống, và lõi novelty phụ thuộc heuristic chưa được đo.
(a) Trần "certifiable" trên gold-doc chỉ 47–69% (CONTRIBUTION_AUDIT.md, fact_extraction_recall.py). Nghĩa là: ngay cả khi retrieval hoàn hảo và sinh hoàn hảo, verifier vẫn không thể chứng nhận >47% câu FinQA. Reviewer sẽ nói: "đóng góp của các anh bị giới hạn trên bởi một bộ trích xuất bảng yếu, chưa được giải quyết."
(b) Thành phần Role — "the workhorse" của CPR (R-alone AUROC ≈ full CPR) — phụ thuộc hoàn toàn vào calculation_plan heuristic trong ledger/select.py. Độ chính xác gán vai trò operand CHƯA hề được đo. Nếu role bị gán sai, toàn bộ tín hiệu sụp. Đây là "thin criterion" mà một reviewer kỹ tính sẽ khoét: "Đóng góp chính của các anh được driven bởi một heuristic không được validate."

BLIND SPOT #3 — Câu chuyện "selective answering" sụp khi base accuracy thấp + định vị giá trị mơ hồ.
Selective answering (C2) — payoff khả triển khai của bài — collapse khi base accuracy thấp: FinQA ở 28% accuracy không thể đạt ngưỡng 60–70% ở bất kỳ coverage nào → coverage@risk ≈ 0 (CONTRIBUTION_AUDIT.md). Kết quả đẹp (5×) chỉ tồn tại trên ConvFinQA. Reviewer sẽ hỏi: "Vậy đóng góp chỉ hoạt động khi model đã đủ tốt? Khi đó dùng confidence của chính LLM mạnh có hơn không?" — và CPR chưa được so với confidence/logprob của một LLM mạnh, chỉ so với value-only và self-consistency raw↔KG (một baseline khá yếu). Thiếu so sánh này, novelty của reliability-signal bị nghi ngờ.

PHẦN 2 — CHIẾN LƯỢC LÀM DÀY ĐÓNG GÓP KHOA HỌC
2.1. Về Kiến trúc/Mô hình (Core Contributions)
Học hàm typing (Learned CPR typing) — ưu tiên #1. Hiện concept-consistency dùng token-overlap thô (ontology chỉ phủ ~14% concept), period dùng partial-credit hand-tuned, role dùng heuristic plan. Đề xuất: huấn luyện một concept encoder tương phản nhỏ trên chuỗi line-item (annotation-free, dùng same-concept/diff-concept trong corpus làm cặp) để thay token-overlap; và một role-assignment probe có giám sát yếu từ calculation traces. Điều này biến CPR từ "tập heuristic" thành "hàm typing học được nhưng vẫn giải thích được" — nâng độ chín lý thuyết, đúng như FRAMEWORK_TCEP.md §5 đã chỉ ra là cần.

Giải bài toán unambiguous operand attribution cho multi-operand path. Đây là open problem thật sự mà repo tự nêu (typed 3-op làm hại AUROC vì một value map tới nhiều cell). Nếu giải được — ví dụ ràng buộc path chỉ qua concept được câu hỏi nhắc tới, hoặc một mô hình attribution có hiệu chỉnh — đây sẽ là một đóng góp lý thuyết độc lập (nâng trần certifiable từ 47→80% và đồng thời tăng AUROC). Đây là lever đơn lẻ lớn nhất.

Calibrated reliability head có lý thuyết: thay logistic calibrator 5-fold hiện tại bằng một conformal prediction layer trên điểm CPR để cho đảm bảo coverage có chứng minh (distribution-free). Selective answering với conformal guarantee là một câu chuyện rất "AAAI trustworthy-ML".

Nâng đồ thị cấu trúc thành reasoning substrate dùng chung retrieve+verify (đã có khung trong kg/structure_graph.py): thêm cạnh mentions(text→cell) và đẳng thức theo-hàng (Total-row = Σ component-rows) để equation-check kích hoạt trên bảng row-major — vá đúng lỗi B5 mà ASSESSMENT.md đã chẩn đoán.

2.2. Về Thực nghiệm (Baselines + Ablation)
Baselines BẮT BUỘC bổ sung (reviewer sẽ đòi):

LLM-confidence baselines mạnh: verbalized confidence, token logprob, P(True), và self-consistency k≥5 (hiện chỉ có raw↔KG agreement). CPR phải thắng các baseline reliability hiện đại, không chỉ value-only.
Generator mạnh: ≥1 model lớn (Qwen2.5-72B / GPT-4o-mini) để chứng minh selective answering không collapse và CPR còn giá trị khi base accuracy cao.
Retrieval: dense + hard metadata-filter + cross-encoder reranker (đã nêu trong ROADMAP.md nhưng chưa chạy đủ 3 dataset) — để chứng minh MMER không chỉ là "rerank trong pool công ty-năm".
Long-document thật: sửa DocFinQA hoặc thay bằng một benchmark long-context khác (TAT-DQA full-PDF). Không có row này, claim "structure giúp khi evidence bị chôn" chỉ dựa proxy.
Ablation BẮT BUỘC viết thêm code:

Role-assignment accuracy probe (vá Blind spot #2b) — đo trực tiếp độ chính xác calculation_plan.
Retrieval→NM correlation: chứng minh gain retrieval thực sự chuyển thành gain answer (hiện chưa có).
Oracle-ledger vs auto-ledger gap (F1 trích fact) — reviewer-required, cô lập đóng góp khỏi nhiễu trích xuất.
Per-answer-type breakdown (lookup vs computed) để củng cố TCEP Law 3 bằng số trên full set.
PHẦN 3 — BÁO CÁO KỸ THUẬT VỀ KIẾN TRÚC & TIẾN HOÁ HỆ THỐNG
3.1. Bản đồ phiên bản & kỹ thuật
Repo thể hiện một chuỗi tiến hoá 5 thế hệ, mỗi thế hệ sửa lỗi thế hệ trước (truy được qua ASSESSMENT.md → ROADMAP.md → 07_research_report.md → AAAI27_RESEARCH_PLAN.md):

Gen	Tên	Kỹ thuật cốt lõi	Kết cục (trung thực)
G1	GSR–CACL	Accounting-KG + Edge-aware GAT + constraint score (pairwise) + 3-stage curriculum (Identity→Structural→Joint) + CHAP negatives	KG/GAT đóng góp ≈ 0 (bảng row-major ⇒ 0 accounting edge khớp). Điểm đẹp đến từ metadata leak. Bị honest BM25 vượt.
G2	LEDGER-RAG v2	Entity embedding (SupCon) + GICS/alias ontology (E1/E2) + concept-coverage C3 + CACL InfoNCE (channel-aligned negatives) + Fact Ledger + generator + verifier + DPO/ORPO/GRPO scaffolding	MRR@3 0.71–0.74 nhưng candidate set leaky (recall=1.0 nhân tạo). Ontology/InfoNCE tốt nhưng cách dựng candidate là leak.
G3	Phase A	Honest BM25 + abbreviation sentinel + gated period/cell	0.6176 W.Avg honest. Negative result: company boost làm hại → BM25 cấp-doc bão hoà ~0.62.
G4	MMER	7 expert độc lập (lexical, dense, lateint/ColBERT-style, entity, concept, cell, graph) + fusion head học được (linear/mlp/gate-MoE), listwise InfoNCE, 5-fold CV	W.Avg MRR@3 0.722 honest (+0.121 vs BM25-in-pool). Phát hiện cốt lõi: fusion > mọi expert đơn lẻ.
G5	CPR + TCEP	Concept–Period–Role verifier + Type-Constrained Evidence Paths framework + selective answering (AURC, conformal-style) + multi-operand derivation + verify-then-reask	Lõi novelty. AUROC 0.66–0.76, selective coverage ↑5× (ConvFinQA), OOD FinanceBench transfer.
Kỹ thuật đã triển khai (xác nhận trong code): SFT/curriculum learning, contrastive metric learning (SupCon + InfoNCE), late interaction (ColBERT-style fact-level MaxSim), mixture-of-experts gating, listwise learning-to-rank, knowledge-graph construction (typed structure graph), symbolic verification (accounting identities), selective prediction/abstention, preference optimization scaffolding (DPO/ORPO/GRPO), verify-then-reask agentic policy.

3.2. Diễn giải lý thuyết — vì sao mỗi kỹ thuật tối ưu cho đặc thù bài toán
(a) Vì sao MMER (mixture-of-experts fusion) thay vì một retriever đơn?
EDA chứng minh không một biểu diễn đơn lẻ nào trị được cả 6 nguyên nhân thất bại đồng thời (07_research_report.md §2): bảng thống trị (dense kém vì số bị subword-tokenize), hard negative cùng công ty (intra-company similarity ~2.8× inter-company → "Apple 2019" vs "Apple 2020" gần trùng), context-sharing (1 vector/doc không đủ khi 92% doc phục vụ >1 câu hỏi), mismatch viết tắt (BM25 = 0 overlap với "GAAP" vs "generally accepted…"), lexical overlap thấp (Jaccard ~0.06), suy luận đa bước. Mỗi failure mode cần một inductive bias khác nhau → mỗi expert là một bias chuyên biệt; fusion head học per-query, per-dataset weighting. Đây chính là lý do toán học fusion thắng: các expert yếu (entity/cell/lateint, standalone <0.22) bổ sung trực giao cho lexical ở đúng các query lexical trượt — fusion khai thác phần bù đó (bằng chứng: trọng số học được w_cov, w_entity > 0 một cách nhất quán).

(b) Vì sao CPR (Concept–Period–Role) thay vì value-grounding?
Đây là lập luận lý thuyết mạnh nhất của bài, hình thức hoá trong TCEP. Một đáp án số a được "grounded" theo value-only nếu a xuất hiện ở đâu đó trong ledger. Nhưng trong tài chính, không gian giá trị dày đặc ⇒ value-match là sự trùng hợp thường xuyên (TCEP Law 1: support flag value-only kích hoạt trên 93–94% đáp án ⇒ gần như vô thông tin; accuracy không đơn điệu theo độ sâu path — 2-operand kém chính xác hơn 3-operand vì path ngắn là trùng hợp nhiều). Lời giải: một grounding chỉ đáng tin khi fact hỗ trợ đồng thời đúng concept (cùng khái niệm câu hỏi), đúng period (cùng kỳ), và đúng role (operand điền đúng vai old/new, part/total mà phép toán yêu cầu). TCEP Law 2 đo được: typed grounding chính xác gấp 2.8–4× untyped và replicate qua nhiều generator. Về mặt toán: CPR định nghĩa reliability R(a) = max_π typeconsistency(π)·damping(π) — chính là scorer của type-consistency trên typed evidence path, với damping phạt ambiguity (1/√#matches) và depth. Đây là lý do nguyên lý, không phải tinh chỉnh: nó suy ra grounding (depth-0 typed path) và derivation (path dài hơn) như trường hợp đặc biệt.

(c) Vì sao period dùng partial-credit (0.5+0.5·pc) thay vì hard filter?
Ablation thực nghiệm (AAAI27_RESEARCH_PLAN.md §3b) cho thấy period như nhân tố nhân tính cứng là có hại (P-alone < value-only trên cả 3; CPR<CR trên TAT-DQA) — vì trên bảng header đa cấp (TAT-DQA), "period mismatch" thường là lỗi parse, không phải sai kỳ thật. Giải pháp: gate tín hiệu period theo độ tin cậy trích period của chính tài liệu (_period_reliability trong cpr_verifier.py:108) và dùng partial-credit để period điều biến chứ không triệt tiêu một match concept+role tốt. Đây là một quyết định thiết kế driven bởi ablation, không phải đoán.

3.3. Cơ chế vận hành & thuật toán (pseudo-code)
Thuật toán 1 — MMER honest retrieval (5-fold CV fusion)


Input: query Q (đã bóc prefix metadata), corpus D, K experts E[1..K]
Output: ranked top-k documents

1.  meta ← extract_from_question(Q)        # year/company/concept CHỈ từ Q (honest contract)
2.  Pool ← ⋃ topN(E_lexical, E_dense, E_lateint over D)   # pool KHÔNG nhồi gold
3.  for each expert E_i in [lexical, dense, lateint, entity, concept, cell, graph]:
4.       s_i[d] ← E_i.score(Q, d)  for d in Pool
5.       s_i ← per-query min-max normalize(s_i)        # cùng thang đo
6.  F ← matrix[|Pool| × K] of s_i                      # feature matrix
7.  # 5-fold CV: query được chấm bởi fusion-head KHÔNG train trên fold của nó
8.  for fold in 1..5:
9.       head ← train_fusion(F_train, gold, loss=listwise_InfoNCE)   # linear/mlp/gate-MoE
10.      score[Q∈fold] ← head(F[Q])
11. return argsort_desc(score)[:k]
Thuật toán 2 — CPR verification (lõi novelty)


Input: prediction string p, FactLedger L, query Q, components ⊆ {concept,period,role,3op}
Output: CPRResult(confidence ∈ [0,1], level, supported, role, reasons)

1.  v ← extract_final_number(p);  if v = ∅: return no_answer
2.  q_concepts ← concepts_in_text(Q);  q_years ← target_year_pair(Q);  task ← infer_task_type(Q)
3.  period_rel ← fraction of L facts with parseable period
4.  pfloor ← (period_rel ≥ 0.4) ? 0 : 0.5            # gate period theo độ tin cậy parse
5.  # --- grounded_score: value khớp fact, có trọng số concept×period, phạt ambiguity ---
6.  for f in L where value_match(v, f):
7.       cc ← concept_consistency(f, q_concepts)      # 1.0 nếu canonical match, else token-overlap
8.       pc_eff ← 0.5 + 0.5·period_consistency(f, q_years, pfloor)   # partial-credit
9.       amb ← 1/√(#value_matches)
10.      grounded_score ← max(grounded_score, cc · pc_eff · amb)
11. # --- derivable_score: đáp án = công thức role-consistent trên operand consistent ---
12. plan ← calculation_plan(Q, select_facts(Q,L))     # gán role old/new, part/total...
13. if plan.answer ≈ v and plan.confidence ≥ 0.5:
14.      derivable_score ← plan.confidence
15. else: derivable_score ← role_consistent_pair_search(v, L, task, q_years)   # temporal/ratio/sum
16. # --- 3-operand fallback (chỉ khi không có gì mạnh hơn; confidence thấp vì coincidence) ---
17. if max(grounded,derivable) < 0.5 and derivation_depth(v, L, max_ops=3)=="3op":
18.      derivable_score ← 0.5
19. value_only_floor ← (vo_grounded) ? 0.25/√#matches : (vo_derivable ? 0.18 : 0)
20. confidence ← max(grounded_score, derivable_score, value_only_floor, 0.05)
21. return CPRResult(confidence, level=derive_level(...), supported = confidence ≥ 0.5, ...)
Thuật toán 3 — Inference policy: verify-then-reask + selective answering


1.  topK ← MMER_retrieve(Q)                            # Thuật toán 1
2.  L ← extract_fact_ledger(topK)                      # orientation-aware từ trường `table` sạch
3.  a_raw ← LLM_answer(Q, raw_table_context(topK))     # answer từ bảng thô TRƯỚC
4.  r ← verify_cpr(a_raw, L, Q)                         # Thuật toán 2
5.  if r.confidence < τ_reask:                          # ungrounded → re-ask
6.       evidence_paths ← structure_graph_paths(L, Q)  # "doc→table→row[X]→col[t]→value"
7.       a ← LLM_answer(Q, evidence_paths ∪ selected_facts);  r ← verify_cpr(a, L, Q)
8.  else: a ← a_raw
9.  if r.confidence < τ_abstain: return ABSTAIN        # selective answering
10. else: return (a, confidence=r.confidence)
PHẦN 4 — MA TRẬN LẬP LUẬN BIỆN CHỨNG (RATIONALE MATRIX)
Vấn đề của bài toán	Kỹ thuật đã chọn	Lý do khoa học	Minh chứng hiệu năng (đã thực thi)
Số bị subword-tokenize, dense kém trên bảng	Lexical BM25 + abbr sentinel	Sparse IR bền với token số; sentinel vá mismatch viết tắt	lexical standalone MRR@3 0.665/0.641/0.418 — best expert đơn
Context-sharing (1 doc, nhiều câu hỏi)	Late interaction (fact-level MaxSim, ColBERT-style)	1 vector/fact thay 1 vector/doc → chấm theo fact liên quan nhất	nâng trần recall ngoài vùng BM25 bỏ sót (pool recall 0.99)
Hard negative cùng công ty	Entity ontology (GICS/alias) + SupCon	Học e=Enc(metadata), cos thay so-khớp chuỗi	separation cùng-vs-khác thực thể 0.985; w_entity học được 0.72–0.76
Không expert nào trị hết 6 failure mode	Fusion head học được (linear/mlp/MoE-gate) + InfoNCE	Các bias trực giao bổ sung nhau; head học per-query weighting	W.Avg 0.722 honest (+0.121 vs BM25) — fusion > mọi đơn lẻ
Value-grounding over-fires (đáp sai trùng cell)	CPR (concept×period×role) trên typed graph	TCEP Law 1: path-existence ≠ reliability; Law 2: typed gấp 2.8–4×	AUROC 0.579→0.658 (FinQA), 0.640→0.755 (ConvFinQA); loại 66–85% false-grounded
Không biết đáp án nào đáng tin để triển khai	Selective answering (AURC + calibrator)	Abstention tại ngưỡng τ trên R(a)	ConvFinQA safe-coverage @≥70% acc 7%→36% (~5×); AURC 0.442→0.375
Đáp án multi-operand không chứng nhận được	Bounded 3-operand derivation	Trần certifiable bị giới hạn bởi độ sâu path	certifiable 2-op→3-op: FinQA 0.48→0.80, ConvFinQA 0.67→0.82
Generator yếu, KG có giúp không?	KG-audit prompt + verify-then-reask	Structure giúp model yếu nhiều hơn	Qwen-3B FinQA NM 0.117→0.188; verify-then-reask NM FinQA 0.278→0.295
Có phải artifact của T²-RAGBench?	OOD FinanceBench (real 10-K)	Transfer chứng minh tín hiệu là structural	AUROC 0.732→0.756; acc-when-supported 0.253→0.406; supported-wrong −73%
Kết luận tổng quan (bức tranh toàn cảnh)
Đây là một dự án nghiên cứu trưởng thành bất thường về kỷ luật khoa học — quy trình honest-evaluation, CV, bootstrap CI, kiểm leakage chủ động, và đặc biệt là văn hoá báo cáo kết quả âm đặt nó trên phần lớn submission AAAI về tính trung thực. Trục đóng góp đã được tinh chỉnh đúng hướng: từ "KG cải thiện accuracy" (sai, đã bị bác bỏ bằng chính số liệu của repo) sang "structure-grounded reliability + selective answering" (TCEP/CPR) — một trục trustworthy-ML hợp thời và có novelty thật.

Tuy nhiên, hiện trạng là borderline, chưa phải accept. Ba rào cản: (1) rủi ro tính toàn vẹn số liệu retrieval (file leaky on-disk phải được dọn ngay); (2) lõi novelty (Role) phụ thuộc heuristic chưa validate + trần trích xuất 47–69% bóp nghẹn hệ thống; (3) thiếu headline thuyết phục (generator mạnh + long-document thật + so với LLM-confidence hiện đại). Bản thân repo đã chẩn đoán chính xác cả ba — vấn đề là thực thi, không phải nhận thức.

Next Actions (ưu tiên cao nhất — về mặt lập trình)
#	Hành động	Vá blind spot	Cost
P0	Dọn integrity: xoá/ghi đè outputs/modular/*/modular.json leaky; chạy lại MMER honest config (--experts lexical,dense,entity,concept,cell,graph,lateint --cv 5, no company_pool); đảm bảo MỌI số trong bài là honest	#1	thấp (GPU)
P1	Role-assignment accuracy probe — đo trực tiếp độ chính xác calculation_plan (lõi CPR)	#2b	thấp
P2	Generator mạnh full-set (Qwen2.5-72B / GPT-4o-mini) → chứng minh selective answering không collapse + CPR còn giá trị	#3	GPU/API
P3	LLM-confidence baselines (verbalized conf, logprob, P(True), self-consistency k≥5) head-to-head với CPR	#3	trung bình
P4	Learned CPR typing (concept encoder tương phản thay token-overlap; mở rộng ontology >14%)	#2b	trung bình
P5	Fact-extraction F1 + oracle-vs-auto ledger gap (reviewer-required)	#2a	thấp
P6	Sửa DocFinQA / thêm 1 benchmark long-document thật + FinanceBench retrieval setting + bootstrap CI	empirical	trung bình
P7	Unambiguous operand attribution cho multi-operand path (open problem → nâng trần 47→80% + AUROC)	#2a	cao (research)
P8	Consolidate docs — gộp ~20 tài liệu mâu thuẫn thành 1 nguồn sự thật, tránh reviewer thấy claim xung đột	integrity	thấp
Khuyến nghị chiến lược cuối: viết bài hoàn toàn quanh TCEP/CPR + selective answering (title hiện tại "Know When You're Right" là đúng), dùng MMER 0.722 honest chỉ như supporting evidence ("chúng tôi giải được phần retrieval"), KHÔNG bán nó như đóng góp chính. Hoàn thành P0–P3 là điều kiện cần để chuyển từ borderline sang accept.