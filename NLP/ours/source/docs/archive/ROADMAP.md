# ROADMAP — phiên bản thí nghiệm & việc còn lại

Mỗi version là một thí nghiệm chạy được, lưu outputs riêng. ✅ = đã làm & kiểm chứng trong repo này.

## v1 — Retrieval đúng đắn (✅ core đã xong)
- ✅ Sửa B1 (không nạp checkpoint), B2 (lệch encoder), B3 (entity là so-khớp) bằng
  `methods/ledger_retrieval.py` + `entity/` (SupCon embedding).
- ✅ Metadata-aware candidate construction (tín hiệu vàng leaderboard).
- ✅ Equation-faithful constraint score (`scoring/constraint_score.py`).
- ✅ Ablation FinQA: dense 0.400 → FULL 0.742 MRR@3.
- ⏭ **Còn lại (GPU run):** chạy ablation đầy đủ cho ConvFinQA + TAT-DQA; thêm baseline
  `dense + hard metadata-filter + cross-encoder reranker` (reviewer 2 bắt buộc so sánh).

## v2 — Entity + CACL chuẩn (✅ core đã xong)
- ✅ Entity embedding học bằng SupCon (separation 0.985), dùng đủ metadata (sector/industry/symbol).
- ✅ 5 mẫu âm **channel-aligned, answer-invalidating** (`negative_sampler/channel_aligned.py`),
  CHAP-E thật.
- ⏭ **Còn lại:** train end-to-end LoRA encoder + GAT + scorer với các mẫu âm này (script `train.py`
  đã có khung 3-stage; nối thêm channel-aligned sampler); **ablation từng loại negative**
  (reviewer bắt buộc) để chứng minh mỗi loại train đúng 1 kênh.

## v3 — KG-for-Generator (✅ core đã xong)
- ✅ Fact Ledger trích `(concept,period,value,unit,scale,provenance)` từ trường `table` sạch
  (orientation-aware) — chạy đúng trên FinQA/ConvFinQA/TAT-DQA.
- ✅ Query-aware fact selection → đưa **đúng cell** cho generator (giải "context-sharing").
- ✅ Generator Qwen + extractive baseline; **Ledger Verifier** (grounding + arithmetic) annotation-free;
  **Number-Match** chuẩn leaderboard; `eval/pipeline.py` lưu toàn bộ artifacts.
- ⏭ **Còn lại (GPU run):** chạy Qwen2.5-3B/Qwen3-4B full test-set; báo cáo Number-Match end-to-end
  (retrieval-thật, không oracle) cho 3 dataset; **F1 trích fact + gap oracle-vs-auto** (reviewer bắt buộc).
- ⏭ **Nâng KG "ngữ nghĩa vững" hơn (HierFinRAG-style):** thêm node Section/Table/Row + cạnh
  text↔cell (mentions) + đẳng thức theo-hàng (Total-row = Σ component-rows) để equation-CS kích hoạt
  trên bảng row-major; liên kết YoY giữa các period.

## v4 — Preference / RLVR cho generator (✅ scaffolding đã xong)
- ✅ `training/preference.py`: reward verifier (R_answer + λ_g·grounding + λ_a·arithmetic, accuracy
  ưu tiên từ điển), build cặp DPO/ORPO, loss DPO/ORPO tự chứa, GRPO advantages.
- ⏭ **Còn lại (GPU run):** sinh cặp ưa thích từ Qwen (temperature>0) → DPO/ORPO; hoặc GRPO/RLVR
  với reward verifier; báo cáo cải thiện Number-Match so với zero-shot.

## Việc reviewer (review_1/2) yêu cầu — checklist để "đánh bại" nghiên cứu khác
- [ ] LEDGER vs **dense + hard metadata-filter + reranker** (story phải hơn "chỉ rank nội-bộ công ty-năm").
- [ ] **F1 trích fact** + **gap oracle-ledger vs auto-ledger** (Achilles' heel).
- [ ] **Ablation từng loại negative** (chứng minh channel-alignment).
- [ ] **Latency** metadata-aware candidate vs 2-stage/ColBERT (đo trên các cỡ corpus).
- [ ] Probe §2 (acc(topic) ≫ acc(entity,year)) để chứng minh động lực decompose.
- [ ] Diễn ngôn: "principle, not architecture" — relevance là *fact-indexed*, không *document-indexed*.

## Lệnh tiện ích
```bash
cd ours/source && export HF_DATASETS_OFFLINE=1 PYTHONPATH=src
for d in finqa convfinqa tatqa; do python scripts/retrieval_ablation.py --dataset $d --sample 300; done
python -m gsr_cacl.eval.pipeline --dataset finqa --stage all --generator hf --gen-model Qwen/Qwen2.5-3B-Instruct
```
