# LEDGER-RAG: Fact-Ledger Retrieval & Generation for Financial Documents

LEDGER-RAG là bản nâng cấp của GSR–CACL, biến **đồ thị tri thức (Fact Ledger)** thành
substrate dùng chung cho **cả retrieve lẫn generate**, sửa các lỗi nghiêm trọng của bản cũ
(xem [ASSESSMENT.md](ASSESSMENT.md)) và bổ sung pha sinh + verifier + tối ưu ưa thích.

Tất cả module nằm trong `src/gsr_cacl/` (mở rộng package hiện có, tái dùng KG/GAT/template).

```
                      ┌──────────────── Fact Ledger (KG dùng chung) ────────────────┐
 query ──► retrieve ──┤  metadata-aware candidates → α·text + β·cos(e_Q,e_D) + γ·CS  ├──► top-K
                      │            (entity embedding học bằng SupCon)                │
                      └─────────────────────────────────────────────────────────────┘
 top-K docs ──► extract Fact Ledger (concept,period,value,unit,scale) ──► query-aware fact selection
            ──► generator (Qwen / extractive) ──► Ledger Verifier (grounding+arithmetic) ──► Number-Match
            ──► (optional) DPO/ORPO/GRPO bằng reward của verifier
```

## Thành phần & file

| Module | File | Vai trò |
|--------|------|--------|
| Fact Ledger | `ledger/{fact,extract,select,numeric}.py` | Trích `(concept, entity, period, value, unit, scale, provenance)` từ trường `table` sạch; chọn fact theo query để đưa **đúng cell** cho generator |
| Entity embedding | `entity/{encoder,supcon,train}.py` | `e=Enc(metadata)` học bằng **SupCon**; `cos(e_Q,e_D)` là điểm retrieval (thay cho so-khớp chuỗi) |
| Equation-faithful CS | `scoring/constraint_score.py::compute_equation_constraint_score` | Chấm điểm **cả phương trình** `Total=Σω·operand` thay vì từng cặp |
| Metadata-aware retrieval | `methods/ledger_retrieval.py` | Candidate = (dense top-N) ∪ (same-company±year); `s=α·text+β·entity+γ·CS` |
| CACL negatives | `negative_sampler/channel_aligned.py` | 5 mẫu âm **channel-aligned, answer-invalidating** (CHAP-E thật) |
| Generation | `generation/{generator,prompts,verifier,metrics}.py` | Qwen + extractive baseline; Ledger Verifier; Number-Match |
| Preference/RL | `training/preference.py` | reward verifier → DPO/ORPO + GRPO advantages |
| Eval harness | `eval/pipeline.py` | retrieve→ledger→generate→verify, lưu toàn bộ outputs |
| Tests | `tests/test_ledger_rag.py` | 7 smoke test, không cần GPU/mạng |

## Chạy thử (đã kiểm chứng trên dataset cache sẵn)

```bash
cd ours/source
export HF_DATASETS_OFFLINE=1 PYTHONPATH=src

# 0) smoke tests (vài giây, CPU)
python tests/test_ledger_rag.py

# 1) ablation retrieval — tách đóng góp từng tín hiệu
python scripts/retrieval_ablation.py --dataset finqa --sample 300

# 2) end-to-end (retrieval + generation). Extractive = không cần GPU/LLM:
python -m gsr_cacl.eval.pipeline --dataset finqa --sample 200 --stage all --generator extractive

# 3) end-to-end với Qwen (khuyến nghị Qwen2.5-3B-Instruct / Qwen3-4B, vừa 1×T4):
python -m gsr_cacl.eval.pipeline --dataset finqa --sample 200 --stage all \
       --generator hf --gen-model Qwen/Qwen2.5-3B-Instruct
```

Outputs lưu ở `outputs/ledger_eval/<run>/`: `config.json`, `retrieval_metrics.json`,
`retrieval_topk.jsonl` (kèm `evidence_block` cho generator), `generation_metrics.json`,
`predictions.jsonl`, `summary.json`.

## Kết quả đã kiểm chứng (thực, không bịa)

Xem [RESULTS.md](RESULTS.md). Tóm tắt FinQA (e5-large, 300 query, corpus 2789):
dense-only **MRR@3 0.381** → **FULL (entity-emb + metadata-filter) MRR@3 0.732, R@3 0.873, R@5 0.933**
(+0.35). Sinh đáp án end-to-end đo bằng **Number-Match** (chuẩn leaderboard); Qwen là path chính,
extractive là sàn không-GPU. Artifact: `outputs/ledger_eval/finqa_ablation/`.

## Phụ thuộc thêm (tùy chọn)
- `rank_bm25` — chỉ cần cho biến thể HybridGSR (RRF). Không bắt buộc.
- `trl` — nếu có sẽ dùng `DPOTrainer`; không có thì `training/preference.py` tự chứa DPO/ORPO.
- Generator HF cần `transformers` (đã có) + tải Qwen (mạng HF).

## Hạn chế đã ghi nhận (trung thực)
- **Equation-CS chưa giúp FinQA**: template khớp theo header-cột nhưng bảng FinQA là *row-major*
  (concept ở hàng) → template không kích hoạt (acc_edges=0). Hướng sửa: trích đẳng thức **theo hàng**
  từ Fact Ledger (Total-row = Σ component-rows). Tín hiệu metadata/entity hiện gánh phần chính
  (đúng như leaderboard ngụ ý: metadata là tín hiệu vàng).
- Header ConvFinQA lỗi/đa năm → gán period đôi khi nhiễu; giá trị vẫn được trích đúng.
- Training quy mô lớn (LoRA encoder, GAT, DPO/GRPO) là bước GPU riêng — script đã sẵn sàng.
