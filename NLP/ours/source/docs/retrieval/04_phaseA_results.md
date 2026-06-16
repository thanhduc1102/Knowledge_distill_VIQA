# Phase A — Abbreviation unification + honest metadata (training-free)

Hai tín hiệu honest mới, đo trên **toàn bộ** test set (n=5749), BM25-only (không e5 —
mọi arm thắng đều `w_dense=0`). Script: [`scripts/phaseA_retrieval.py`](../../scripts/phaseA_retrieval.py).
Module: [`gsr_cacl.retrieval.normalize`](../../src/gsr_cacl/retrieval/normalize.py) +
[`gsr_cacl.retrieval.self_query`](../../src/gsr_cacl/retrieval/self_query.py).

## Kết quả (MRR@3)

| arm | FinQA | ConvFinQA | TAT-DQA | **W.Avg** |
|---|---|---|---|---|
| stage1 (bm25+period+cell) | 0.6834 | 0.6602 | 0.4123 | 0.6155 |
| **+abbr** ✅ | **0.6864** | **0.6619** | **0.4149** | **0.6176** |
| +abbr+coyear(0.05) | 0.6853 | 0.6585 | 0.4127 | 0.6149 |
| +abbr+coyear(0.1) | 0.6795 | 0.6550 | 0.4101 | 0.6112 |
| +abbr+company_flat(0.05) | 0.6838 | 0.6560 | 0.4127 | 0.6131 |

`company-in-question rate` (honest, sau khi bóc prefix): FinQA 89.6%, ConvFinQA 89.8%, TAT-DQA 86.5%.

## Hai phát hiện

**1. Abbreviation unification GIÚP — nhỏ nhưng nhất quán & honest.** +abbr cải thiện cả 3
dataset trên mọi metric; rõ nhất ở **recall** (TAT-DQA R@5 0.611→0.621, R@3 0.537→0.544) —
đúng với chẩn đoán EDA #5 (mismatch hình thái là vấn đề *recall*). Cơ chế: gắn một
*sentinel token* chung cho mỗi khái niệm (vd `cct_gaap`) vào bất kỳ văn bản nào chứa **hoặc**
viết tắt **hoặc** dạng đầy đủ → "GAAP" trong câu hỏi khớp "generally accepted accounting
principles" trong tài liệu. Đối xứng, không học, chỉ cộng thêm (không bao giờ xóa match
lexical). → **Đã tích hợp mặc định vào `HybridBM25Retrieval(abbr_expand=True)`.**

**2. Boost metadata (company) — KHÔNG giúp honestly, kể cả company×year joint. Negative
result quan trọng.** Mọi arm có boost company (flat hoặc joint company∧year) đều *thấp hơn*
+abbr. Lý do cơ học: ở MRR@3≈0.68, BM25+period **đã** rút hết giá trị company/year (company
nằm sẵn trong 86–90% câu hỏi). Lỗi còn lại **không phải** "sai công ty/năm" mà là "đúng
công ty, đúng năm, **sai section/sub-table**" — vấn đề context-sharing (1 doc trả 3+ câu hỏi,
EDA #7). Một boost cộng đều cho mọi doc cùng (company,year) chỉ **làm phẳng** thứ hạng tinh
của BM25 *bên trong* nhóm đó → không tách được gold khỏi hard-negative gần nhất, chỉ gây hại.

## Hệ quả chiến lược

- Tầng fusion BM25 **đã bão hòa ở ~0.62 W.Avg honest**. Tín hiệu metadata cấp tài liệu
  (company/year) đã được khai thác hết bằng cách honest; thêm kênh metadata phẳng không vượt được.
- Điều này giải thích vì sao #1 leaderboard (metadata-BM25 @ 0.90) **không thể** chỉ là
  "BM25 + boost metadata" — phải là enrichment/matching ở **dưới mức tài liệu** (index-time
  hoặc within-doc).
- Đòn bẩy lên SOTA nằm ở **granularity dưới (company,year)**: fact/section-level late
  interaction (Phase C) — nơi lỗi residual thực sự sống. Phase A xác nhận bằng thực nghiệm
  rằng signal-engineering trên BM25 cấp-doc không còn dư địa.

## Tái lập

```bash
cd ours/source && PYTHONPATH=src python scripts/phaseA_retrieval.py --sample 0
# → outputs/phaseA/phaseA.json
```
