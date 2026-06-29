# Kết quả đã kiểm chứng (verified, không bịa)

> Mọi số dưới đây chạy **thật** trên T²-RAGBench (cache sẵn) với embedding
> `intfloat/multilingual-e5-large-instruct`, 2× Tesla T4. Tái lập bằng
> `scripts/retrieval_ablation.py` và `gsr_cacl.eval.pipeline`.

## 1. Ablation retrieval — FinQA (300 query, corpus 2,789)

> Artifact tái lập: `outputs/ledger_eval/finqa_ablation/ablation.{json,md}` (committed).
> (Một lần chạy 200-query trước đó cho FULL MRR@3 0.742 — cùng khoảng.)

| arm | MRR@3 | R@1 | R@3 | R@5 | NDCG@3 |
|---|---|---|---|---|---|
| A) dense only (e5) | 0.381 | 0.300 | 0.483 | 0.547 | 0.407 |
| B) dense + equationCS | 0.381 | 0.300 | 0.483 | 0.547 | 0.407 |
| C) dense + trained entity-emb (rerank) | 0.669 | 0.573 | 0.787 | 0.813 | 0.699 |
| **D) FULL: entity-emb + metadata-filter** | **0.732** | **0.617** | **0.873** | **0.933** | **0.768** |

**Đọc kết quả:**
- Dense-only **0.381** ≈ baseline e5 trên leaderboard ⇒ pipeline đã hiệu chỉnh đúng.
- Entity-embedding học bằng SupCon: **+0.288 MRR@3** (rerank thuần).
- Thêm metadata-aware candidate construction: **+0.351** so dense → **0.732**, **R@3 0.873, R@5 0.933**.
- Cao hơn con số contribution1 tự công bố (FinQA MRR@3 0.638) và vượt xa baseline retrieval
  của leaderboard (QwQ-32B + Hybrid BM25: FinQA MRR@3 0.398). (Đỉnh leaderboard GPT-5.4 +
  Metadata-aware BM25 đạt 0.903 nhưng dùng LLM mạnh + BM25 tinh chỉnh; LEDGER-RAG ở đây
  retrieval-only, encoder generic, hoàn toàn tái lập.)

> Kết quả 300-query lưu ở `outputs/ledger_eval/finqa_ablation/ablation.{json,md}`.

## 2. Entity embedding (SupCon) — kiểm chứng phân tách thực thể

Train trên metadata FinQA (3,000 mẫu, 12 epoch, CPU vài giây):

| cặp | cosine trung bình |
|---|---|
| cùng company + cùng year | **1.000** |
| khác thực thể | **0.015** |
| **separation** | **0.985** |

→ Hiện thực đúng ý tưởng "metadata embedding làm điểm số" của contribution1 (khác bản cũ chỉ so chuỗi).

## 3. End-to-end generation — Number-Match (chuẩn leaderboard)

Sàn không-GPU (extractive) chỉ để pipeline chạy được; **Qwen là path chính**.

| cấu hình | NumberMatch (oracle doc, 12 mẫu FinQA) |
|---|---|
| extractive (rule, sàn) | thấp (3–20%) — chỉ là floor |
| **Qwen2.5-0.5B-Instruct** (model tí hon, chỉ để validate code path) | **0.333** |
| Qwen2.5-3B / Qwen3-4B (khuyến nghị) | *cần GPU run — kỳ vọng cao hơn nhiều* |

Verifier reward trung bình ~0.46–0.56 (đa số đáp án grounded/derivable từ ledger).

## 4. KG-for-Generator (Fact Ledger) — kiểm chứng định tính

Ví dụ FinQA (câu net-change 2014→2015): ledger trích đúng
`2014 net revenue [2014]=$5735 (in millions)`, `2015 net revenue [2015]=$5829` → generator nhận
**đúng 2 cell** cần thiết (đáp án 94 = 5829−5735). TAT-DQA: `Asia Pacific [2019]=7.4 > [2018]=4.4`
→ trả đúng năm 2019.

## 5. Smoke tests
`python tests/test_ledger_rag.py` → **7/7 PASS** (numeric, ledger, verifier, equation-CS,
channel-aligned negatives, entity SupCon separation, preference reward + GRPO).
