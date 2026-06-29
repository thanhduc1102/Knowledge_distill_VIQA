# Documentation Index — Structure-Grounded Financial Numerical QA (AAAI-27)

> **Bắt đầu từ đây.** `docs/` đã được dọn gọn: chỉ còn vài tài liệu *sống* (cập nhật liên tục) ở gốc;
> mọi báo cáo theo-phiên-bản cũ nằm trong [archive/](archive/) (không xoá, chỉ lưu trữ).

## Đọc theo thứ tự

| Tài liệu | Nội dung | Khi nào đọc |
|---|---|---|
| **[SYSTEM.md](SYSTEM.md)** | Bản đồ toàn trình retrieval→generation→reliability, phương pháp/kỹ thuật, tri thức tài chính, hiện trạng từng cấu phần | Hiểu hệ thống làm gì & ở đâu |
| **[RESULTS.md](RESULTS.md)** | Nguồn sự thật duy nhất cho MỌI kết quả (retrieval incl. metadata-SOTA, generation, reliability, ablation, OOD) + lệnh tái lập | Tra số liệu, kiểm chứng claim |
| **[FRAMEWORK_TCEP.md](FRAMEWORK_TCEP.md)** | Lý thuyết: Type-Constrained Evidence Paths (3 định luật) | Hiểu nền tảng của CPR |
| **[CONTRIBUTION_AUDIT.md](CONTRIBUTION_AUDIT.md)** | Đánh giá đóng góp theo từng khía cạnh + rủi ro reviewer | Trước khi viết/nộp paper |
| [literature/RELATED_WORK.md](literature/RELATED_WORK.md) | Kho tài liệu liên quan + nhật ký phát hiện | Định vị vs prior work |
| [retrieval/](retrieval/) · [first_idea/](first_idea/) | Lưu trữ chuyên đề retrieval & ý tưởng gốc | Tham chiếu chuyên sâu |
| [archive/](archive/) | Báo cáo cũ theo phiên bản (gen-1/gen-2, các bản tiếng Việt) | Truy vết lịch sử |

## Quy ước cập nhật (để không sprawl trở lại)
- **Mọi kết quả mới → thêm/sửa trong [RESULTS.md](RESULTS.md)** (không tạo file `RESULTS_v3.md`).
- **Mọi thay đổi kiến trúc/kỹ thuật → sửa [SYSTEM.md](SYSTEM.md)**.
- **Tài liệu/nghiên cứu liên quan → [literature/RELATED_WORK.md](literature/RELATED_WORK.md)** (kèm "gap it leaves").
- Báo cáo một-lần / theo-phiên-bản → để trong [archive/](archive/).

## Headline hiện tại (2026-06-29)

**Định vị bài:** *Know When You're Right, Cheaply — Cost-Efficient Structure-Grounded Reliability for
Financial Numerical QA.* Hai trục:
- **Retrieval (2 setting):** *honest* MMER 8-expert (meta từ câu hỏi) **W.Avg 0.798**; *provided* (metadata-aware,
  chuẩn benchmark) **W.Avg 0.873 — VƯỢT leaderboard #1 (~0.82)** (FinQA 0.914 / ConvFinQA 0.932) mà không dùng LLM frontier.
- **Retrieval → output:** đưa gold doc lên rank-1 nâng **Number-Match +0.34..+0.54** (retrieval là đòn bẩy của NM).
- **Reliability (lõi):** trên Gemini 2.5 Flash, **`cpr+verbalized` ở 2× chi phí vượt self-consistency ở 6×**;
  CPR bắt +9.5–16.5% confident-hallucination model-internal bỏ sót. Tổng quát hóa sang **DocFinQA long-document**.
- **Round-3 (3 đề xuất, ablation §7c):** metadata 3-trường ⊕ MMER = **THẮNG** (SOTA); learned operand attribution =
  THẮNG ở ceiling (deriv-hit 2–3×) nhưng trung tính khi cắm CPR; cross-encoder generic = THUA (cần fine-tune in-domain);
  learned concept encoder = coverage 22%→98% (precision 0.69).

**Đóng góp được chốt:** C1 cost-efficient reliability · C2 cơ chế bắt confident-hallucination ·
C3 phổ generator-strength · C4 fusion selective-answering. Negatives trung thực: verify-then-reask net-negative
trên model mạnh; học lại trọng số CPR không giúp; complementarity chắc chỉ ở ConvFinQA.

**Đòn bẩy tiếp theo (#1):** learned operand attribution (nâng trần FinQA 0.45→0.80 + sửa role-F1 0.5).
Danh sách đầy đủ: [RESULTS.md §8](RESULTS.md).

## Tái lập nhanh
```bash
cd ours/source && export PYTHONPATH=src   # GOOGLE_API_KEY / HF_TOKEN trong .env
python scripts/research/metadata_aware_bm25.py        # retrieval SOTA repro
python scripts/research/strong_reliability_eval.py    # reliability headline + cost frontier
```
Toàn bộ lệnh: [RESULTS.md §8](RESULTS.md).
