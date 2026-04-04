# VLSP 2025 - Kaggle Deployment Guide

Hướng dẫn chi tiết triển khai Knowledge Distillation Pipeline trên Kaggle.

---

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Upload code & data lên Kaggle](#2-upload-lên-kaggle)
3. [Chạy notebook trên Kaggle](#3-chạy-notebook-trên-kaggle)
4. [Xử lý lỗi thường gặp](#4-xử-lý-lỗi-thường-gặp)
5. [Ghi chú phiên bản](#5-ghi-chú-phiên-bản)

---

## 1. Tổng quan

### Pipeline flow
```
data_prep → teacher_distill → SFT → GRPO → inference → evaluate
```

### Yêu cầu
- **GPU**: RTX 6000 Pro 96GB (recommended), T4, P100, hoặc A100
- **Kaggle Account**: Có API Key (Classic API Key, KHÔNG phải KGAT token)
- **Local machine**: Python 3.10+, pip, git

### GPU Profiles
| Profile | Teacher | Student | Flash Attn | dtype |
|---|---|---|---|---|
| `p100_16gb` | Qwen3.5-4B (4bit) | Qwen3.5-0.8B | ❌ | fp16 |
| `rtx6000_96gb` | Qwen3.5-27B | Qwen3.5-4B | ✅ | bf16 |

---

## 2. Upload lên Kaggle

### Bước 2.1: Cấu hình credentials

1. Vào https://www.kaggle.com/settings → API section
2. Click **"Create New Token"** → download `kaggle.json`
3. Copy `username` và `key` vào file `.env`:

```bash
# .env (trong thư mục gốc project)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=abc123def456...   # 32 ký tự hex, KHÔNG phải KGAT_...
```

> ⚠️ **QUAN TRỌNG**: Key phải là Classic API Key (32 hex chars). Nếu key bắt đầu bằng `KGAT_`, đó là OAuth token — KHÔNG hoạt động. Hãy tạo lại API Token.

### Bước 2.2: Upload pipeline code + datasets

```bash
# Upload pipeline code (pipeline/, src/, configs/)
python scripts/kaggle_upload.py --upload-code

# Upload datasets
python scripts/kaggle_upload.py --upload-data

# Upload offline Python wheels (optional - nếu Kaggle không có internet)
python scripts/kaggle_upload.py --upload-wheels

# Upload tất cả cùng lúc
python scripts/kaggle_upload.py --all
```

### Bước 2.3: Push notebook lên Kaggle (KHUYẾN NGHỊ ⭐)

**Đây là cách đơn giản nhất** — notebook sẽ được tạo trên Kaggle với tất cả cells đã tách sẵn, dataset inputs đã cấu hình:

```bash
# Push notebook trực tiếp lên Kaggle
python scripts/kaggle_upload.py --push-notebook

# Hoặc upload code + push notebook cùng lúc
python scripts/kaggle_upload.py --upload-code --push-notebook
```

Sau khi push thành công, bạn sẽ thấy link notebook. Chỉ cần mở link đó trên Kaggle và chạy.

> **Nếu `kaggle` CLI chưa cài**: `pip install kaggle`

### Bước 2.4: Upload models (optional)

```bash
# Download models trước
huggingface-cli download Qwen/Qwen3.5-27B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-27B
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-4B

# Upload
python scripts/kaggle_upload.py --upload-models --model-dir ./kaggle_offline_package/models
```

Nếu không upload models, notebook sẽ tự download từ HuggingFace (cần internet trên Kaggle).

### Bước 2.5: Kiểm tra uploads

```bash
python scripts/kaggle_upload.py --check
```

---

## 3. Chạy notebook trên Kaggle

### Cách 1: Dùng `--push-notebook` (KHUYẾN NGHỊ ⭐)

1. Chạy `python scripts/kaggle_upload.py --upload-code --push-notebook`
2. Mở link notebook trên Kaggle (hiển thị sau khi push thành công)
3. Click **"Edit"** để mở notebook editor
4. **Settings** (góc phải):
   - Accelerator → **GPU RTX 6000 Pro** (hoặc T4/P100)
   - Internet → ON (nếu cần download models)
5. **Add Input Datasets** (nếu chưa có):
   - `thanhduc1108/vlsp2025-kd-pipeline` ← Pipeline code
   - `thanhduc1108/finqa-en` ← FinQA dataset
   - `thanhduc1108/vinumericalqa-private` ← ViNumQA dataset
   - `thanhduc1108/vlsp2025-kd-wheels` ← Wheels (optional)
6. Chạy từng cell từ trên xuống dưới

### Cách 2: Import .ipynb thủ công

1. Generate notebook file:
   ```bash
   python scripts/generate_notebook.py
   ```
   File `.ipynb` nằm tại: `kaggle/kaggle_kd_notebook.ipynb`

2. Vào Kaggle → **New Notebook**
3. **File** → **Upload Notebook** → chọn file `kaggle_kd_notebook.ipynb`
4. Thêm datasets (xem bước 5 ở Cách 1)
5. Bật GPU trong Settings
6. Chạy từng cell

### Cách 3: Copy-paste (KHÔNG KHUYẾN NGHỊ)

> ⚠️ Cách này dễ lỗi vì tất cả code nằm trong 1 cell.

Nếu vẫn muốn dùng `kaggle_kd_notebook.py`:
1. Mở `kaggle/kaggle_kd_notebook.py`
2. Copy **TOÀN BỘ** nội dung file
3. Paste vào **1 cell** của Kaggle notebook
4. Chạy cell đó

**Lưu ý quan trọng khi copy-paste:**
- Đảm bảo copy đúng phiên bản mới nhất (sau khi `--upload-code`)
- Đảm bảo đã thêm tất cả input datasets trước khi chạy
- Nếu lỗi `ModuleNotFoundError: No module named 'pipeline'`, kiểm tra:
  - Dataset `vlsp2025-kd-pipeline` đã được thêm chưa?
  - Thư mục `/kaggle/input/vlsp2025-kd-pipeline/pipeline/` có tồn tại không?

### Cấu trúc cells trong notebook

| Cell | Phase | Mô tả |
|------|-------|-------|
| 0 | Pre-flight | Kiểm tra tất cả input datasets có đủ không |
| 1 | Setup | Cài đặt dependencies (pip) |
| 2 | Setup | Copy code, tạo thư mục, symlink datasets |
| 3 | Setup | Kiểm tra GPU, detect GPU profile |
| 4 | Setup | Tìm model paths (offline → HuggingFace fallback) |
| 5 | Config | Tạo pipeline configuration |
| 6 | Phase 1 | Data Preparation |
| 7 | Phase 2 | Teacher Distillation |
| 8 | Phase 3 | SFT Training |
| 9 | Phase 4 | GRPO Training |
| 10 | Phase 5 | Inference + Majority Voting |
| 11 | Phase 6 | Evaluation (EA + PA) |
| 12 | Optional | Baseline Comparison (zero-shot) |
| 13 | Results | Results Summary Table |
| 14 | Output | Save all outputs |

---

## 4. Xử lý lỗi thường gặp

### Lỗi 1: `ModuleNotFoundError: No module named 'pipeline'`

**Nguyên nhân**: Dataset `vlsp2025-kd-pipeline` chưa được thêm vào notebook.

**Fix**:
1. Kiểm tra: Cell 0 (Pre-flight) phải báo `✓ FOUND` cho `vlsp2025-kd-pipeline`
2. Nếu báo `✗ MISSING` → thêm dataset: **+ Add Data** → tìm `vlsp2025-kd-pipeline`
3. Sau khi thêm, **chạy lại từ Cell 0**

### Lỗi 2: `401 Client Error` khi upload

**Nguyên nhân**: KAGGLE_KEY là KGAT OAuth token.

**Fix**:
1. Vào https://www.kaggle.com/settings → API
2. Click "Expire API Token" → "Create New Token"
3. Download `kaggle.json`, copy key vào `.env`

### Lỗi 3: `model_type 'qwen3_5' not recognized`

**Nguyên nhân**: `transformers` quá cũ (cần ≥5.0).

**Fix**: Cell 1 sẽ tự cài `transformers>=5.0`. Nếu vẫn lỗi, thêm cell:
```python
!pip install -q transformers>=5.0 --force-reinstall
```

### Lỗi 4: CUDA OOM (Out of Memory)

**Fix**: Giảm batch size hoặc dùng quantization:
- `p100_16gb` profile tự động dùng 4-bit quantization
- Hoặc tắt GRPO phase (bỏ qua Cell 9)

### Lỗi 5: No GPU detected

**Fix**: Settings → Accelerator → chọn GPU (T4/P100/RTX 6000)

### Lỗi 6: Flash Attention not available

Không sao, pipeline tự fallback sang eager attention. Flash Attention chỉ tăng tốc.

---

## 5. Ghi chú phiên bản

### Dependencies đã test
| Package | Version | Ghi chú |
|---------|---------|---------|
| Python | 3.10.x | Kaggle mặc định |
| PyTorch | 2.5.x+ | Kaggle có sẵn |
| transformers | **≥ 5.0** | **BẮT BUỘC** cho Qwen3.5 |
| peft | ≥ 0.18 | LoRA training |
| bitsandbytes | ≥ 0.49 | 4-bit quantization |
| accelerate | ≥ 1.0 | Device mapping |
| trl | ≥ 1.0 | GRPO training |
| flash-attn | ≥ 2.5 | Optional, RTX/A100 only |

### Thời gian ước tính (RTX 6000 Pro 96GB)
| Phase | Thời gian |
|-------|-----------|
| Data Prep | ~1 min |
| Teacher Distillation | ~60-120 min |
| SFT Training | ~30-60 min |
| GRPO Training | ~60-120 min |
| Inference | ~20-40 min |
| Evaluation | ~1 min |
| **Tổng** | **~3-6 giờ** |

> Kaggle cho phép chạy tối đa **12 giờ/session**.

---

## Tóm tắt quy trình

```
[Máy local - có internet]              [Kaggle - GPU]
     │                                      │
     ├─ pip install kaggle kagglehub        │
     ├─ Cấu hình .env                      │
     ├─ python scripts/kaggle_upload.py     │
     │   --upload-code                      │
     │   --upload-data                      │
     │   --push-notebook ←──────── notebook xuất hiện trên Kaggle
     │                                      │
     │                                      ├─ Mở notebook link
     │                                      ├─ Bật GPU trong Settings
     │                                      ├─ Thêm datasets (nếu cần)
     │                                      ├─ Chạy từng cell
     │                                      └─ Download results
```
