# VLSP 2025 - Kaggle Deployment Guide (Offline RTX 6000 Pro 96GB)

Hướng dẫn chi tiết từng bước để triển khai baseline trên Kaggle RTX 6000 Pro 96GB **không có internet**.

---

## Mục lục
1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Chuẩn bị offline package (có internet)](#2-chuẩn-bị-offline-package)
3. [Upload lên Kaggle](#3-upload-lên-kaggle)
4. [Chạy notebook trên Kaggle](#4-chạy-notebook-trên-kaggle)
5. [Xử lý lỗi thường gặp](#5-xử-lý-lỗi-thường-gặp)
6. [Ghi chú quan trọng về phiên bản](#6-ghi-chú-phiên-bản)

---

## 1. Tổng quan hệ thống

### Models & VRAM
| Model | Params | Quantization | VRAM ước tính | Candidates |
|-------|--------|-------------|---------------|------------|
| Qwen3.5-4B | 4B | None (BF16) | ~9 GB | 15 |
| Qwen3.5-9B | 9B | None (BF16) | ~19 GB | 15 |
| Qwen3.5-27B | 27B | None (BF16) | ~55 GB | 10 |
| Qwen3.5-35B-A3B | 35B MoE | None (BF16) | ~72 GB | 15 |
| Qwen3.5-122B-A10B | 122B MoE | 4-bit NF4 | ~70 GB | 10 |

### Pipeline flow
```
Load Model → Load Test Data → Generate N candidates per sample 
→ Majority Vote → Save predictions → Evaluate EA/PA
```

---

## 2. Chuẩn bị offline package

**Yêu cầu**: Máy có internet, Python 3.10+, pip, ~500GB disk (cho model weights)

### Bước 2.1: Clone repo
```bash
git clone https://github.com/thanhduc1102/Knowledge_distill_VIQA.git
cd Knowledge_distill_VIQA
```

### Bước 2.2: Download dataset
```bash
bash scripts/download_data.sh
```

### Bước 2.3: Download models và pip wheels
```bash
# Tạo offline package (~500GB cho tất cả models)
bash scripts/prepare_kaggle_offline.sh ./kaggle_offline_package

# Hoặc download từng model riêng lẻ:
pip install huggingface_hub[cli]

# Download model nhỏ trước để test
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-4B

# Download models lớn hơn
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-9B
huggingface-cli download Qwen/Qwen3.5-27B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-27B
huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-35B-A3B
huggingface-cli download Qwen/Qwen3.5-122B-A10B --local-dir ./kaggle_offline_package/models/Qwen_Qwen3.5-122B-A10B
```

### Bước 2.4: Download pip wheels cho offline install
```bash
mkdir -p ./kaggle_offline_package/wheels

# Wheels cần thiết (Kaggle thường đã có sẵn torch)
pip download -d ./kaggle_offline_package/wheels \
  transformers>=5.4.0 \
  accelerate>=0.34.0 \
  peft>=0.13.0 \
  bitsandbytes>=0.44.0 \
  safetensors>=0.4.0 \
  sentencepiece>=0.2.0 \
  huggingface_hub>=1.0.0 \
  tokenizers>=0.20.0
```

### Bước 2.5: Verify package
```bash
ls -lh kaggle_offline_package/
# models/   - Tất cả model weights
# wheels/   - Pip packages
# data/     - ViNumQA dataset
```

---

## 3. Upload lên Kaggle

### Phương pháp 1: Kaggle API (khuyến nghị cho files lớn)
```bash
pip install kaggle

# Tạo dataset metadata
cat > kaggle_offline_package/dataset-metadata.json << 'EOF'
{
  "title": "VLSP2025 Offline Package",
  "id": "YOUR_USERNAME/vlsp2025-offline",
  "licenses": [{"name": "MIT"}]
}
EOF

# Upload (lần đầu)
kaggle datasets create -p kaggle_offline_package/

# Update (lần sau)
kaggle datasets version -p kaggle_offline_package/ -m "Update models"
```

### Phương pháp 2: Upload qua Kaggle Web UI
1. Vào https://www.kaggle.com/datasets → **New Dataset**
2. Upload thư mục `kaggle_offline_package/`
3. Đặt tên: `vlsp2025-offline`

### Phương pháp 3: Upload từng model riêng (để tránh giới hạn 100GB/dataset)
Tạo nhiều dataset:
- `vlsp2025-code-data`: Code + data + wheels (~100MB)
- `vlsp2025-qwen35-4b`: Model Qwen3.5-4B (~8GB)
- `vlsp2025-qwen35-9b`: Model Qwen3.5-9B (~18GB)
- ... (mỗi model 1 dataset)

> **LƯU Ý**: Kaggle giới hạn ~100GB/dataset. Với models lớn (27B+), bạn CẦN tách ra nhiều datasets.

---

## 4. Chạy notebook trên Kaggle

### Bước 4.1: Tạo notebook
1. Vào Kaggle → **New Notebook**
2. Settings:
   - **Accelerator**: GPU RTX 6000 Pro 96GB
   - **Internet**: OFF
   - **Persistence**: Files

### Bước 4.2: Thêm datasets
1. Click **Add Data** → Add dataset `vlsp2025-offline`
2. Nếu tách nhiều datasets, thêm tất cả

### Bước 4.3: Chạy code

**Cell 1 - Setup**:
```python
import subprocess, sys, os

# Path tới offline package
OFFLINE = "/kaggle/input/vlsp2025-offline"

# Install dependencies offline (nếu cần)
wheels = os.path.join(OFFLINE, "wheels")
if os.path.exists(wheels):
    subprocess.run([sys.executable, "-m", "pip", "install", 
                    "--no-index", "--find-links", wheels,
                    "transformers", "accelerate", "peft", "bitsandbytes"],
                   capture_output=True)

# Copy code
os.system(f"cp -r {OFFLINE}/code/* /kaggle/working/")
os.chdir("/kaggle/working")
sys.path.insert(0, "/kaggle/working")

# Copy data
os.makedirs("data/receive", exist_ok=True)
os.system(f"cp {OFFLINE}/data/receive/*.json data/receive/")
```

**Cell 2 - Verify**:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import transformers
print(f"Transformers: {transformers.__version__}")

import json
with open("data/receive/test.json") as f:
    test_data = json.load(f)
print(f"Public test: {len(test_data)} samples")
```

**Cell 3 - Run baseline (ví dụ Qwen3.5-4B)**:
```python
# Chỉnh MODEL_PATH theo tên dataset bạn upload
MODEL_PATH = "/kaggle/input/vlsp2025-offline/models/Qwen_Qwen3.5-4B"
# Hoặc nếu tách dataset: "/kaggle/input/vlsp2025-qwen35-4b"

from baseline.run_baseline import run_single_baseline
from baseline.config import RTX6000_MODELS, BaselineModelConfig

# Override model path to use local weights
import baseline.config as bcfg
original_id = bcfg.RTX6000_MODELS["qwen3.5-4b"].model_id
bcfg.RTX6000_MODELS["qwen3.5-4b"].model_id = MODEL_PATH

result = run_single_baseline(
    "qwen3.5-4b",
    gpu_profile="rtx6000_96gb",
    data_path="data/receive/test.json",
    output_dir="outputs/baseline",
)

# Restore
bcfg.RTX6000_MODELS["qwen3.5-4b"].model_id = original_id
print(f"EA: {result['execution_accuracy']:.2%}")
print(f"PA: {result['program_accuracy']:.2%}")
```

**Cell 4 - Chạy tất cả models tuần tự**:
```python
# Mapping model key → local path
MODEL_PATHS = {
    "qwen3.5-4b": "/kaggle/input/vlsp2025-offline/models/Qwen_Qwen3.5-4B",
    "qwen3.5-9b": "/kaggle/input/vlsp2025-offline/models/Qwen_Qwen3.5-9B",
    "qwen3.5-27b": "/kaggle/input/vlsp2025-offline/models/Qwen_Qwen3.5-27B",
    "qwen3.5-35b-a3b": "/kaggle/input/vlsp2025-offline/models/Qwen_Qwen3.5-35B-A3B",
    "qwen3.5-122b-a10b": "/kaggle/input/vlsp2025-offline/models/Qwen_Qwen3.5-122B-A10B",
}

import baseline.config as bcfg
results = []

for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f"SKIP {key}: path not found")
        continue
    
    # Override model_id to local path
    bcfg.RTX6000_MODELS[key].model_id = path
    
    try:
        r = run_single_baseline(key, "rtx6000_96gb", "data/receive/test.json", "outputs/baseline")
        results.append(r)
    except Exception as e:
        print(f"ERROR {key}: {e}")
        results.append({"model": key, "error": str(e)})

# Print summary
for r in results:
    if "error" in r:
        print(f"{r['model']}: ERROR")
    else:
        print(f"{r['model']}: EA={r['execution_accuracy']:.2%} PA={r['program_accuracy']:.2%}")
```

### Thời gian ước tính trên RTX 6000 Pro 96GB (497 samples)
| Model | Thời gian/sample | Tổng (10 candidates) |
|-------|------------------|---------------------|
| Qwen3.5-4B | ~5s | ~40 min |
| Qwen3.5-9B | ~10s | ~80 min |
| Qwen3.5-27B | ~30s | ~250 min |
| Qwen3.5-35B-A3B | ~8s (MoE fast) | ~65 min |
| Qwen3.5-122B-A10B | ~25s (4-bit) | ~200 min |

> Kaggle cho phép chạy tối đa **12 giờ/session**. Bạn có thể chạy ~2-3 models lớn mỗi session.

---

## 5. Xử lý lỗi thường gặp

### Lỗi 1: `model_type 'qwen3_5' not recognized`
**Nguyên nhân**: Phiên bản `transformers` quá cũ.
**Fix**: Upgrade transformers
```python
# Trong wheels/ phải có transformers>=5.4.0
subprocess.run([sys.executable, "-m", "pip", "install", 
                "--no-index", "--find-links", "/kaggle/input/vlsp2025-offline/wheels",
                "transformers>=5.4.0"])
```

### Lỗi 2: CUDA OOM (Out of Memory)
**Nguyên nhân**: Model quá lớn cho GPU.
**Fix**: Enable 4-bit quantization
```python
bcfg.RTX6000_MODELS["qwen3.5-27b"].quantization = "4bit"
```

### Lỗi 3: `bitsandbytes` not found
**Fix**: Đảm bảo wheel file có trong package
```python
subprocess.run([sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", "/kaggle/input/vlsp2025-offline/wheels",
                "bitsandbytes"])
```

### Lỗi 4: Flash Attention not available
**Khuyến nghị**: Không bắt buộc nhưng giúp tăng tốc inference.
```python
# Nếu không có flash-attn, tắt đi:
bcfg.RTX6000_MODELS["qwen3.5-9b"].use_flash_attention = False
```
Pipeline sẽ tự động fallback sang eager attention.

### Lỗi 5: `tokenizers` version conflict
**Fix**: Đảm bảo pin đúng version
```python
subprocess.run([sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", "/kaggle/input/vlsp2025-offline/wheels",
                "tokenizers>=0.20.0", "huggingface_hub>=1.0.0"])
```

### Lỗi 6: Model weights incomplete/corrupted
**Fix**: Re-download model weights với `--force-download`
```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./models/Qwen_Qwen3.5-4B --force-download
```

---

## 6. Ghi chú phiên bản

### Phiên bản đã test
| Package | Version | Ghi chú |
|---------|---------|---------|
| Python | 3.10.x / 3.12.x | Kaggle dùng 3.10 |
| PyTorch | 2.5.x - 2.8.x | Kaggle thường có sẵn |
| transformers | **≥ 5.4.0** | **BẮT BUỘC** cho Qwen3.5 |
| peft | ≥ 0.13.0 | Cho LoRA (training phase) |
| bitsandbytes | ≥ 0.44.0 | Cho 4-bit quantization |
| accelerate | ≥ 0.34.0 | Device mapping |
| flash-attn | ≥ 2.5.0 | Optional, RTX 6000 hỗ trợ |

### Conflict tiềm ẩn trên Kaggle
1. **transformers vs sentence-transformers**: Kaggle pre-install `sentence-transformers` yêu cầu `transformers<5.0`. Cần force upgrade:
   ```python
   subprocess.run([sys.executable, "-m", "pip", "install", 
                   "--force-reinstall", "--no-deps",
                   "--no-index", "--find-links", wheels_dir,
                   "transformers>=5.4.0"])
   ```

2. **torch version mismatch**: Kaggle đã có torch, KHÔNG nên upgrade torch từ wheels (sẽ break CUDA drivers). Chỉ upgrade transformers/peft/accelerate.

3. **protobuf version**: Kaggle có thể có protobuf cũ. Thêm vào wheels:
   ```bash
   pip download -d wheels protobuf>=4.0
   ```

### Offline install strategy trên Kaggle
```python
import subprocess, sys

WHEELS = "/kaggle/input/vlsp2025-offline/wheels"

# KHÔNG upgrade torch (sẽ break CUDA)
# CHỈ upgrade transformers ecosystem
for pkg in ["transformers", "accelerate", "peft", "bitsandbytes", 
            "safetensors", "tokenizers", "huggingface_hub"]:
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", WHEELS,
        "--force-reinstall", "--no-deps", pkg
    ], capture_output=True)
```

---

## Tóm tắt quy trình

```
[Máy có internet]                    [Kaggle - không internet]
     │                                      │
     ├─ Clone repo                          ├─ Add datasets
     ├─ Download models                     ├─ Install wheels offline
     ├─ Download wheels                     ├─ Copy code + data
     ├─ Download data                       ├─ Run baseline
     └─ Upload to Kaggle Datasets           └─ Download results
```
