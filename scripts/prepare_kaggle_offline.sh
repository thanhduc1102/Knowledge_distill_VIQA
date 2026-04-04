#!/bin/bash
# ============================================================================
# Kaggle Offline Preparation Script
# ============================================================================
# This script prepares a complete offline package for running on Kaggle
# with RTX 6000 Pro 96GB WITHOUT internet access.
#
# Run this ONCE on a machine with internet access to download:
# 1. All Python wheel dependencies
# 2. Hugging Face model weights
# 3. Dataset files
#
# Then upload the output to Kaggle as a Dataset.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Output directory for the offline package
OFFLINE_DIR="${1:-$PROJECT_ROOT/kaggle_offline_package}"

echo "============================================"
echo " VLSP 2025 - Kaggle Offline Package Builder"
echo "============================================"
echo " Output: $OFFLINE_DIR"
echo ""

mkdir -p "$OFFLINE_DIR"
mkdir -p "$OFFLINE_DIR/wheels"
mkdir -p "$OFFLINE_DIR/models"
mkdir -p "$OFFLINE_DIR/data"

# ── Step 1: Download Python wheels ───────────────────────────────────

echo "[1/4] Downloading Python wheels..."
echo "  Target: Python 3.10, Linux x86_64, CUDA 12.x"

# Core requirements for offline install
pip download \
  --dest "$OFFLINE_DIR/wheels" \
  --python-version 3.10 \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  torch==2.5.1+cu124 \
  -f https://download.pytorch.org/whl/torch_stable.html \
  2>/dev/null || echo "  NOTE: torch wheel download may need manual handling"

pip download \
  --dest "$OFFLINE_DIR/wheels" \
  transformers>=4.46.0 \
  accelerate>=0.34.0 \
  peft>=0.13.0 \
  bitsandbytes>=0.44.0 \
  datasets>=2.20.0 \
  pandas>=2.0.0 \
  tqdm>=4.60.0 \
  pyarrow>=14.0.0 \
  pyyaml>=6.0 \
  sentencepiece>=0.2.0 \
  protobuf>=4.0 \
  safetensors>=0.4.0 \
  huggingface_hub>=0.25.0 \
  tokenizers>=0.20.0 \
  flash-attn \
  2>&1 | tail -5

echo "  Wheels downloaded: $(ls "$OFFLINE_DIR/wheels" | wc -l) files"

# ── Step 2: Download HuggingFace Models ──────────────────────────────

echo ""
echo "[2/4] Downloading Hugging Face models..."
echo "  This will take significant time and disk space."

# Install huggingface_hub CLI if needed
pip install -q huggingface_hub[cli] 2>/dev/null

MODELS=(
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3.5-122B-A10B"
)

for model in "${MODELS[@]}"; do
  model_dir="$OFFLINE_DIR/models/$(echo $model | tr '/' '_')"
  if [ -d "$model_dir" ] && [ "$(ls -A "$model_dir" 2>/dev/null)" ]; then
    echo "  SKIP (already exists): $model"
  else
    echo "  Downloading: $model -> $model_dir"
    huggingface-cli download "$model" \
      --local-dir "$model_dir" \
      --local-dir-use-symlinks False \
      2>&1 | tail -3
  fi
done

echo "  Model downloads complete."

# ── Step 3: Copy Dataset ─────────────────────────────────────────────

echo ""
echo "[3/4] Copying dataset files..."
cp -r "$PROJECT_ROOT/data/receive" "$OFFLINE_DIR/data/"
echo "  Data files copied."

# ── Step 4: Copy Project Code ────────────────────────────────────────

echo ""
echo "[4/4] Copying project code..."
rsync -av --exclude='.git' --exclude='__pycache__' \
  --exclude='checkpoints' --exclude='outputs' \
  --exclude='data/vinumqa_raw' --exclude='kaggle_offline_package' \
  --exclude='*.pyc' --exclude='.gitignore' \
  "$PROJECT_ROOT/" "$OFFLINE_DIR/code/" 2>/dev/null || \
  cp -r "$PROJECT_ROOT/"{baseline,pipeline,src,configs,scripts,requirements.txt} "$OFFLINE_DIR/code/"

echo "  Code copied."

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo " Package Ready!"
echo "============================================"
echo ""
echo " Directory: $OFFLINE_DIR"
echo " Size: $(du -sh "$OFFLINE_DIR" | cut -f1)"
echo ""
echo " Contents:"
echo "   wheels/  - Python packages for offline pip install"
echo "   models/  - Pre-downloaded HuggingFace model weights"
echo "   data/    - ViNumQA dataset (train/valid/test)"
echo "   code/    - Project source code"
echo ""
echo " Next steps:"
echo "   1. Upload this directory to Kaggle as a Dataset"
echo "   2. In Kaggle notebook, add this dataset"
echo "   3. Run: kaggle/kaggle_baseline_notebook.py"
echo ""
echo " Upload to Kaggle:"
echo "   kaggle datasets create -p $OFFLINE_DIR"
echo "   OR manually upload via web UI"
echo ""
