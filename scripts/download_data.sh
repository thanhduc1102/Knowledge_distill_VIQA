#!/bin/bash
# ============================================================================
# Download ViNumQA and FinQA datasets
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Downloading Datasets ==="

# ── ViNumQA (from Google Drive) ──────────────────────────────────────
echo ""
echo "[1/2] ViNumQA dataset..."

pip install -q gdown 2>/dev/null

VINUMQA_DIR="data/vinumqa_raw"
if [ ! -d "$VINUMQA_DIR" ] || [ -z "$(ls -A "$VINUMQA_DIR" 2>/dev/null)" ]; then
    gdown --folder "https://drive.google.com/drive/folders/1ZFtIF88ZJzubbkB0pIJA_rxz3NrzC8gG" \
        -O "$VINUMQA_DIR/"
    echo "  Downloaded to $VINUMQA_DIR"
else
    echo "  Already exists: $VINUMQA_DIR"
fi

# Copy to standard location
mkdir -p data/receive
VINUMQA_SRC=$(find "$VINUMQA_DIR" -name "train.json" -exec dirname {} \; | head -1)
if [ -n "$VINUMQA_SRC" ]; then
    cp "$VINUMQA_SRC/train.json" data/receive/train.json
    cp "$VINUMQA_SRC/valid.json" data/receive/valid.json
    cp "$VINUMQA_SRC/test.json" data/receive/test.json
    [ -f "$VINUMQA_SRC/private_test.json" ] && cp "$VINUMQA_SRC/private_test.json" data/receive/private_test.json
    echo "  Copied to data/receive/"
fi

echo "  Data summary:"
for f in data/receive/*.json; do
    count=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
    echo "    $(basename $f): $count samples"
done

# ── FinQA (from GitHub) ──────────────────────────────────────────────
echo ""
echo "[2/2] FinQA dataset..."

FINQA_DIR="data/finqa"
if [ ! -d "$FINQA_DIR" ] || [ -z "$(ls -A "$FINQA_DIR" 2>/dev/null)" ]; then
    mkdir -p "$FINQA_DIR"
    FINQA_REPO="https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
    for split in train dev test; do
        echo "  Downloading $split.json..."
        curl -sL "$FINQA_REPO/${split}.json" -o "$FINQA_DIR/${split}.json"
    done
    echo "  Downloaded to $FINQA_DIR"
else
    echo "  Already exists: $FINQA_DIR"
fi

echo "  FinQA summary:"
for f in "$FINQA_DIR"/*.json; do
    count=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
    echo "    $(basename $f): $count samples"
done

echo ""
echo "=== Download Complete ==="
