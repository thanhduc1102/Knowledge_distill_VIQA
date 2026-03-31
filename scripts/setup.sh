#!/bin/bash
# Setup script for VLSP 2025 Knowledge Distillation Pipeline
# Usage: bash scripts/setup.sh [--full]

set -e

echo "=== VLSP 2025 KD Pipeline Setup ==="
echo ""

# Core dependencies
echo "Installing core dependencies..."
pip install -q openai pandas tqdm pyarrow pyyaml

# ML dependencies
echo "Installing ML dependencies..."
pip install -q peft accelerate datasets

# Quantization support
echo "Installing bitsandbytes..."
pip install -q bitsandbytes

# TRL for GRPO (optional but recommended)
if [[ "$1" == "--full" ]]; then
    echo "Installing TRL for GRPO training..."
    pip install -q trl
fi

# Verify installations
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers; print(f'Transformers: {transformers.__version__}')
import peft; print(f'PEFT: {peft.__version__}')
import accelerate; print(f'Accelerate: {accelerate.__version__}')
import datasets; print(f'Datasets: {datasets.__version__}')

try:
    import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')
except: print('BitsAndBytes: NOT INSTALLED')

try:
    import trl; print(f'TRL: {trl.__version__}')
except: print('TRL: NOT INSTALLED (run with --full to install)')
"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/receive data/finqa data/pipeline
mkdir -p checkpoints/sft checkpoints/grpo

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Place ViNumQA data in data/receive/ (train.json, valid.json, test.json)"
echo "  2. Place FinQA data in data/finqa/ (train.json, dev.json, test.json)"
echo "  3. Run: python -m pipeline.run --gpu-profile p100_16gb --phases all"
