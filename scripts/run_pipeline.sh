#!/bin/bash
# Run the full KD pipeline
# Usage: bash scripts/run_pipeline.sh [gpu_profile] [phases...]
# Examples:
#   bash scripts/run_pipeline.sh                        # Full pipeline on P100
#   bash scripts/run_pipeline.sh a100_80gb              # Full pipeline on A100
#   bash scripts/run_pipeline.sh p100_16gb data_prep sft  # Only data_prep + sft

set -e

cd "$(dirname "$0")/.."

GPU_PROFILE="${1:-p100_16gb}"
shift 2>/dev/null || true
PHASES="${@:-all}"

echo "=== VLSP 2025 KD Pipeline ==="
echo "GPU Profile: $GPU_PROFILE"
echo "Phases: $PHASES"
echo "Config: configs/${GPU_PROFILE}.yaml"
echo ""

# Check if config exists
CONFIG_PATH="configs/${GPU_PROFILE}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    echo "Using built-in profile: $GPU_PROFILE"
    CONFIG_PATH=""
fi

if [ -n "$CONFIG_PATH" ]; then
    python -m pipeline.run \
        --gpu-profile "$GPU_PROFILE" \
        --config "$CONFIG_PATH" \
        --phases $PHASES
else
    python -m pipeline.run \
        --gpu-profile "$GPU_PROFILE" \
        --phases $PHASES
fi
