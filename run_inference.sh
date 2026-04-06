#!/bin/bash
# =============================================================================
# run_inference.sh  –  Run Conv-LoRA SAM evaluation on the IQ-9075 NPU
#
# Prerequisites:
#   1. SNPE SDK installed on this device (snpe-net-run in PATH, or set SNPE_ROOT)
#   2. DLC files present in dlc/  (copy them manually — they are not in git)
#   3. Dataset CSV and images/labels placed under datasets/
#
# Usage:
#   ./run_inference.sh                          # all images, NPU
#   ./run_inference.sh --max_images 5           # quick 5-image test
#   ./run_inference.sh --use_cpu                # CPU fallback (no NPU)
#   SNPE_ROOT=/path/to/sdk ./run_inference.sh   # explicit SDK path
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: python3 not found. Install Python 3."
    exit 1
fi

# Check DLC files exist
if [[ ! -f "dlc/convlora_sam_encoder_ptq.dlc" || ! -f "dlc/convlora_sam_decoder_ptq.dlc" ]]; then
    echo "ERROR: DLC files not found in dlc/"
    echo "Copy them to this device:"
    echo "  dlc/convlora_sam_encoder_ptq.dlc  (~2.4 GB)"
    echo "  dlc/convlora_sam_decoder_ptq.dlc  (~15 MB)"
    exit 1
fi

# Build snpe_root arg if SNPE_ROOT is set
SNPE_ARG=""
if [[ -n "$SNPE_ROOT" ]]; then
    SNPE_ARG="--snpe_root $SNPE_ROOT"
fi

echo "============================================================"
echo "  Conv-LoRA SAM — NPU Evaluation"
echo "  Device: $(uname -n)"
echo "  Python: $PYTHON"
echo "  SNPE_ROOT: ${SNPE_ROOT:-'(from PATH)'}"
echo "============================================================"

$PYTHON evaluate_on_device.py \
    --data_csv datasets/test.csv \
    $SNPE_ARG \
    "$@"

echo ""
echo "Results saved in: work/results/iou_results.txt"
