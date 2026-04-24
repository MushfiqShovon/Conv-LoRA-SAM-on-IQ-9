#!/bin/bash
# =============================================================================
# run_inference.sh  –  Run Conv-LoRA SAM evaluation on the IQ-9075 NPU
#
# Prerequisites:
#   1. SNPE SDK installed on this device (snpe-net-run in PATH, or set SNPE_ROOT)
#   2. DLC files downloaded into dlc/  (see README.md)
#   3. Dataset images downloaded into datasets/gan-generated/ (see README.md)
#
# Usage:
#   ./run_inference.sh                              # test2_class2, all images, NPU
#   ./run_inference.sh test1_class1                 # specify CSV name
#   ./run_inference.sh test2_class2 --max_images 5  # quick test
#   ./run_inference.sh test2_class2 --use_cpu       # CPU fallback
#   SNPE_ROOT=/path/to/sdk ./run_inference.sh       # explicit SDK path
#
#   Run all 18 test CSVs:
#   for csv in test{1..6}_class{1..3}; do ./run_inference.sh $csv; done
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# First positional argument = data_name (optional, default: test2_class2)
DATA_NAME="${1:-test2_class2}"
# Shift so remaining args are passed through to python
[[ $# -gt 0 ]] && shift

PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: python3 not found. Install Python 3."
    exit 1
fi

# Install required Python packages if missing
echo "Checking Python dependencies ..."
$PYTHON -c "import PIL" 2>/dev/null || $PYTHON -m pip install --user pillow -q
$PYTHON -c "import scipy" 2>/dev/null || $PYTHON -m pip install --user scipy -q
$PYTHON -c "import numpy" 2>/dev/null || $PYTHON -m pip install --user numpy -q

# Check DLC files exist
if [[ ! -f "dlc/convlora_sam_encoder_ptq.dlc" || ! -f "dlc/convlora_sam_decoder_ptq.dlc" ]]; then
    echo "ERROR: DLC files not found in dlc/"
    echo "Download them from Google Drive (see README.md):"
    echo "  dlc/convlora_sam_encoder_ptq.dlc  (~2.4 GB)"
    echo "  dlc/convlora_sam_decoder_ptq.dlc  (~15 MB)"
    exit 1
fi

# Check dataset CSV exists
if [[ ! -f "datasets/gan-generated/${DATA_NAME}.csv" ]]; then
    echo "ERROR: CSV not found: datasets/gan-generated/${DATA_NAME}.csv"
    echo "Available CSVs:"
    ls datasets/gan-generated/*.csv 2>/dev/null | xargs -n1 basename || echo "  (none)"
    exit 1
fi

# Build snpe_root arg if SNPE_ROOT is set
SNPE_ARG=""
if [[ -n "$SNPE_ROOT" ]]; then
    SNPE_ARG="--snpe_root $SNPE_ROOT"
fi

echo "============================================================"
echo "  Conv-LoRA SAM — NPU Evaluation"
echo "  Dataset:  ${DATA_NAME}"
echo "  Device:   $(uname -n)"
echo "  Python:   $PYTHON"
echo "  SNPE SDK: ${SNPE_ROOT:-'(from PATH)'}"
echo "============================================================"

$PYTHON evaluate_on_device.py \
    --data_name "$DATA_NAME" \
    $SNPE_ARG \
    "$@"

echo ""
echo "Results saved in: work/results/iou_${DATA_NAME}.txt"
