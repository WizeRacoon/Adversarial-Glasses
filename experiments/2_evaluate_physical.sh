#!/bin/bash
# ---------------------------------------------------------
# PHASE 2: PHYSICAL LIVE EVALUATION
# ---------------------------------------------------------

# Make sure this experiment matches the glasses you are currently wearing!
EXP_NAME="..."

echo "==================================================="
echo "Starting Phase 2 Evaluation for: $EXP_NAME"
echo "==================================================="

# Safely navigate to the src directory relative to this script's location
cd "$(dirname "$0")/../src"

# The baseline should ALWAYS be the exact same photo for all 4 tests 
# to ensure the ArcFace/ResNet similarities are measured on the same scale.
BASELINE="../data/private/baseline_webcam.jpg"
OUT_DIR="../outputs/$EXP_NAME"

# Ensure the output directory exists before running
mkdir -p "$OUT_DIR"

# Run the live profiler. CSV and PNG plots will save alongside the PDF.
python evaluate.py \
    --mode profile \
    --baseline "$BASELINE" \
    --out_dir "$OUT_DIR"