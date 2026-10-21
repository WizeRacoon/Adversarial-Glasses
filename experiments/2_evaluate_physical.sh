#!/bin/bash
# ---------------------------------------------------------
# PHASE 2: PHYSICAL LIVE EVALUATION
# ---------------------------------------------------------

EXP_NAME="..."

echo "==================================================="
echo "Starting Phase 2 Evaluation for: $EXP_NAME"
echo "==================================================="

cd ../src

# The baseline should ALWAYS be the exact same photo for all 4 tests 
# to ensure the ArcFace/ResNet similarities are measured on the same scale.
BASELINE="../data/private/baseline_webcam.jpg"
OUT_DIR="../outputs/$EXP_NAME"

# Run the live profiler. CSV and PNG plots will save alongside the PDF.
python evaluate.py \
    --mode profile \
    --baseline $BASELINE \
    --out_dir $OUT_DIR