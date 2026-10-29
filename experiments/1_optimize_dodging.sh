#!/bin/bash
# ---------------------------------------------------------
# PHASE 1: DIGITAL OPTIMIZATION
# ---------------------------------------------------------

EXP_NAME="..."

echo "==================================================="
echo "Starting Phase 1 Optimization for: $EXP_NAME"
echo "==================================================="

# Safely navigate to the src directory relative to this script's location
cd "$(dirname "$0")/../src"

DATA_DIR="../data/private/$EXP_NAME"
OUT_DIR="../outputs/$EXP_NAME"

# Create the specific output directory for this experiment
mkdir -p "$OUT_DIR"

# Run the optimization and route all artifacts (PDF, JSON, PT) to the new folder
python optimize.py \
    --attack dodging \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR"

echo "Optimization complete. PDF saved to $OUT_DIR. Go print the glasses!"