#!/bin/bash
# experiments/run_batch_dodging.sh

# Resolve the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="${PROJECT_ROOT}/outputs/exp_batch_dodging_${TIMESTAMP}"

echo "=========================================="
echo " Starting Batch Dodging Experiment: ${TIMESTAMP} "
echo " Output Directory: ${EXPERIMENT_DIR} "
echo "=========================================="

mkdir -p "${EXPERIMENT_DIR}"

# 1. Optimize the glasses 
# (Ensure DATA_DIR in src/config.py is set to "../data/private/my_dataset")
echo "-> Phase 1: Generating adversarial textures from batch dataset..."
cd "${PROJECT_ROOT}/src" || exit
python optimize.py \
    --attack dodging \
    --data_dir ../data/private/my_dataset \
    --out_dir "${EXPERIMENT_DIR}"

# 2. Automatically launch the profiling sequence
# Uses the first image from your new dataset as the clean baseline reference
echo "-> Phase 2: Launching physical quantitative evaluation..."
echo "(Put on your printed glasses when prompted)"
python evaluate.py \
    --mode profile \
    --baseline ../data/private/my_dataset/face_000.jpg \
    --frames 100 \
    --out_dir "${EXPERIMENT_DIR}"

echo "Experiment complete. Results saved in ${EXPERIMENT_DIR}."