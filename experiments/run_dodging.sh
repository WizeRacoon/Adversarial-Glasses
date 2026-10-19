#!/bin/bash
# experiments/run_dodging.sh

# Resolve the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="${PROJECT_ROOT}/outputs/exp_dodging_${TIMESTAMP}"

echo "=========================================="
echo " Starting Dodging Experiment: ${TIMESTAMP} "
echo " Output Directory: ${EXPERIMENT_DIR} "
echo "=========================================="

mkdir -p "${EXPERIMENT_DIR}"

# 1. Optimize the glasses
echo "-> Phase 1: Generating adversarial textures..."
cd "${PROJECT_ROOT}/src" || exit
python optimize.py \
    --attack dodging \
    --out_dir "${EXPERIMENT_DIR}"

# 2. Automatically launch the profiling sequence
echo "-> Phase 2: Launching physical quantitative evaluation..."
echo "(Put on your printed glasses when prompted)"
python evaluate.py \
    --mode profile \
    --baseline ../data/sample/target.png \
    --frames 100 \
    --out_dir "${EXPERIMENT_DIR}"

echo "Experiment complete. Results saved in ${EXPERIMENT_DIR}."