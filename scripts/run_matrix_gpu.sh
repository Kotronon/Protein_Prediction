#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BATCH_SIZE="${BATCH_SIZE:-200}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-0}"

echo "Checking CUDA..."
python scripts/check_gpu.py

echo "Running UdonPred 7x7 matrix on CUDA with batch size ${BATCH_SIZE}..."
python scripts/run_udonpred_matrix.py \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"

echo "Done. Matrix written to results/udonpred_matrix/matrix.csv"

