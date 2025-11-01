#!/usr/bin/env bash
set -euo pipefail

# Sweep multiple matrix sizes and methods; write per-run logs and one merged CSV via parse script.
# You can override arrays via environment variables before calling this script.

mkdir -p logs results

SIZES=(${SIZES:-"64x32" "128x64" "256x128" "512x256"})
METHODS=(${METHODS:-"brent" "secant" "fixed_point" "newton"})
SEED=${SEED:-42}
TOL=${TOL:-1e-8}
MAX_ITER=${MAX_ITER:-100}
MSIGN_STEPS=${MSIGN_STEPS:-5}

for SIZE in "${SIZES[@]}"; do
  N="${SIZE%x*}"
  M="${SIZE#*x}"
  for METHOD in "${METHODS[@]}"; do
    LOG="logs/${METHOD}_n${N}_m${M}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
    echo "[run] method=${METHOD} n=${N} m=${M} seed=${SEED} tol=${TOL} max_iter=${MAX_ITER} msign_steps=${MSIGN_STEPS}"
    python -u root_solver.py \
      --method "${METHOD}" \
      --n "${N}" --m "${M}" \
      --seed "${SEED}" \
      --tol "${TOL}" \
      --max_iter "${MAX_ITER}" \
      --msign_steps "${MSIGN_STEPS}" | tee "${LOG}"
  done
done

bash scripts/parse_logs_to_csv.sh logs results/benchmarks.csv
echo "[done] CSV saved to results/benchmarks.csv"
