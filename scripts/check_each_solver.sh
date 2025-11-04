#!/usr/bin/env bash
set -euo pipefail

# Quick sanity check on a single small case across all methods.
# Logs are stored under logs/ with timestamp in filename.

mkdir -p logs

N=${N:-128}
M=${M:-256}
SEED=${SEED:-42}
TOL=${TOL:-1e-8}
MAX_ITER=${MAX_ITER:-100}
MSIGN_STEPS=${MSIGN_STEPS:-5}

METHODS=("brent" "bisection" "secant" "fixed_point" "newton")

for METHOD in "${METHODS[@]}"; do
  LOG="logs/${METHOD}_n${N}_m${M}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[quickcheck] method=${METHOD} n=${N} m=${M} seed=${SEED} tol=${TOL} max_iter=${MAX_ITER} msign_steps=${MSIGN_STEPS}"
  python -u root_solver.py \
    --method "${METHOD}" \
    --n "${N}" --m "${M}" \
    --seed "${SEED}" \
    --tol "${TOL}" \
    --max_iter "${MAX_ITER}" \
    --msign_steps "${MSIGN_STEPS}" | tee "${LOG}"
done

echo "[done] logs written to logs/"
