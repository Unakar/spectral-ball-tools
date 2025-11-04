#!/usr/bin/env bash
set -euo pipefail

# Quick sanity check on a single small case across all methods.
# Logs are stored under logs/ with timestamp in filename.

mkdir -p logs

N=${N:-4096}
M=${M:-128}
SEED=${SEED:-42}
TOL=${TOL:-1e-4}
MAX_ITER=${MAX_ITER:-100}
MSIGN_STEPS=${MSIGN_STEPS:-10}
THETA_SOURCE=${THETA_SOURCE:-power}
POWER_ITERS=${POWER_ITERS:-30}

METHODS=("brent" "bisection" "secant" "fixed_point" "newton")

for METHOD in "${METHODS[@]}"; do
  LOG="logs/${METHOD}_n${N}_m${M}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[quickcheck] method=${METHOD} n=${N} m=${M} seed=${SEED} tol=${TOL} max_iter=${MAX_ITER} msign_steps=${MSIGN_STEPS} theta=${THETA_SOURCE} power_iters=${POWER_ITERS}"
  python -u root_solver.py \
    --method "${METHOD}" \
    --n "${N}" --m "${M}" \
    --seed "${SEED}" \
    --tol "${TOL}" \
    --max_iter "${MAX_ITER}" \
    --msign_steps "${MSIGN_STEPS}" \
    --theta_source "${THETA_SOURCE}" \
    --power_iters "${POWER_ITERS}" | tee "${LOG}"
done

echo "[done] logs written to logs/"
