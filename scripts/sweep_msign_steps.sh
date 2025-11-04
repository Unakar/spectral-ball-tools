#!/usr/bin/env bash
set -euo pipefail

# Evaluate sensitivity to msign iteration steps for a fixed size and method.

mkdir -p logs results

N=${N:-128}
M=${M:-256}
SEED=${SEED:-42}
TOL=${TOL:-1e-8}
MAX_ITER=${MAX_ITER:-100}
METHOD=${METHOD:-fixed_point}
MSIGN_STEPS_LIST=(${MSIGN_STEPS_LIST:-3 5 7 9})
THETA_SOURCE=${THETA_SOURCE:-power}
POWER_ITERS=${POWER_ITERS:-3}

for STEPS in "${MSIGN_STEPS_LIST[@]}"; do
  LOG="logs/${METHOD}_n${N}_m${M}_ms${STEPS}_$(date +%Y%m%d_%H%M%S).log"
  echo "[sweep] method=${METHOD} n=${N} m=${M} msign_steps=${STEPS} theta=${THETA_SOURCE} power_iters=${POWER_ITERS}"
  python -u root_solver.py \
    --method "${METHOD}" \
    --n "${N}" --m "${M}" \
    --seed "${SEED}" \
    --tol "${TOL}" \
    --max_iter "${MAX_ITER}" \
    --msign_steps "${STEPS}" \
    --theta_source "${THETA_SOURCE}" \
    --power_iters "${POWER_ITERS}" | tee "${LOG}"
done

bash scripts/parse_logs_to_csv.sh logs results/sweep_msign_steps.csv
echo "[done] CSV saved to results/sweep_msign_steps.csv"
