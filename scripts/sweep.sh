#!/usr/bin/env bash
set -euo pipefail

# Sweep multiple matrix sizes and methods; write per-run logs and one merged CSV via parse script.
# You can override arrays via environment variables before calling this script.

mkdir -p logs results plots

SIZES=(${SIZES:-"128x128" "4096x128"})
METHODS=(${METHODS:-"brent" "bisection" "secant" "fixed_point"})
SEED=${SEED:-42}
REPEATS=${REPEATS:-3}
TOL=${TOL:-1e-4}
MAX_ITER=${MAX_ITER:-50}
MSIGN_STEPS=${MSIGN_STEPS:-10}
THETA_SOURCE=${THETA_SOURCE:-svd}
POWER_ITERS=${POWER_ITERS:-30}

for SIZE in "${SIZES[@]}"; do
  N="${SIZE%x*}"
  M="${SIZE#*x}"
  for METHOD in "${METHODS[@]}"; do
    for R in $(seq 1 ${REPEATS}); do
      CUR_SEED=$((SEED + R - 1))
      LOG="logs/${METHOD}_n${N}_m${M}_s${CUR_SEED}_th${THETA_SOURCE}_pi${POWER_ITERS}_ms${MSIGN_STEPS}_r${R}_$(date +%Y%m%d_%H%M%S).log"
      { \
        echo "[run] method=${METHOD} n=${N} m=${M} seed=${CUR_SEED} rep=${R}/${REPEATS} tol=${TOL} max_iter=${MAX_ITER} msign_steps=${MSIGN_STEPS} theta=${THETA_SOURCE} power_iters=${POWER_ITERS}"; \
        python3 -u root_solver.py \
          --method "${METHOD}" \
          --n "${N}" --m "${M}" \
          --seed "${CUR_SEED}" \
          --tol "${TOL}" \
          --max_iter "${MAX_ITER}" \
          --msign_steps "${MSIGN_STEPS}" \
          --theta_source "${THETA_SOURCE}" \
          --power_iters "${POWER_ITERS}"; \
      } | tee "${LOG}"
    done
  done
done

# Collect logs -> CSVs and plots
RAW_CSV=${RAW_CSV:-results/raw.csv}
AVG_CSV=${AVG_CSV:-results/avg.csv}
PLOT_DIR=${PLOT_DIR:-plots}
python3 -u scripts/collect_and_plot.py \
  --logs logs \
  --out-raw "${RAW_CSV}" \
  --out-avg "${AVG_CSV}" \
  --plot-dir "${PLOT_DIR}" \
  --out-avg-dir results || echo "[warn] collect_and_plot failed; please ensure matplotlib is installed."

echo "[done] Raw CSV: ${RAW_CSV}  | AVG CSV: ${AVG_CSV}  | Plots in ${PLOT_DIR}/"
