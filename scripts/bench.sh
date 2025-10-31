#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/bench.sh 64 32 "brent secant fixed newton" "0 1 2" out_64x32.csv
# Env overrides:
#   PYTHON, MSIGN_STEPS (10), TOL (1e-8), DEVICE (auto), CASES (1)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY="${PYTHON:-python3}"
M="${1:-64}"
N="${2:-32}"
METHODS="${3:-brent newton fixed}"
SEEDS="${4:-0 1 2}"
OUT="${5:-bench_${M}x${N}.csv}"
MSIGN_STEPS="${MSIGN_STEPS:-10}"
TOL="${TOL:-1e-8}"
DEVICE="${DEVICE:-auto}"
CASES="${CASES:-1}"

echo "m,n,method,seed,iters,f_evals,time_ms,f_abs,converged,f0,rel_phi_err0,rel_phi_err_lam" > "$OUT"

for seed in $SEEDS; do
  for method in $METHODS; do
    LOG="$(mktemp)"
    $PY "$DIR/solve_lambda.py" \
      --method "$method" --m "$M" --n "$N" \
      --seed "$seed" --msign-steps "$MSIGN_STEPS" --tol "$TOL" \
      --device "$DEVICE" --cases "$CASES" > "$LOG"

    f0=$(grep -m1 "Case seed=" "$LOG" | sed -E 's/.*f\(0\)=([eE0-9+.-]+).*/\1/')
    rel0=$(grep -m1 "Case seed=" "$LOG" | sed -E 's/.*rel_phi_err@0=([eE0-9+.-]+).*/\1/')
    line=$(grep -m1 '^\[' "$LOG" || true)
    f_abs=$(echo "$line" | sed -E 's/.*\| \|f\|=([eE0-9+.-]+).*/\1/')
    iters=$(echo "$line" | sed -E 's/.*\| iters=([0-9]+).*/\1/')
    f_evals=$(echo "$line" | sed -E 's/.*\| f_evals=([0-9]+).*/\1/')
    time_ms=$(echo "$line" | sed -E 's/.*\| time=([0-9.]+) ms.*/\1/')
    converged=$(echo "$line" | sed -E 's/.*\| converged=([A-Za-z]+).*/\1/')
    rellam=$(grep -m1 '^Check:' "$LOG" | sed -E 's/.*rel_phi_err@lam=([eE0-9+.-]+).*/\1/')

    echo "$M,$N,$method,$seed,$iters,$f_evals,$time_ms,$f_abs,$converged,$f0,$rel0,$rellam" >> "$OUT"
    rm -f "$LOG"
  done
done

echo "Wrote $OUT"

