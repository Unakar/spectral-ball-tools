#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/bench_grid.sh benchmarks
# Env overrides:
#   METHODS ("brent newton fixed"), SEEDS ("0 1 2"), MSIGN_STEPS (10), TOL (1e-8), DEVICE (auto), CASES (1)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="${1:-benchmarks}"
mkdir -p "$OUT_DIR"

METHODS="${METHODS:-brent newton fixed}"
SEEDS="${SEEDS:-0 1 2}"
export MSIGN_STEPS="${MSIGN_STEPS:-10}"
export TOL="${TOL:-1e-8}"
export DEVICE="${DEVICE:-auto}"
export CASES="${CASES:-1}"

for M in 32 64 128; do
  for N in 16 32 64; do
    OUT="$OUT_DIR/bench_${M}x${N}.csv"
    bash "$DIR/scripts/bench.sh" "$M" "$N" "$METHODS" "$SEEDS" "$OUT"
  done
done

# Merge CSVs if python3 is available
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PY' "$OUT_DIR" || true
import sys, csv, glob, os
outdir = sys.argv[1]
files = sorted(glob.glob(os.path.join(outdir, 'bench_*x*.csv')))
rows=[]
for f in files:
    with open(f, newline='') as fh:
        rd=csv.DictReader(fh)
        rows.extend(rd)
if rows:
    out=os.path.join(outdir,'bench_all.csv')
    with open(out,'w',newline='') as fw:
        wr=csv.DictWriter(fw, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print('Merged ->', out)
PY
fi

echo "Done. CSVs are under $OUT_DIR"

