#!/usr/bin/env bash
set -euo pipefail

# Parse solver logs (from root_solver.py) into a single CSV.
# Usage:
#   bash scripts/parse_logs_to_csv.sh logs/ results/benchmarks.csv

LOG_DIR=${1:-logs}
OUT_CSV=${2:-results/benchmarks.csv}

mkdir -p "$(dirname "${OUT_CSV}")"
TMP_CSV=$(mktemp)

# CSV header
HEADER="file,method,n,m,seed,tol,max_iter,msign_steps,lambda,abs_f,abs_constraint,iters,fevals,time_ms,ortho_err,bracket_lo,bracket_hi"
echo "${HEADER}" > "${TMP_CSV}"

parse_file () {
  local f="$1"
  local filename="$(basename "$f")"

  # Infer meta fields from filename when possible
  # Patterns handled: {method}_n{N}_m{M}_s{SEED}*.log and variants with _ms{MSIGN} etc.
  local method=$(echo "$filename" | sed -E 's/^([a-z_]+).*/\1/')
  local n=$(echo "$filename" | sed -nE 's/.*_n([0-9]+).*/\1/p')
  local m=$(echo "$filename" | sed -nE 's/.*_m([0-9]+).*/\1/p')
  local seed=$(echo "$filename" | sed -nE 's/.*_s([0-9]+).*/\1/p')
  local msign_steps=$(echo "$filename" | sed -nE 's/.*_ms([0-9]+).*/\1/p')

  # TOL / MAX_ITER are printed in the header? If not, leave blank.
  local tol=""
  local max_iter=""
  # Try to read from first lines if printed, else empty
  # (If you want strict values, pass them via env and echo here.)

  # Extract lambda (line containing 'λ* =')
  local lambda=$(grep -E 'λ\* = ' "$f" | tail -n1 | sed -E 's/.*λ\* = ([^ ]+).*/\1/')

  # Extract |f(λ*)|
  local abs_f=$(grep -E '\|f\(λ\*\)\|' "$f" | tail -n1 | sed -E 's/.*\|f\(λ\*\)\| *: *([0-9.eE+-]+).*/\1/')
  if [[ -z "$abs_f" ]]; then
    # fixed_point path prints later; still same pattern
    abs_f=$(grep -E '\|f\(λ\*\)\|' "$f" | tail -n1 | sed -E 's/.*: *([0-9.eE+-]+).*/\1/')
  fi

  # Extract |tr(ΘᵀΦ)| for fixed_point (optional)
  local abs_constraint=$(grep -E '\|tr\(Θ' "$f" | tail -n1 | sed -E 's/.*\|tr.*: *([0-9.eE+-]+).*/\1/')

  # iters / fevals
  # Prefer new unified pattern 'iters       : N iters'; fallback to legacy 'iters/evals'
  local iters=$(grep -E 'iters\s*:' "$f" | tail -n1 | sed -E 's/.*iters *: *([0-9]+).*/\1/')
  local fevals=$(grep -E 'iters/evals' "$f" | tail -n1 | sed -E 's/.*, ([0-9]+) f-evals.*/\1/')

  # time (ms)
  local time_ms=$(grep -E 'time *:' "$f" | tail -n1 | sed -E 's/.*time *: *([0-9.]+) ms.*/\1/')
  if [[ -z "$time_ms" ]]; then
    # fixed_point prints "time : <sum> ms (sum) <avg> ms/iter (avg)" – capture sum
    time_ms=$(grep -E 'time *:' "$f" | tail -n1 | sed -E 's/.*time *: *([0-9.]+) ms.*/\1/')
  fi

  # orthogonality error
  local ortho_err=$(grep -E 'orthogonality error @λ\*' "$f" | tail -n1 | sed -E 's/.*: *([0-9.eE+-]+).*/\1/')

  # bracket (if any)
  local bracket=$(grep -E 'bracket *:' "$f" | tail -n1 | sed -E 's/.*\[([0-9.eE+-]+), *([0-9.eE+-]+)\].*/\1,\2/')
  local bracket_lo=$(echo "$bracket" | cut -d',' -f1)
  local bracket_hi=$(echo "$bracket" | cut -d',' -f2)

  # Defaults if missing
  [[ -z "$seed" ]] && seed="-"
  [[ -z "$msign_steps" ]] && msign_steps="-"
  [[ -z "$abs_constraint" ]] && abs_constraint="-"
  [[ -z "$iters" ]] && iters="-"
  [[ -z "$fevals" ]] && fevals="-"
  [[ -z "$time_ms" ]] && time_ms="-"
  [[ -z "$ortho_err" ]] && ortho_err="-"
  [[ -z "$lambda" ]] && lambda="-"
  [[ -z "$abs_f" ]] && abs_f="-"
  [[ -z "$bracket_lo" ]] && bracket_lo="-"
  [[ -z "$bracket_hi" ]] && bracket_hi="-"

  echo "${filename},${method},${n},${m},${seed},${tol},${max_iter},${msign_steps},${lambda},${abs_f},${abs_constraint},${iters},${fevals},${time_ms},${ortho_err},${bracket_lo},${bracket_hi}" >> "${TMP_CSV}"
}

shopt -s nullglob
for f in "${LOG_DIR}"/*.log; do
  parse_file "$f"
done

mv "${TMP_CSV}" "${OUT_CSV}"
echo "[parse] parsed $(wc -l < "${OUT_CSV}") rows (incl. header) -> ${OUT_CSV}"

# Optional averaging across repeats (group by method,n,m,tol,max_iter,msign_steps)
if [[ "${AVERAGE:-0}" == "1" ]]; then
  TMP_RAW="${OUT_CSV}"
  TMP_AVG=$(mktemp)
  echo "${HEADER}" > "${TMP_AVG}"
  awk -F',' 'NR==1{next}
  function isnum(x){return (x ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/)}
  {
    key=$2","$3","$4","$6","$7","$8
    k[key]=key
    c[key]+=1
    if(isnum($9))  s1[key]+=$9
    if(isnum($10)) s2[key]+=$10
    if(isnum($11)) s3[key]+=$11
    if(isnum($12)) s4[key]+=$12
    if(isnum($14)) s5[key]+=$14
    if(isnum($15)) s6[key]+=$15
  }
  END{
    for (key in k){
      split(key, a, ",");
      method=a[1]; n=a[2]; m=a[3]; tol=a[4]; maxit=a[5]; ms=a[6];
      cnt=c[key]
      lambda=(cnt?s1[key]/cnt:"-")
      absf=(cnt?s2[key]/cnt:"-")
      absc=(cnt?s3[key]/cnt:"-")
      iters=(cnt?s4[key]/cnt:"-")
      timems=(cnt?s5[key]/cnt:"-")
      ortho=(cnt?s6[key]/cnt:"-")
      printf("group-%s-%sx%s, %s, %s, %s, avg, %s, %s, %s, %s, %s, -, %s, %s, -, -\n", method, n, m, method, n, m, ms, lambda, absf, absc, iters, timems, ortho) >> "'${TMP_AVG}'"
    }
  }' "${TMP_RAW}"
  mv "${TMP_AVG}" "${OUT_CSV}"
  echo "[avg] grouped averages -> ${OUT_CSV}"
fi
