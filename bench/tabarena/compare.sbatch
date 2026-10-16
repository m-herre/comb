#!/bin/bash
# Leaderboards of every arm after the workers (CPU only): OUT/result.json, slices.csv, paired_<suite>.csv.
#   sbatch -d afterany:JOBS -o OUT/compare-%j.out bench/tabarena/compare.sbatch OUT [extra run.py args]
#SBATCH --job-name=comb-tabarena-compare
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail
OUT=$1; shift
PY=${TABARENA_PY:?set TABARENA_PY to the TabArena environment's python}
export TABARENA_CACHE=${TABARENA_CACHE:-$HOME/.cache/tabarena-latest} ELO_SCOPE=lite PYTHONPATH="$SITE" PYTHONHASHSEED=0
cd "$OUT"
"$PY" "$HARNESS/run.py" --out "$OUT" --compare "$@" 2>&1
