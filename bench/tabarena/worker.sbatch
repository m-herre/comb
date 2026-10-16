#!/bin/bash
# One fit worker (one GPU) of bench/tabarena/run.py; submit as many as the cluster gives, they share the queue.
#   sbatch -o OUT/slurm-%j.out bench/tabarena/worker.sbatch OUT ID [extra run.py args]
# Env: HARNESS (this folder), SITE (comb + lightpfn installed with --target), COMB_CHECKPOINT, TABARENA_PY.
#SBATCH --job-name=comb-tabarena
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
set -euo pipefail
OUT=$1 ID=$2; shift 2
PY=${TABARENA_PY:?set TABARENA_PY to the TabArena environment's python}
export TABARENA_CACHE=${TABARENA_CACHE:-$HOME/.cache/tabarena-latest} ELO_SCOPE=lite
export PYTHONPATH="$SITE" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$OUT/tmp/$ID" TABARENA_MODEL_ARTIFACTS_BASE_PATH="$OUT/tmp/$ID"  # node-local /tmp fills up
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name --format=csv,noheader
cd "$OUT"  # never import from the submit directory
"$PY" "$HARNESS/run.py" --out "$OUT" --worker "$ID" "$@" 2>&1
rm -rf "$TMPDIR"
