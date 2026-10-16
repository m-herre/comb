#!/bin/bash
# GPU checks + timing benchmark on one RTX PRO 6000 (the TabArena timing hardware).
#   sbatch -o bench/results/gpu-%j.out bench/gpu.sbatch
#SBATCH --job-name=comb-gpu
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export OMP_NUM_THREADS=16  # weights: COMB_CHECKPOINT if set, else the released weights
PY=${PY:-python}  # with comb installed; lightpfn and tabicl add those models to the timing
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
git rev-parse HEAD 2>/dev/null || true
$PY bench/gpu_check.py
$PY bench/timing.py --out bench/results/timing.csv
