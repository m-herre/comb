# Pretraining Comb

Comb's weights were pretrained from scratch in three stages with the code in `comb/train/` (a source checkout; the
wheel holds inference only), then fine-tuned in a fourth stage ([Stage 4](#stage-4-fine-tune)), which is not part of
this trainer yet. The recipe is TabICLv2's three stages with the in-context stage run over cells throughout; the
commands below, at their defaults, are the released run (seed 42, Muon, fp32 with TF32).

## Setup

```bash
git clone https://github.com/m-herre/comb && cd comb
pip install -e ".[train,dev]"
export PYTHONHASHSEED=0
```

The `train` extra pins torch, numpy, scipy and scikit-learn: the synthetic prior reproduces bit for bit only with
these versions, which need Python 3.13 (inference runs on Python 3.10+).

## The three stages

Global batch 64 tables, Muon on all parameters, warm-up then cosine decay. Defaults are the stage values below.

| Stage | Steps | Learning rate | Rows per table | Features |
|---|---|---|---|---|
| 1 | 500,000 | 8e-4 | up to 1,024 | 1-100 |
| 2 | 40,000 | 1e-4 | 400-10,240 (log-uniform) | 1-100 |
| 3 | 10,000 | 2e-5 | 400-60,000 (log-uniform) | 1-100 |

```bash
torchrun --standalone --nproc_per_node=4 -m comb.train.train --stage 1 --out runs/comb/stage1
torchrun --standalone --nproc_per_node=4 -m comb.train.train --stage 2 --out runs/comb/stage2 \
    --init runs/comb/stage1/step-500000.ckpt
torchrun --standalone --nproc_per_node=4 -m comb.train.train --stage 3 --out runs/comb/stage3 \
    --init runs/comb/stage2/step-40000.ckpt
python scripts/export_weights.py runs/comb/stage3/step-10000.ckpt weights/
```

- **Cell budget.** Each step draws M log-uniformly from `--cell-m-range` (default `128,2048`), from the seed and the
  step, so every GPU and every resume draws the same M. Validation and the saved checkpoint use `--cell-m` (1024).
  Tables with at most M training rows run without compression.
- **Resume.** Rerunning a command resumes from the latest `step-*.ckpt` in `--out`; `--init` loads weights only.
- **Logging.** W&B (offline by default: `--wandb-mode online`), plus `metrics.jsonl` in `--out`. Cell metrics are
  under `cell/` (cells used, rows per cell, exact-path fraction).
- **Hardware.** The released run used 4 x H200: about 0.33 s per step in stage 1; the three stages took about four days of wall-clock time.
  Each GPU runs `--workers` (16) prior processes; give a 4-GPU job about 224 CPU cores.

## Stage 4 (fine-tune)

Initialised from the stage-3 checkpoint at step 10,000, 5,000 steps, learning rate 2e-4 with cosine decay, global
batch 64, 4 x RTX PRO 6000. Two changes to stage 3:

- cell budget drawn from 512 to 8192 per step, M = 4096 at validation and in the saved weights;
- half of the tables from a realism prior (class imbalance, more binary tasks, high-cardinality and ID-like
  categoricals, missing values with train-mean imputation), half from stage 3's prior.

The realism prior is not part of this trainer yet.

## Tests

```bash
pytest tests/test_train.py     # tiny CPU runs: logging, resume, DDP, stage chain
```
