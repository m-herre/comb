# Results

All numbers are for the released weights at the defaults (M = 4096, 4 members, fp16 on CUDA) unless they say
otherwise.

## TALENT (held out)

106 TALENT classification tasks that played no part in Comb's development, with protocol and analysis fixed before
the single run ([pre-registration](../bench/talent/PREREGISTRATION.md), [manifest](../bench/talent/manifest.csv),
`bench/talent/run_talent.py`, `bench/talent/analyze.py`, raw results in [bench/results/talent/](../bench/results/talent/)).
Every model through its own scikit-learn wrapper at its defaults, fp32, on RTX PRO 6000 GPUs: Comb with 8 members
(the default at the time), TabICLv2 (`tabicl` 2.2.0), XGBoost 3.4.1, CatBoost 1.2.10. Metric: mean over tasks of the
paired AUC difference (one-vs-rest macro AUC for multiclass), 95% task-bootstrap CI.

| paired difference, 106 tasks | mean | 95% CI | wins / losses / ties |
|---|---|---|---|
| **AUC, Comb − TabICLv2** | **−0.37 pp** | [−0.58, −0.19] | 26 / 76 / 4 |
| accuracy, Comb − TabICLv2 | −0.44 pp | [−0.72, −0.21] | 28 / 61 / 17 |
| AUC, Comb − CatBoost (default) | +2.76 pp | [+1.87, +3.87] | 90 / 13 / 3 |
| AUC, Comb − XGBoost (default) | +3.59 pp | [+2.65, +4.68] | 96 / 6 / 4 |
| AUC vs TabICLv2, < 5,000 training rows (59 tasks) | −0.21 pp | [−0.38, −0.06] | |
| AUC vs TabICLv2, 5,000-50,000 rows (41) | −0.41 pp | [−0.84, −0.12] | |
| AUC vs TabICLv2, > 50,000 rows (6) | −1.70 pp | [−3.20, −0.47] | |

Mean AUC: TabICLv2 0.9037, Comb 0.9001, CatBoost 0.8724, XGBoost 0.8642. The pre-registered "TabICLv2-level" rule (CI
lower bound above −0.25 pp) failed. The gap to TabICLv2 grows with table size. At today's defaults (4 members, fp16,
run after the pre-registered read) Comb scores 0.08 pp lower and −0.45 pp [−0.67, −0.26] against TabICLv2.

## TabArena and BeyondArena, official protocol

TabArena's pipeline under its official validation protocol (TabArena-Lite: 8-fold bagging, `TabArena-v0.1 [8x1]`;
BeyondArena core2k: its own protocol, `8x1 + tiny 5x5 <= 500 rows + task-specific`), classification, run by us on RTX
PRO 6000 GPUs (`bench/tabarena/run.py` with `COMB_OFFICIAL=1`). Elo against the published leaderboards' verified entries
(hosted APIs under review and entries with `verified=False` left out, `bench/tabarena/listing.py`; TabArena-Lite 100
methods, BeyondArena 44, ours included); train and inference time are TabArena's medians per 1,000 rows. Not an
official TabArena entry. Raw results: [bench/results/tabarena/](../bench/results/tabarena/).

| Model | TabArena-Lite Elo | train / infer (s) | BeyondArena Elo | train / infer (s) |
|---|---|---|---|---|
| TabICLv2 (published entries) | 1558 | 0.93 / 0.141 | 1189 | 11.96 / 2.256 |
| Comb, `cell_m=16384` (8 k-means steps) | 1474 (+84 / −69) | 0.32 / 0.019 | 1186 (+40 / −43) | 0.64 / 0.042 |
| **Comb, defaults** (M = 4096, 4 members, 4 k-means steps) | **1443** (+89 / −84) | **0.20 / 0.014** | **1167** (+36 / −45) | 0.56 / **0.033** |
| Comb, `n_estimators=2` (8 k-means steps) | 1433 (+103 / −96) | 0.14 / 0.010 | 1124 (+38 / −46) | 0.47 / 0.027 |
| LightPFN (same pipeline and GPUs ¹) | 1420 (+70 / −68) | 0.23 / 0.029 | 1096 (+33 / −38) | 0.44 / 0.072 |

¹ Reproduces LightPFN's published TabArena-Lite entry (1419).

- BeyondArena by table size, defaults (Elo): tiny 1391, small 1261, medium 1006, large 633 (`cell_m=16384`: 696).
  By task type Comb is weakest on temporal splits (745, 10 tasks). [Per-slice Elo of every method](../bench/results/tabarena/official_slices.csv).
- Rank at the defaults among the verified published entries: 25th of 96 on TabArena-Lite and 14th of 40 on
  BeyondArena.
- Comb's train time is its `fit` (every row encoded and the cells built) once per bagged fold.

### How the defaults were chosen

The defaults were chosen on the same two suites as the results above; TALENT played no part in the choice.

A single-fit sweep (one fit per split, no bagging; same pipeline, both suites),
results in `bench/results/tabarena/singlefit_sweep_*_slices.csv`:

- **Members:** 1 / 2 / 4 / 8 at M = 4096: 1377 / 1432 / 1440 / 1458 (TabArena-Lite), 1091 / 1110 / 1163 / 1182
  (BeyondArena); fit + predict time grows about linearly.
- **Cell budget:** 1024 / 2048 / 4096 / 8192 / 16384 at 4 members: below 4096 costs 30-60 Elo for almost no time;
  above, Elo rises on medium and large tables (8192: +23 / +16, 16384: +28 / +22) for 10-35% more time. A size rule
  `clip(N, lo, hi)` is the same as a fixed `hi` (tables with N <= M are exact anyway); `clip(32 sqrt(N), 4096, 16384)`
  lands between 4096 and 16384.
- **Precision:** fp16 on small tables gives the same Elo as fp32 (1458 vs 1457) at about a third of the time.
- **Softmax temperature:** 0.9 (default) and 1.0 tie; 0.8 and 0.7 lose 25-55 Elo.
- **k-means steps at inference:** 2 / 4 / 8 tie within noise; 4 is the default.

## Speed

`bench/timing.py` on one RTX PRO 6000 (Blackwell): synthetic tables (`make_classification`, 20 features, 2 classes),
1,000 test rows, median of 3 runs after a warm-up, defaults for every model (Comb: M = 4096, 4 members, fp16; LightPFN 1.0.0; TabICLv2: `tabicl` 2.2.0). Raw:
[bench/results/timing.csv](../bench/results/timing.csv).

| Training rows | Model | fit (s) | predict (s) | AUC |
|---|---|---|---|---|
| 1,000 | Comb, 8 members | 0.017 | 0.010 | 0.992 |
|  | Comb, 4 members | 0.013 | 0.007 | 0.992 |
|  | Comb, 2 members | 0.012 | 0.007 | 0.991 |
|  | LightPFN, 4 members | 0.070 | 0.011 | 0.991 |
|  | TabICLv2, 8 members | 0.226 | 0.110 | 0.992 |
| 5,000 | Comb, 8 members | 0.105 | 0.017 | 0.994 |
|  | Comb, 4 members | 0.056 | 0.011 | 0.994 |
|  | Comb, 2 members | 0.036 | 0.008 | 0.993 |
|  | LightPFN, 4 members | 0.129 | 0.016 | 0.992 |
|  | TabICLv2, 8 members | 0.246 | 0.189 | 0.993 |
| 20,000 | Comb, 8 members | 0.279 | 0.017 | 0.997 |
|  | Comb, 4 members | 0.135 | 0.010 | 0.997 |
|  | Comb, 2 members | 0.082 | 0.008 | 0.996 |
|  | LightPFN, 4 members | 0.646 | 0.036 | 0.997 |
|  | TabICLv2, 8 members | 0.353 | 0.805 | 0.997 |
| 100,000 | Comb, 8 members | 0.788 | 0.019 | 0.996 |
|  | Comb, 4 members | 0.433 | 0.015 | 0.995 |
|  | Comb, 2 members | 0.258 | 0.013 | 0.995 |
|  | LightPFN, 4 members | 0.673 | 0.036 | 0.995 |
|  | TabICLv2, 8 members | 0.949 | 7.968 | 0.995 |

Reading:

- Comb's prediction time is flat in the number of training rows above M.
- Comb does its in-context work over the training rows once, in `fit` (embedding every row, k-means, the cells), so
  predicting does not touch the training rows. LightPFN subsamples above 20,000 training rows.
- Members cost about linearly in `fit` on large tables (100,000 rows: 2 / 4 / 8 members 0.26 / 0.43 / 0.79 s) and
  little in predict.
- Edits rebuild the cells in time independent of the number of rows: about 18 ms at M = 4096.
- These are single synthetic tables on one GPU; they show scaling, not a benchmark-wide speed ranking (the
  official TabArena times above are that).
- Memory at the defaults (`bench/memory.py`, peak above the loaded network, [memory.csv](../bench/results/memory.csv)):
  fit 4.9 GB at 300,000 rows x 50 features and 8.1 GB at 1,000,000 x 20; predicting 10,000 rows 1.8 / 0.8 GB. On an
  8 GB GPU (simulated): fit 2.7 / 4.4 GB, predict 0.9 / 0.4 GB. `cell_m=16384` with 8 members ran out of a 100 GB
  host-RAM budget on one very wide table (gene expression, thousands of features) under TabArena's bagging; at the defaults every table ran.

## Exactness of edits

With the released weights (20,000 rows, 5 classes, fp16 on GPU) and with a small random network (`tests/test_edit.py`,
fp32 on CPU), streamed adds, merged shards and add-then-forget equal the one-shot build bit for bit. Record: [bench/results/gpu_check.txt](../bench/results/gpu_check.txt).
