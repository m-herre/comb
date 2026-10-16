# Changelog

## 1.0.0 (2026-10-16)

First release.

Model:

- `CombClassifier`: scikit-learn classifier with `add`, `forget`, `empty_like` and `merge`, exact with respect to the
  state frozen at `fit`. A class whose rows were all forgotten gets probability 0.
- Weights `cell-clf-v3-s4`: 27.6M parameters (TabICLv2's architecture), pretrained from scratch on synthetic data
  (500k + 40k + 10k steps), then fine-tuned for 5k steps with the cell budget drawn from 512 to 8192 and half of the
  tables from a more realistic prior (imbalance, high-cardinality categoricals, missing values).
- Defaults chosen in TabArena / BeyondArena runs: cell budget M = 4096 (from the weights' `config.json`),
  `n_estimators=4`, fp16 on every CUDA table (`use_amp="auto"`), 4 k-means steps at inference (training uses 8).
- `fit` builds the cells once; `predict_proba` reads only the cells. Without compression (N <= M) no k-means steps
  run.

Results:

- Official-protocol TabArena-Lite / BeyondArena Elo at the defaults 1443 / 1167; on the fast end of the Pareto front
  of both leaderboards' verified entries.
- Held-out TALENT (106 tasks, pre-registered, 8 members in fp32): AUC −0.37 pp [−0.58, −0.19] vs TabICLv2,
  +2.8 pp vs CatBoost and +3.6 pp vs XGBoost at their defaults; at the 4-member defaults −0.45 pp vs TabICLv2.

Engineering:

- Every pass (training rows in `fit` and edits, test rows in `predict`) is sized from a memory budget of the device;
  edits and prediction stream their rows.
- The network is built once per weights file and device and shared by every model in a process, including across
  device moves. Pickles hold neither the network (re-attached from the weights file on load) nor a copy of the
  training table, and load on CPU-only machines.
- Fused Triton kernels for fp16 inference on CUDA, with PyTorch fallbacks; cached scalable-softmax scales.
- The released weights are checked against their sha256 after download.
- Pretraining code in `comb/train/` (source checkout).
- `bench/`: GPU checks, timing and memory, the TALENT pre-registration, manifest, run and analysis (`bench/talent/`;
  every setting written out, weights checked by sha256), Comb as a TabArena model with the
  sweep harness and Pareto plots (`bench/tabarena/`), with the results in `bench/results/`.
