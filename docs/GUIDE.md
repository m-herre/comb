# User guide

## Parameters

```python
CombClassifier(checkpoint=None, n_estimators=4, softmax_temperature=0.9, batch_size=8, device=None,
               use_amp="auto", random_state=42, cell_m=None, kmeans_iters=4)
```

| Parameter | Default | Meaning |
|---|---|---|
| `checkpoint` | `None` | Weights folder (`config.json` + `model.safetensors`), a `.safetensors` file or a training `.ckpt`. `None`: `COMB_CHECKPOINT`, else the released weights from Hugging Face |
| `n_estimators` | 4 | Ensemble members: views of the table with permuted features, shifted class labels and two normalisations (none / Yeo-Johnson), averaged in logit space. 2 is faster at a small cost in accuracy, 8 slightly more accurate at about 1.5x the time |
| `softmax_temperature` | 0.9 | Temperature of the softmax over the averaged logits |
| `batch_size` | 8 | Most members per forward pass (`None`: all); fit uses fewer when the table would exceed the memory budget |
| `device` | `None` | `None` / `"auto"`: CUDA if available, else CPU |
| `use_amp` | `"auto"` | fp16 autocast. `"auto"`: on every CUDA table, fp32 on the CPU |
| `random_state` | 42 | Seed of the ensemble's permutations (the model itself is deterministic) |
| `kmeans_iters` | 4 | Lloyd steps that place the cell anchors at `fit` (only when the table is compressed). Pretraining used 8; 4 scored the same in official TabArena runs and fits faster |
| `cell_m` | `None` | Cell budget M per member (checkpoint default 4096). Larger M is closer to exact attention; edits rebuild the cells in time growing with M² (about 6 ms at M = 1024, 18 ms at 4096 on an RTX PRO 6000). The weights were fine-tuned with M between 512 and 8192; above 8192 is outside that range |

Fitted attributes: `classes_`, `n_features_in_`, `n_rows_` (rows in the model, after edits), `editable_`.

## Methods

| Method | Does |
|---|---|
| `fit(X, y)` | Fits the preprocessing and the frozen state, then adds the rows |
| `predict_proba(X)`, `predict(X)` | Read the cells; cost independent of the number of training rows |
| `add(X, y)` | Adds rows (classes seen in `fit` only). Exact |
| `forget(X, y)` | Removes rows previously added by `fit`, `add` or `merge`, identified by their values. Exact |
| `empty_like()` | A copy with the same frozen state and no rows, for building shards |
| `merge(other)` | Adds the rows of a shard made by `empty_like()`. Exact |
| `to(device)` | Moves the fitted model |

See [EDITING.md](EDITING.md) for what "exact" means and the limits.

## Data

- **Input:** NumPy arrays or pandas DataFrames (a NumPy array of strings is read as a DataFrame). In DataFrames,
  `category`, `object`, `string` and `bool` columns are encoded as ordinal codes fitted in `fit`; unseen categories and missing values share one extra code (-1). Missing
  numeric values are imputed with the training mean.
- **Constant columns** in the `fit` data are dropped.
- **Classes:** 2 to 10 natively. More classes run through a hierarchical scheme (no cells, so slower on large tables
  and not editable).
- **Size:** fitting embeds every training row once (time linear in the rows); prediction reads at most
  `n_estimators x M` cells, whatever the number of training rows.
- **Columns:** a DataFrame's column names are recorded at `fit` (`feature_names_in_`); later calls may pass the
  same columns in another order, other names raise `ValueError`. NumPy arrays are matched by position.
- **Not supported:** sparse input (`TypeError`), infinities (`ValueError`; NaN is fine), `sample_weight`, regression.
  `cell_m` must be at least the number of classes.

## Saving

A fitted `CombClassifier` pickles (`pickle`, `joblib`) with its state: the cell statistics and the frozen
preprocessing (the raw training table is not stored). The network is not in the pickle: it records the weights' path
and their sha256, and loading re-attaches those exact weights, from the recorded path or else from
`COMB_CHECKPOINT` or the released weights; if none of them matches, loading raises instead of using other weights. To
move a model to another machine, make the same weights available there. Tensors are saved on the CPU, so a model
fitted on a GPU loads on a machine without one; where the GPU is available it loads back onto it, and then predicts
and edits exactly as before.

## Memory

Every pass (training rows in `fit` and `add`, test rows in `predict`) is sized from one memory budget: an eighth of
the GPU's memory, between 128 MB and 4 GB (1 GB on CPU). Past that size the GPU is saturated, so the budget costs no
speed on large GPUs and lets small ones run large tables. What stays in memory is the network (110 MB, shared by all
models in a process) and the fitted state (about 0.1-0.2 GB for the default 4 members). `fit` also holds the training rows'
representations, 2 KB per row and member (512 fp32 values); when that exceeds half the GPU memory, members are fitted
in separate passes (down to one: about 2 GB per million rows, plus the rows' inputs). `add`, `forget` and `predict` stream their rows.

## Speed tips

- Use a GPU; most of the time on small tables is fixed overhead (kernel launches), on large tables `fit`.
- `n_estimators=2` is faster at a small cost in accuracy (see [RESULTS.md](RESULTS.md)).
- Fit once, predict many times: prediction does not touch the training rows.
