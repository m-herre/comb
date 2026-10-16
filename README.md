<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/logo.svg" alt="Comb" width="420">
  </picture>
</p>

<p align="center"><b>A tabular in-context classifier whose training set you can edit: add, forget and merge rows exactly, without retraining.</b></p>

<p align="center">
  <a href="https://pypi.org/project/tabcomb/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tabcomb?color=f2a516"></a>
  <a href="https://huggingface.co/m-herre/Comb"><img alt="Weights" src="https://img.shields.io/badge/weights-Hugging%20Face-f2a516"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-1c2333">
  <img alt="Parameters" src="https://img.shields.io/badge/parameters-27.6M-1c2333">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-1c2333">
  <img alt="Pretraining" src="https://img.shields.io/badge/pretrained%20on-synthetic%20data%20only-1c2333">
</p>

<p align="center">
  <a href="https://github.com/m-herre/comb/blob/main/docs/GUIDE.md">User guide</a> ·
  <a href="https://github.com/m-herre/comb/blob/main/docs/EDITING.md">Editing</a> ·
  <a href="https://github.com/m-herre/comb/blob/main/docs/RESULTS.md">Results</a> ·
  <a href="https://github.com/m-herre/comb/blob/main/docs/TRAINING.md">Training</a> ·
  <a href="https://github.com/m-herre/comb/issues">Issues</a>
</p>

---

## What it is

Comb is a pretrained transformer for classification on tables, with a scikit-learn interface. Like TabPFN and
TabICL it learns in context: `fit` stores the training table and `predict_proba` answers in one forward pass, with
no gradient steps on your data.

The difference is in what `fit` stores. Comb compresses the training rows into at most **M = 4096 cells** per
ensemble member: class-conditional clusters of the rows' learned representations, each kept as an exact row
count and sum. Prediction reads only the cells, so

- **predicting does not get slower as the training set grows**: a test row reads at most M cells, however many
  training rows there are;
- **the training set is editable**: adding rows adds to the cell sums, forgetting rows subtracts them, and two
  shards merge by adding. The result equals building the final row set in one go, with no retraining and no access
  to the other rows.

Comb is the network of [TabICLv2](https://github.com/soda-inria/tabicl) with its in-context stage run over cells
(the column and row stages are TabICLv2's), pretrained from scratch on TabICLv2's synthetic prior with cells in the
loop. Tables with at most 4096 training rows are not compressed at all.

## Benchmarks

**TabArena and BeyondArena.** TabArena's pipeline under its official protocol (8-fold bagging; BeyondArena: its
core2k split), classification, run by us on RTX PRO 6000 GPUs. Elo against the leaderboards' verified entries; times
are TabArena's medians per 1,000 rows.

| Model | TabArena-Lite Elo | train / infer (s) | BeyondArena Elo | train / infer (s) |
|---|---|---|---|---|
| TabICLv2 (published entries) | 1558 | 0.93 / 0.141 | 1189 | 11.96 / 2.256 |
| Comb, `cell_m=16384` ¹ | 1474 | 0.32 / 0.019 | 1186 | 0.64 / 0.042 |
| **Comb (defaults)** | **1443** (+89 / −84) | **0.20 / 0.014** | **1167** (+36 / −45) | 0.56 / **0.033** |
| Comb, `n_estimators=2` ¹ | 1433 | 0.14 / 0.010 | 1124 | 0.47 / 0.027 |
| LightPFN (same pipeline) | 1420 (+70 / −68) | 0.23 / 0.029 | 1096 (+33 / −38) | 0.44 / 0.072 |

¹ Run with 8 k-means steps; the defaults use 4 (same Elo, less fit time).

TabICLv2 scores higher at 5x / 21x the train time and 10x / 69x the inference time; `cell_m=16384` comes within 3 Elo
of it on BeyondArena. On both leaderboards Comb sits at the fast end of the Pareto front for train and inference time,
and `n_estimators=2` has the lowest inference time of any entry. These are self-run results, not an official TabArena
entry ([details](https://github.com/m-herre/comb/blob/main/docs/RESULTS.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/pareto_tabarena-dark.svg">
  <img src="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/pareto_tabarena.svg" alt="TabArena-Lite: Elo vs train and inference time per 1,000 rows">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/pareto_beyondarena-dark.svg">
  <img src="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/pareto_beyondarena.svg" alt="BeyondArena core2k: Elo vs train and inference time per 1,000 rows">
</picture>

**TALENT, held out.** 106 classification tasks, run once ([pre-registered](https://github.com/m-herre/comb/blob/main/bench/talent/PREREGISTRATION.md));
paired per task, 95% bootstrap CI, every model at its defaults, fp32, Comb with 8 members:

| Comb minus | mean AUC difference | 95% CI | wins / losses |
|---|---|---|---|
| TabICLv2 | −0.37 pp | [−0.58, −0.19] | 26 / 76 |
| CatBoost | +2.76 pp | [+1.87, +3.87] | 90 / 13 |
| XGBoost | +3.59 pp | [+2.65, +4.68] | 96 / 6 |

The gap to TabICLv2 is smallest on small tables (−0.21 pp below 5,000 rows) and grows with size.

**Speed.** One RTX PRO 6000, synthetic tables with 20 features, 1,000 test rows, defaults for every model
(`bench/timing.py`):

| Training rows | Comb predict | LightPFN predict | TabICLv2 predict | Comb fit | LightPFN fit | TabICLv2 fit |
|---|---|---|---|---|---|---|
| 1,000 | **0.007 s** | 0.011 s | 0.110 s | **0.013 s** | 0.070 s | 0.226 s |
| 5,000 | **0.011 s** | 0.016 s | 0.189 s | **0.056 s** | 0.129 s | 0.246 s |
| 20,000 | **0.010 s** | 0.036 s | 0.805 s | **0.135 s** | 0.646 s | 0.353 s |
| 100,000 | **0.015 s** | 0.036 s ² | 7.968 s | **0.433 s** | 0.673 s ² | 0.949 s |

² LightPFN reads at most 20,000 training rows by default and subsamples above that.

Comb's prediction time stays flat as the training set grows. Every pass is sized from a memory budget: on an 8 GB GPU,
Comb fits a million rows at 4.4 GB peak.

## Install

```bash
pip install tabcomb
```

Python 3.10+ and PyTorch 2.4+ (CPU or CUDA). The first `fit` downloads the weights (110 MB) from
[Hugging Face](https://huggingface.co/m-herre/Comb) and caches them; `COMB_CHECKPOINT=/path/to/weights` uses a local
copy instead.

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from comb import CombClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = CombClassifier().fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

A GPU is optional. DataFrames work as they are: categorical, string and boolean columns are encoded, and missing or unseen values are
handled. `CombClassifier` is a regular scikit-learn estimator (`clone`, `Pipeline`, `cross_val_score`, pickling).

## Editing

```python
clf = CombClassifier().fit(X_jan, y_jan)

clf.add(X_feb, y_feb)            # new data arrives: no refit
clf.forget(X_user, y_user)       # a deletion request: those rows leave the cells

shard = clf.empty_like()         # same frozen state, no rows
shard.add(X_site_b, y_site_b)    # e.g. built at another site, never seeing the first site's rows
clf.merge(shard)
```

Edits are exact with respect to the state frozen at `fit` (preprocessing, the column stage and the cell anchors).
Forgotten rows leave the cells; rows that were part of the original `fit` also shaped that frozen state, so removing
every trace of those needs a refit. Why the edits are exact and what they cannot do (new classes): [docs/EDITING.md](https://github.com/m-herre/comb/blob/main/docs/EDITING.md).

## Devices

| `device=` | Runs on |
|---|---|
| `None` / `"auto"` (default) | CUDA if available, else CPU |
| `"cpu"`, `"cuda"`, `"cuda:1"` | that device |

On CUDA, Comb uses fp16 autocast (fused Triton kernels where available); on the CPU it runs in fp32
(`use_amp="auto"`). A fitted model moves with `clf.to("cpu")`.

## When to use it

- Classification with 2 to 10 classes (more classes work through a hierarchical fallback, without editing).
- Tables from hundreds to millions of rows, especially when predictions are frequent or the data changes:
  streams, deletion requests (machine unlearning), sharded or federated data.
- Prefer TabPFN-3.5 or TabICLv2 when only accuracy matters and the table is small.
- Faster: `n_estimators=2`. More accurate, mostly on large tables: `cell_m=16384` (and `n_estimators=8`).

## How it works

Every ensemble member runs three stages (TabICLv2's architecture):

1. **Column stage**: each column's values are embedded by a set transformer whose inducing points read the training
   rows. Comb stores those inducing-point states at `fit`, so a new row is embedded without reading the other rows.
2. **Row stage**: a transformer across the columns of each row gives one 512-dimensional vector per row.
3. **In-context stage over cells**: per class, k-means anchors (fixed at `fit`) partition the training rows into at
   most M cells. Each cell keeps its row count n and the sum of its rows' vectors; the 12 in-context blocks read the
   cell means, with log n added to the attention logits, so a cell weighs as much as the rows it holds. Test rows read
   the cells only.

Training, fitting and predicting use the same cells, so the model was pretrained for exactly this compression
(cell budget drawn from 128 to 2048 per step, then from 512 to 8192 in a final fine-tune).

## Roadmap

- **Comb v2: regression.** The same editable cells for regression targets.
- A checkpoint with more realistic synthetic tables (class imbalance, missing values, high-cardinality categories) in
  the prior.
- Native high-cardinality categorical features.
- An official TabArena submission.

## Repository layout

| Path | Contents |
|---|---|
| `comb/` | the package: `sklearn.py` (`CombClassifier`, editing), `model.py`, `cell.py`, `preprocess.py`, `kernels.py` (Triton), `checkpoint.py` |
| `comb/train/` | pretraining: synthetic prior, trainer, optimiser (source checkout only, not in the wheel) |
| `tests/` | unit tests and exact-editing tests |
| `bench/` | GPU checks, timing and memory benchmarks, the TALENT read (`bench/talent/`: pre-registration, manifest, runner), the TabArena harness; results in `bench/results/` |
| `examples/` | quick start, streaming and forgetting, sharded fitting |
| `scripts/` | weight export, logo |

## Citation

```bibtex
@software{herre2026comb,
  author = {Herre, Markus},
  title  = {Comb: a tabular in-context classifier with an exactly editable training set},
  year   = {2026},
  url    = {https://github.com/m-herre/comb}
}
```

## License

Code and weights: [Apache-2.0](https://github.com/m-herre/comb/blob/main/LICENSE). Comb's network, preprocessing and prior are ported from
[TabICL](https://github.com/soda-inria/tabicl) (BSD-3-Clause, [LICENSE-tabicl](https://github.com/m-herre/comb/blob/main/LICENSE-tabicl)); see
[NOTICE](https://github.com/m-herre/comb/blob/main/NOTICE).
