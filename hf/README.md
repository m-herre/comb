---
license: apache-2.0
library_name: tabcomb
pipeline_tag: tabular-classification
tags:
  - tabular
  - in-context-learning
  - tabular-foundation-model
  - machine-unlearning
---

<img src="https://raw.githubusercontent.com/m-herre/comb/main/docs/assets/logo.svg" alt="Comb" width="360">

# Comb

Weights of **Comb**, a tabular in-context classifier whose training set can be edited exactly: add, forget and merge
rows without retraining. Code, docs and results: **https://github.com/m-herre/comb**.

```bash
pip install tabcomb
```

```python
from comb import CombClassifier

clf = CombClassifier().fit(X_train, y_train)   # downloads these weights on first use
proba = clf.predict_proba(X_test)
clf.add(X_new, y_new); clf.forget(X_old, y_old)
```

| | |
|---|---|
| Architecture | TabICLv2 network (column set transformer, row transformer, 12-block in-context stage) with the in-context stage over at most M = 4096 class-conditional cells (default; set with `cell_m`) |
| Parameters | 27,552,258 (fp32, 110 MB) |
| Pretraining | from scratch on synthetic data only (TabICLv2's structural causal prior), 500k + 40k + 10k steps with the cell budget drawn from 128 to 2048 per step, then a 5k-step fine-tune with M from 512 to 8192 and a more realistic prior mix |
| Task | classification, 2 to 10 classes (more through a hierarchical fallback) |
| TALENT, 106 held-out classification tasks (8 members, fp32) | AUC −0.37 pp [−0.58, −0.19] vs TabICLv2, +2.8 pp vs CatBoost, +3.6 pp vs XGBoost (their defaults) |
| TabArena-Lite / BeyondArena Elo (self-run, official protocol, defaults) | 1443 / 1167 |
| Version | 1.0.0 |
| License | Apache-2.0 (code and weights); architecture ported from TabICL (BSD-3-Clause) |

Files: `config.json` (constructor arguments of `comb.model.TabICL`) and `model.safetensors`.
