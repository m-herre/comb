# Editing a fitted Comb model

`CombClassifier` keeps its training set as exact statistics, so rows can be added, forgotten and merged after `fit`
without retraining and without the other rows.

```python
clf = CombClassifier().fit(X, y)       # freezes the state below, then adds the rows
clf.add(X_new, y_new)                  # exact
clf.forget(X_old, y_old)               # exact
shard = clf.empty_like().add(X_b, y_b) # same frozen state, its own rows
clf.merge(shard)                       # exact
clf.n_rows_                            # rows currently in the model
```

## What `fit` freezes

Per ensemble member, everything that depends on the training table is fixed at `fit` and reused by every edit:

| Stage | Frozen state |
|---|---|
| Preprocessing | label encoder (the classes), categorical encoder and mean imputer, constant-feature mask, standard scaler, Yeo-Johnson parameters, outlier bounds; the member's feature permutation and class shift; the precision (fp16 or fp32) |
| Column stage | the set transformer's inducing-point states, computed from the `fit` rows. A row's embedding reads only these states, never the other rows |
| In-context stage | the cell anchors: per class, k-means centroids of the `fit` rows' representations (M = 4096 cells in total by default) |

What changes with edits is only the statistics: for every cell, its row count `n` (an integer) and the sum `s` of its
rows' representations (float64). The model reads cell `p` as the pseudo-row `s / n` plus its class embedding, with
`log n` added to every attention logit over it; the attention's length scale uses the total row count.

## Why the edits are exact

A row's cell and its contribution `(1, r)` to that cell depend only on the row and the frozen state: its representation
`r` is computed in blocks of a fixed number of rows (set by `fit` from the memory budget, the last block padded),
and its nearest anchor in fixed blocks of 1024. Preprocessing is per row too: a row so far outside the `fit` range
that Yeo-Johnson overflows is clipped to that range on its own (upstream TabICL clips the whole input in that
case, so the two differ only on such inputs). Classes whose rows were all forgotten get probability 0. So

- `add(rows)` adds the rows' contributions to the statistics,
- `forget(rows)` subtracts them,
- `merge(shard)` adds the shard's statistics,

and the result is the statistics of the final row set, up to the order of float64 additions. The in-context stage then
re-reads the at most M cells (cost independent of the number of rows).

**Measured**: with the released weights on an RTX PRO 6000 in fp16 (`bench/gpu_check.py`, 20,000 rows in chunks
of 5,000) and with a small random network on CPU in fp32 (`tests/test_edit.py`), streamed adds, merged shards and
add-then-forget give logits **bit-identical** to building the same rows in one go (max |difference| 0.0).

## The reference is the frozen state, not a refit

An edited model equals a model built from the final rows **on the frozen state of the original `fit`**. It is not the
model a fresh `fit` on the final rows would give: that would refit the preprocessing, the column stage and the anchors.
When the data drift far from the `fit` rows, a periodic refit restores that.

Practical consequences:

- `fit` on a representative sample. The frozen state is fitted on the `fit` rows only; on tables of a few hundred rows,
  fitting it on a small fraction of the final data costs accuracy.
- Tables that `fit` did not compress (at most M rows: one cell per row) become compressed after edits: new rows
  join the cell of their nearest anchor. `add` warns once when the model holds more than max(M, 4 x the `fit` rows):
  the cells stay those of the `fit` rows, and a refit on the full data may predict better.

## Limits

- **No new classes.** Anchors exist only for classes seen in `fit`; `add` raises `ValueError` for an unseen label.
- **At most 10 classes.** Above that, Comb predicts through a hierarchical fallback that does not keep cells, and edits
  raise `NotImplementedError`. `clf.editable_` tells.
- **One device.** Bit-exactness holds for edits made on the device (and precision) of `fit`. After `clf.to(device)`
  edits are exact relative to the moved state, but match edits made before the move only up to the new device's
  arithmetic (fp32 on CPU where `fit` ran fp16 on CUDA). Moving back to CUDA restores fp16.
- **`forget` identifies rows by their values.** Forgetting a row that was never added removes a contribution that was
  never there. It is only detected when it would empty a cell below zero (`ValueError`, and the model is unchanged).
- **`merge` needs the same frozen state**: shards made with `empty_like()` from one fitted model (or copies of it).
  Merging two independently fitted models raises `ValueError`, and so does `clf.merge(clf)` (every row would count
  twice). Merging the same shard twice is not detected.

## Use cases

- **Streams.** `add` new batches as they arrive; prediction cost does not grow.
- **Deletion requests (machine unlearning).** After `forget`, the rows' contributions are gone from the model state; the
  model is the one that would have been built without them (on the same frozen state). Note that the frozen state
  itself was fitted on the `fit` rows: if those must be forgotten too, refit.
- **Sharded or federated data.** Each site builds a shard from a shared `empty_like()` copy; only the cell statistics
  (count and sum per cell, `n_estimators x M x 512` numbers) travel, never rows. Cell sums are aggregates, not
  anonymised data: small cells can reveal individual rows.
