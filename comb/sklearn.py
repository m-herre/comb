# Ensemble, preprocessing and aggregation ported from tabicl 2.2.0 (BSD-3-Clause, Copyright (c) 2025, Soda team @
# Inria); see LICENSE-tabicl.
"""``CombClassifier``: a scikit-learn classifier whose fitted state can be edited exactly.

``fit`` freezes everything that depends on the training table -- per ensemble member the preprocessing (encoders,
imputer, scaler, Yeo-Johnson, outlier bounds, feature permutation, class shift), the column stage's inducing-point
states and the class-conditional cell anchors -- and then adds the training rows as exact per-cell statistics (row
count and float64 sum of the row representations). ``predict_proba`` reads only the <= M cells, so its cost does
not grow with the number of training rows.

``add`` / ``forget`` change those statistics by the given rows; ``empty_like`` and ``merge`` build and combine
shards. Every edit equals building the final row set in one go on the same frozen state (see docs/EDITING.md).
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import uuid
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from . import cell
from .cell import fit_anchors, nearest, segment_sums
from .checkpoint import find_weights, load_model, resolve, weights_digest
from .preprocess import GPU_TABLE_COPIES, EnsembleGenerator, TransformToNumerical

# Peak bytes per unit of work of the released network (measured with fp16 autocast / fp32, plus margin). Every pass is
# sized from one memory budget, so Comb runs on small GPUs; past ~2^18 tokens per pass the GPU is saturated anyway.
TOKEN_BYTES = {True: 2048, False: 5120}  # per (member, row, column) token through the column and row stages
COL_BYTES = {True: 1024, False: 2560}  # per (member, train row, column) in the column stage's set transformer (fit)
ROW_BYTES = 2048  # per (member, train row) kept during fit: the fp32 row representation
BUDGET_RANGE = (2**27, 2**32)  # bytes per pass on CUDA: an eighth of the device memory, within these bounds
CPU_BUDGET = 2**30
MERGE_GROUPS_MAX_CELLS = 1_000_000  # rows x features of one member up to which all members share forwards


def _device(device) -> torch.device:
    """``device`` with its CUDA index, so equal devices compare equal (``"cuda"`` == ``"cuda:0"``)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _softmax(x, axis=-1, temperature=0.9):
    x = x / temperature
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


@dataclass
class _Batch:
    """Ensemble members that run as one forward: their views and their exact cell statistics."""

    members: List[tuple]  # (norm method, feature permutation, class permutation) per member
    block: int = 0  # train rows per embedding pass, fixed at fit (padded): a row's representation never depends
    # on the rows embedded with it, so streamed edits equal a one-shot build to the last bit
    hiddens: Optional[List[torch.Tensor]] = None  # column-stage inducing states; None: not cacheable (fallback)
    anchors: Optional[torch.Tensor] = None  # (B, P, d) fp32
    cls: Optional[torch.Tensor] = None  # (B, P) long, -1 padding
    count: Optional[torch.Tensor] = None  # (B, P) long
    total: Optional[torch.Tensor] = None  # (B, P, d) float64
    summary: Optional[dict] = None  # the ICL blocks' states over the cells
    fallback: dict = field(default_factory=dict)  # full-forward path: preprocessed train rows and labels


class CombClassifier(ClassifierMixin, BaseEstimator):
    """Comb: a tabular in-context classifier with an exactly editable training set.

    Parameters
    ----------
    checkpoint : str or None
        Weights folder / file; None uses ``COMB_CHECKPOINT`` or downloads the released weights.
    n_estimators : int
        Ensemble members (feature permutations x class shifts x normalisations), averaged in logit space.
    softmax_temperature : float
        Temperature of the final softmax over the averaged logits.
    batch_size : int or None
        Most members per forward (None: all); fit uses fewer when the table would exceed the memory budget.
    device : str or None
        ``"auto"`` / None: CUDA if available, else CPU.
    use_amp : bool or "auto"
        fp16 autocast on CUDA; ``"auto"`` uses it on every CUDA table (fp32 on the CPU).
    random_state : int
        Seed of the ensemble's permutations.
    cell_m : int or None
        Cell budget M per member (None: the checkpoint's, 4096 for the released weights). Tables with at most M rows are not compressed.
    kmeans_iters : int
        Lloyd steps that place the cell anchors at ``fit`` (pretraining used 8; 4 scored the same and fits faster).
    """

    def __init__(self, checkpoint=None, n_estimators=4, softmax_temperature=0.9, batch_size=8, device=None,
                 use_amp="auto", random_state=42, cell_m=None, kmeans_iters=cell.INFER_KMEANS_ITERS):
        self.checkpoint = checkpoint
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.batch_size = batch_size
        self.device = device
        self.use_amp = use_amp
        self.random_state = random_state
        self.cell_m = cell_m
        self.kmeans_iters = kmeans_iters

    # -- setup ----------------------------------------------------------------------------------------------

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True  # missing values are imputed
        return tags

    def _validate_X(self, X, reset):
        if issparse(X):
            raise TypeError("Comb does not support sparse input; pass a dense array or DataFrame")
        if isinstance(X, np.ndarray) and X.ndim == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")
        if not hasattr(X, "columns"):
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
            if X.dtype.kind in "OUS":  # strings: as a DataFrame, so their columns are encoded as categories
                X = pd.DataFrame(X).infer_objects()
        num = X.select_dtypes("number").to_numpy(np.float64) if hasattr(X, "columns") else X
        if num.dtype.kind in "fc" and np.isinf(num).any():
            raise ValueError("Input X contains infinity; Comb imputes missing values (NaN) but not infinities")
        names = list(X.columns) if hasattr(X, "columns") else None
        if reset:
            self.n_features_in_ = X.shape[1]
            if names is not None and all(isinstance(c, str) for c in names):
                self.feature_names_in_ = np.asarray(names, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        elif names is not None and hasattr(self, "feature_names_in_") and names != list(self.feature_names_in_):
            fitted = list(self.feature_names_in_)
            if set(names) != set(fitted):
                missing, extra = sorted(set(fitted) - set(names)), sorted(set(names) - set(fitted), key=str)
                raise ValueError(f"X has different columns than in fit: missing {missing}, unexpected {extra}")
            X = X[fitted]  # same columns in another order
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                             f"{self.n_features_in_} features as input.")
        return X

    def _setup(self, n_samples):
        if not (self.use_amp is True or self.use_amp is False or self.use_amp == "auto"):
            raise ValueError(f"use_amp must be True, False or 'auto'; got {self.use_amp!r}")
        self.device_ = _device(("cuda" if torch.cuda.is_available() else "cpu") if self.device in (None, "auto")
                               else self.device)
        if self.use_amp is True and self.device_.type != "cuda":
            raise ValueError(f"use_amp=True is fp16 autocast on CUDA; use use_amp=False or 'auto' on {self.device_}")
        amp = True if self.use_amp == "auto" else bool(self.use_amp)  # fp16 on every CUDA table: same Elo as fp32, faster
        self.use_amp_ = amp and self.device_.type == "cuda"
        self.fit_amp_ = self.use_amp_  # the precision is part of the frozen state (``to`` keeps it where it can)
        self._load_model()
        if not (isinstance(self.kmeans_iters, (int, np.integer)) and self.kmeans_iters >= 0):
            raise ValueError(f"kmeans_iters must be a non-negative integer; got {self.kmeans_iters!r}")
        if self.cell_m is not None:
            if int(self.cell_m) < 1:
                raise ValueError(f"cell_m must be a positive integer; got {self.cell_m!r}")
            self.model_.icl_predictor.num_cells = int(self.cell_m)
        self.cell_m_ = self.model_.icl_predictor.num_cells

    def _load_model(self):
        """Build the network ``model_`` on ``device_``; depends only on the checkpoint and the device, so a host can
        run it once per process and share the network (inference never writes to it)."""
        self.checkpoint_path_ = str(resolve(self.checkpoint).resolve())  # the exact weights, for later re-attaching
        self.model_ = load_model(self.checkpoint_path_, self.device_)

    def _budget(self) -> int:
        """Bytes one pass may use: fixed per device (not the free memory), so results do not depend on other jobs."""
        if self.device_.type != "cuda":
            return CPU_BUDGET
        total = torch.cuda.get_device_properties(self.device_).total_memory
        return int(min(max(total // 8, BUDGET_RANGE[0]), BUDGET_RANGE[1]))

    def _capacity(self) -> int:
        """Bytes ``fit`` may hold for its training rows (half the device memory, a quarter of the RAM on CPU); beyond
        it, members are fitted in separate passes."""
        if self.device_.type == "cuda":
            return torch.cuda.get_device_properties(self.device_).total_memory // 2
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 4
        except (AttributeError, ValueError, OSError):
            return 2**33

    def _rows_per_pass(self, n_members, n_features) -> int:
        """Rows (train or test) per pass of ``n_members`` members through the column and row stages."""
        tokens = n_members * (n_features + self.model_.row_interactor.num_cls)
        return max(16, self._budget() // (TOKEN_BYTES[self.use_amp_] * tokens))

    def _row_chunks(self, n_rows, n_features):
        """Slices of rows to preprocess at once (its float64 temporaries within the pass budget), so ``predict`` and
        ``add`` hold only one chunk of a large input on the device."""
        step = max(1024, self._budget() // (8 * GPU_TABLE_COPIES * n_features))
        return [slice(s, s + step) for s in range(0, n_rows, step)]

    def _network(self, device):
        """``model_`` on ``device``: the process-wide network of the fitted weights file on that device (loaded once,
        shared, never moved in place) with this model's cell budget."""
        model = load_model(self.checkpoint_path_, device)
        model.icl_predictor.num_cells = self.cell_m_
        return model

    def _precision(self):
        return torch.autocast(device_type="cuda") if self.use_amp_ else contextlib.nullcontext()

    # -- views ----------------------------------------------------------------------------------------------

    def _encode_X(self, X):
        return self.ensemble_generator_.unique_filter_.transform(self.X_encoder_.transform(self._validate_X(X, False)))

    def _rows(self, X, y):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        y = np.asarray(y)
        check_consistent_length(X, y)
        return y

    def _encode_y(self, y):
        unknown = np.setdiff1d(np.unique(y), self.classes_)
        if len(unknown):
            raise ValueError(f"labels {unknown.tolist()} were not seen in fit; Comb cannot add new classes")
        return self.y_encoder_.transform(y)

    def _transform(self, Xk) -> dict:
        """Encoded rows ``Xk`` under each normalisation, float32 on the device (shared by all members)."""
        return {m: torch.as_tensor(pre.transform(Xk, self.device_)).to(self.device_, torch.float32)
                for m, pre in self.ensemble_generator_.preprocessors_.items()}

    def _views(self, batch: _Batch, Xt: dict, rows=slice(None)) -> torch.Tensor:
        """(B, n, H) member views of ``rows`` of the transformed rows ``Xt``."""
        return torch.stack([Xt[method][rows][:, list(feat)] for method, feat, _ in batch.members])

    def _labels(self, batch: _Batch, y_enc) -> torch.Tensor:
        return torch.as_tensor(np.stack([np.asarray(yp)[y_enc] for _, _, yp in batch.members]),
                               device=self.device_, dtype=torch.float32)

    def _make_batches(self, n_rows, n_features):
        gen = self.ensemble_generator_
        groups = [[(m, fs, yp) for fs, yp in gen.ensemble_configs_[m]] for m in gen.ensemble_configs_]
        if len(groups) > 1 and n_rows * n_features <= MERGE_GROUPS_MAX_CELLS:
            groups = [[mem for g in groups for mem in g]]  # small tables are launch-bound: one stack
        # fit holds every member's train rows (views, representations, the column stage's floor of one column)
        per_member = n_rows * (4 * n_features + ROW_BYTES + COL_BYTES[self.use_amp_])
        cap = max(1, self._capacity() // max(per_member, 1))
        batches = []
        for g in groups:
            n_batches = int(np.ceil(len(g) / min(self.batch_size or len(g), cap, len(g))))
            batches += [_Batch(members=[g[i] for i in idx]) for idx in np.array_split(np.arange(len(g)), n_batches)]
        return batches

    # -- the frozen featurizer and the cell statistics -------------------------------------------------------

    @torch.no_grad()
    def _embed(self, batch: _Batch, Xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """Label-free train-row representations (B, n, d) fp32 of at most ``batch.block`` rows, padded to it."""
        n, pad = Xb.shape[1], batch.block - Xb.shape[1]
        if pad:
            Xb = torch.cat([Xb, Xb[:, -1:].expand(-1, pad, -1)], 1)
            yb = torch.cat([yb, yb[:, -1:].expand(-1, pad)], 1)
        model = self.model_
        with self._precision():
            R = model.row_interactor(model.col_embedder.embed_rows(Xb, batch.hiddens, yb))
        return torch.nan_to_num(R[:, :n].float())  # an fp16 overflow stays a (huge) finite row, not a dropped one

    def _sums(self, batch: _Batch, R, yb):
        """Exact cell statistics (count (B, P), float64 sum (B, P, d)) of representations R (B, n, d)."""
        return segment_sums(R, nearest(R, batch.anchors, yb.long(), batch.cls), batch.anchors.shape[1], torch.float64)

    def _stats(self, batch: _Batch, Xt: dict, y_enc):
        """Cell statistics of rows streamed one block at a time (memory independent of the number of rows)."""
        count, total = torch.zeros_like(batch.count), torch.zeros_like(batch.total)
        for s in range(0, len(y_enc), batch.block):
            rows = slice(s, s + batch.block)
            yb = self._labels(batch, y_enc[rows])
            c, t = self._sums(batch, self._embed(batch, self._views(batch, Xt, rows), yb), yb)
            count += c
            total += t
        return count, total

    @torch.no_grad()
    def _refresh(self, batch: _Batch):
        """Rebuild the ICL summary from the cell statistics (O(L M^2), independent of the number of rows)."""
        if not bool((batch.count > 0).any(1).all()):
            batch.summary = None  # an empty member: nothing to predict from until rows are added
            return
        icl = self.model_.icl_predictor
        X, bias, n = icl.cells_from_stats(batch.count, batch.total, batch.cls)
        with self._precision():
            batch.summary = icl.summary_from_cells(X.float(), bias, n)

    # -- fit -----------------------------------------------------------------------------------------------

    def fit(self, X, y):
        X = self._validate_X(X, reset=True)
        y = self._rows(X, y)
        check_classification_targets(y)
        self._setup(X.shape[0])
        self.y_encoder_ = LabelEncoder()
        y_enc = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        if self.cell_m_ < min(self.n_classes_, self.model_.max_classes):
            raise ValueError(f"cell_m={self.cell_m_} is below the number of classes ({self.n_classes_}): every class "
                             "needs at least one cell")
        self.X_encoder_ = TransformToNumerical()
        Xn = self.X_encoder_.fit_transform(X)
        self.ensemble_generator_ = EnsembleGenerator(n_estimators=self.n_estimators, random_state=self.random_state,
                                                     classification=True, device=self.device_)
        self.ensemble_generator_.fit(Xn, y_enc)
        Xt = self._transform(self.ensemble_generator_.unique_filter_.transform(Xn))
        n, h = len(y_enc), next(iter(Xt.values())).shape[1]
        self.batches_ = self._make_batches(n, h)
        self.state_id_ = uuid.uuid4().hex  # shared by empty_like copies: what merge checks
        model, m = self.model_, self.model_.icl_predictor.num_cells
        col_elems = self._budget() // COL_BYTES[self.use_amp_]
        for batch in self.batches_:
            Xb, yb = self._views(batch, Xt), self._labels(batch, y_enc)
            with torch.no_grad(), self._precision():
                batch.hiddens = model.col_embedder.train_hiddens(Xb, yb, col_elems)
            if batch.hiddens is None:  # more classes than the head: full forward
                batch.fallback = {"X": Xb, "y": yb}
                continue
            per_pass = self._rows_per_pass(len(batch.members), h)
            batch.block = -(-n // -(-n // per_pass) // 64) * 64  # equal blocks of at most per_pass rows (64-aligned)
            del Xb
            # the rows enter exactly as ``add`` would add them; only the anchors need all of them at once
            R = torch.cat([self._embed(batch, self._views(batch, Xt, slice(s, s + batch.block)),
                                       self._labels(batch, y_enc[s:s + batch.block]))
                           for s in range(0, n, batch.block)], 1)
            batch.anchors, batch.cls = fit_anchors(R, yb.long(), m, iters=int(self.kmeans_iters),
                                                   n_classes=model.max_classes)
            batch.count = torch.zeros(batch.cls.shape, dtype=torch.long, device=self.device_)
            batch.total = torch.zeros(*batch.cls.shape, R.shape[-1], dtype=torch.float64, device=self.device_)
            for s in range(0, n, batch.block):
                c, t = self._sums(batch, R[:, s:s + batch.block], yb[:, s:s + batch.block])
                batch.count += c
                batch.total += t
            del R
            self._refresh(batch)
        self.n_rows_ = self.n_fit_rows_ = n
        return self

    @property
    def editable_(self) -> bool:
        check_is_fitted(self, "batches_")
        return all(not b.fallback for b in self.batches_)

    # -- edits ---------------------------------------------------------------------------------------------

    def _apply(self, Xk, y_enc, sign):
        stats = [None if b.fallback else (torch.zeros_like(b.count), torch.zeros_like(b.total)) for b in self.batches_]
        for rows in self._row_chunks(*Xk.shape):
            Xt = self._transform(Xk[rows])
            for batch, acc in zip(self.batches_, stats):
                if acc is not None:
                    c, t = self._stats(batch, Xt, y_enc[rows])
                    acc[0].add_(c)
                    acc[1].add_(t)
        new = []
        for batch, acc in zip(self.batches_, stats):
            if acc is None:
                new.append(None)
                continue
            count, total = acc
            count = batch.count + sign * count
            if bool((count < 0).any()):
                raise ValueError("forget: some rows are not in the model (a cell would get a negative count)")
            total = batch.total + sign * total
            new.append((count, torch.where((count > 0)[..., None], total, torch.zeros_like(total))))
        for batch, upd in zip(self.batches_, new):  # all members checked before any is changed
            if upd is not None:
                batch.count, batch.total = upd
                self._refresh(batch)
        self.n_rows_ += sign * len(y_enc)
        grown = max(self.cell_m_, 4 * self.n_fit_rows_)
        if sign > 0 and self.n_rows_ > grown and not getattr(self, "_warned_growth", False):
            self._warned_growth = True
            warnings.warn(f"the model now holds {self.n_rows_} rows but its frozen state (preprocessing, cell anchors) "
                          f"was fitted on {self.n_fit_rows_}: added rows only fill the existing cells, so a refit on "
                          "the full data may predict better", UserWarning, stacklevel=3)

    def _check_editable(self):
        check_is_fitted(self, "batches_")
        if not self.editable_:
            raise NotImplementedError("this fit cannot be edited: it has more classes than the model's head "
                                      f"({self.model_.max_classes}) and predicts through the hierarchical fallback")

    def add(self, X, y):
        """Add training rows (only classes seen in ``fit``); exact. Returns self."""
        self._check_editable()
        y = self._rows(X, y)
        if len(y):
            self._apply(self._encode_X(X), self._encode_y(y), sign=1)
        return self

    def forget(self, X, y):
        """Remove training rows previously added (by ``fit``, ``add`` or ``merge``); exact. Returns self.

        The rows are identified by their values: forgetting a row that was never added changes the model unless it
        lands in an empty cell, which raises ``ValueError``."""
        self._check_editable()
        y = self._rows(X, y)
        if len(y):
            self._apply(self._encode_X(X), self._encode_y(y), sign=-1)
        return self

    def empty_like(self):
        """A copy with the same frozen state and no rows: build a shard with ``add``, combine with ``merge``."""
        self._check_editable()
        other = type(self).__new__(type(self))
        # the frozen state (network, encoders, hiddens, anchors) is shared: nothing writes to it after fit
        other.__dict__.update(self.__dict__, n_rows_=0, batches_=[
            dataclasses.replace(b, count=torch.zeros_like(b.count), total=torch.zeros_like(b.total), summary=None)
            for b in self.batches_])
        return other

    def merge(self, other: "CombClassifier"):
        """Add the rows of ``other``, a shard made from this model by ``empty_like`` (or from the same fit). Exact."""
        self._check_editable()
        if other is self:
            raise ValueError("merge(self) would count every row twice; merge a shard made by empty_like")
        if getattr(other, "state_id_", None) != self.state_id_:
            raise ValueError("merge needs two models with the same frozen state (build shards with empty_like)")
        for a, b in zip(self.batches_, other.batches_):
            a.count = a.count + b.count.to(a.count.device)
            a.total = a.total + b.total.to(a.total.device)
            self._refresh(a)
        self.n_rows_ += other.n_rows_
        return self

    # -- predict -------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _batch_logits(self, batch: _Batch, Xt: dict) -> torch.Tensor:
        """(B, n_test, classes) logits of one member batch, in passes sized by the memory budget."""
        n_test, h = next(iter(Xt.values())).shape
        step = self._rows_per_pass(len(batch.members), h)
        if batch.fallback:
            Xtr, ytr = batch.fallback["X"], batch.fallback["y"]
            step = max(step - Xtr.shape[1], 256)
        elif batch.summary is None:
            raise RuntimeError("the model holds no rows; add some before predicting")
        cache = {"col": batch.hiddens, "slots": batch.summary, "num_classes": self.n_classes_}
        outs = []
        for s in range(0, n_test, step):
            Xb = self._views(batch, Xt, slice(s, s + step))
            with self._precision():
                out = (self.model_(torch.cat([Xtr, Xb], 1), ytr, self.softmax_temperature) if batch.fallback
                       else self.model_.decode(cache, Xb))
            outs.append(out.float().cpu())
        return torch.cat(outs, 1)

    def _member_logits(self, X) -> np.ndarray:
        """(n_members, n_test, n_classes) logits, member classes in each member's shifted order."""
        check_is_fitted(self, "batches_")
        if len(X) == 0:
            raise ValueError("Found array with 0 sample(s) while a minimum of 1 is required")
        Xk = self._encode_X(X)
        outs = []
        for rows in self._row_chunks(*Xk.shape):
            Xt = self._transform(Xk[rows])
            outs.append(torch.cat([self._batch_logits(b, Xt) for b in self.batches_]))
        return torch.cat(outs, 1).numpy()

    def predict_proba(self, X):
        outputs = self._member_logits(X)
        shuffles = [yp for b in self.batches_ for _, _, yp in b.members]
        avg = np.zeros_like(outputs[0])
        for out, shuffle in zip(outputs, shuffles):
            avg += out[..., list(shuffle)]
        avg /= len(shuffles)
        avg[..., ~self._supported()] = -np.inf  # classes whose rows were all forgotten: probability 0
        return _softmax(avg, axis=-1, temperature=self.softmax_temperature)

    def _supported(self) -> np.ndarray:
        """(n_classes,) bool: the classes the model holds rows of (all of them unless edits removed a class)."""
        batch = next((b for b in self.batches_ if not b.fallback), None)
        if batch is None:
            return np.ones(self.n_classes_, dtype=bool)
        cls, count = batch.cls[0].cpu(), batch.count[0].cpu()  # one member holds every row
        live = torch.zeros(self.n_classes_, dtype=torch.bool)
        live[cls[(cls >= 0) & (count > 0)]] = True
        # member labels are yp[y] (``_labels``): original class k is the member's class yp[k]
        return live.numpy()[list(batch.members[0][2])]

    def predict(self, X):
        check_is_fitted(self, "batches_")
        return self.y_encoder_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def to(self, device):
        """Move the fitted model and its state to ``device``. Edits stay exact relative to the state on the new
        device; they match edits made on the ``fit`` device only up to that device's arithmetic (and fp32 where
        ``fit`` ran fp16 on CUDA)."""
        check_is_fitted(self, "batches_")
        device = _device(device)
        if device != self.device_:
            self._place(device)
        return self

    def _place(self, device, refresh=True, network=True):
        self.device_ = device
        self.use_amp_ = self.fit_amp_ and device.type == "cuda"
        if network:
            self.model_ = self._network(device)
        mv = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t  # noqa: E731
        self.batches_ = [dataclasses.replace(
            b, hiddens=None if b.hiddens is None else [mv(h) for h in b.hiddens], anchors=mv(b.anchors),
            cls=mv(b.cls), count=mv(b.count), total=mv(b.total), summary=None,
            fallback={k: mv(v) for k, v in b.fallback.items()}) for b in self.batches_]
        for b in self.batches_:
            if refresh and not b.fallback:
                self._refresh(b)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "batches_" in state:
            # tensors on the CPU, so the pickle loads on any machine, and without the ICL summaries (a cache of
            # the cell statistics, rebuilt by ``__setstate__`` on the original device where it can). No network:
            # ``__setstate__`` re-attaches the weights file's shared network (the pickle is weightless)
            other = type(self).__new__(type(self))
            other.__dict__.update(self.__dict__)
            other._place(torch.device("cpu"), refresh=False, network=False)
            state.update(other.__dict__, home_device_=self.device_, weights_sha256_=weights_digest(self.checkpoint_path_))
            state.pop("model_", None)
        return state

    def __setstate__(self, state):
        home = state.pop("home_device_", None)
        digest = state.pop("weights_sha256_", None)
        self.__dict__.update(state)
        if digest is not None:  # the exact weights it was fitted with, wherever they are on this machine
            self.checkpoint_path_ = find_weights(self.checkpoint_path_, digest)
        if home is not None:
            self._place(home if home.type != "cuda" or torch.cuda.is_available() else torch.device("cpu"))

