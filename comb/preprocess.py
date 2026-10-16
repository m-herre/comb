# Ported from tabicl 2.2.0 (BSD-3-Clause, Copyright (c) 2025, Soda team @ Inria); see LICENSE-tabicl.
"""Default TabICL inference preprocessing: numeric encoding and the ensemble of dataset views.

Port of ``tabicl/_sklearn/preprocessing.py`` restricted to the default inference path:
normalisations ``none``/``power``, feature shuffling ``latin`` (random permutations above 4000
features, as upstream) and class shuffling ``shift``. Every call into Python's ``random`` and into
numpy happens in upstream's order, so shuffles and preprocessed arrays are bit-identical.

Pipeline per ensemble member: drop constant columns -> (x - mu) / (sigma + 1e-6) clipped to +-100
-> optional Yeo-Johnson (standardised) -> two-stage z-score outlier soft clipping -> feature
permutation (and, for classification, a circular shift of the class labels).
"""
from __future__ import annotations

import copy
import itertools
import math
import random
from collections import OrderedDict

import numpy as np
import torch
from scipy import optimize, special

from . import kernels
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing._data import _is_constant_feature
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_X_y


def _check(X, n_features=None, **kw):
    """Upstream's ``validate_data``: ``check_array`` defaults plus the feature-count check."""
    X = check_array(X, **kw)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, but the transformer was fitted with {n_features} features.")
    return X


# SciPy's private Yeo-Johnson helpers (scipy/stats/_morestats.py, BSD-3-Clause, SciPy 1.18), NumPy only. Vendored
# because their signatures change between SciPy versions; same operations, so the results are SciPy 1.18's.
def _log_var(logx, xp, axis):
    """log of the variance of x, from log(x)."""
    logmean = np.broadcast_to(special.logsumexp(logx, axis=axis, keepdims=True) - math.log(logx.shape[axis]),
                              logx.shape)
    ones = np.ones_like(logx)
    logxmu, _ = special.logsumexp(np.stack((logx, logmean), axis=0), axis=0, b=np.stack((ones, -ones), axis=0),
                                  return_sign=True)
    return special.logsumexp(2 * logxmu, axis=axis) - math.log(logx.shape[axis])


def _yeojohnson_transform(x, lmbda, xp=None):
    """x transformed by the Yeo-Johnson power transform with parameter lmbda."""
    dtype = np.result_type(x, lmbda)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    eps = np.finfo(dtype).eps
    out = np.zeros_like(x, dtype=dtype)
    pos = x >= 0
    if abs(lmbda) < eps:
        out[pos] = np.log1p(x[pos])
    else:  # stable ((x + 1) ** lmbda - 1) / lmbda
        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda
    if abs(lmbda - 2) > eps:
        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
    else:
        out[~pos] = -np.log1p(-x[~pos])
    return out


class TransformToNumerical:
    """Make a table numeric (upstream ``TransformToNumerical``).

    DataFrame: a ``ColumnTransformer`` with the categorical columns first (string/object/category/
    boolean dtypes -> ``OrdinalEncoder(int64)``; unseen and missing values -> -1), then the numeric
    columns (mean ``SimpleImputer`` keeping all-NaN columns); columns of any other dtype are dropped.
    ndarray: only the mean imputer.

    Tables whose columns are all of the common dtypes (category / bool / object of strings, int / float64) take a
    lean path with the same arithmetic (sorted categories, codes, the imputer's masked mean): the same output, bit
    for bit, without sklearn's per-call validation, which dominates a small table's fit. Anything else uses sklearn.
    """

    def fit(self, X, y=None):
        self.fast_ = _FastNumerical.fit(X) if hasattr(X, "columns") else None
        if self.fast_ is not None:
            return self
        cat = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
        num = SimpleImputer(keep_empty_features=True)
        if not hasattr(X, "columns"):
            np.asarray(X).astype(np.float64)  # raises on non-numeric arrays, as upstream
            self.tfm_ = num
        else:
            cat_pos = [X.columns.get_loc(c) for c in
                       make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)]
            num_pos = [X.columns.get_loc(c) for c in make_column_selector(dtype_include="number")(X)]
            self.tfm_ = ColumnTransformer(transformers=[("categorical", cat, cat_pos), ("continuous", num, num_pos)])
        self.tfm_.fit(X)
        return self

    def transform(self, X):
        if getattr(self, "fast_", None) is not None:
            out = self.fast_.transform(X)
            if out is None and hasattr(X, "astype") and list(getattr(X, "columns", [])) == self.fast_.columns:
                out = self.fast_.transform(X.astype(dict(zip(self.fast_.columns, self.fast_.dtypes))))
            if out is None:
                raise ValueError("X's columns or dtypes cannot be encoded like the table given to fit")
            return out
        return self.tfm_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def _masked_mean(X):
    """``SimpleImputer(strategy="mean", keep_empty_features=True)``'s statistics, by its own arithmetic."""
    m = np.ma.mean(np.ma.masked_array(X, mask=np.isnan(X)), axis=0)
    mean = np.ma.getdata(m).copy()
    mean[np.ma.getmaskarray(m)] = 0
    return mean


class _FastNumerical:
    """The lean path of ``TransformToNumerical`` (see there); ``fit`` returns None where it does not apply."""

    NUM = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float64")

    @classmethod
    def fit(cls, X):
        import pandas as pd

        # the column split of the ColumnTransformer, by its own selectors (other columns are dropped, as there)
        cat = [X.columns.get_loc(c) for c in
               make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)]
        num = [X.columns.get_loc(c) for c in make_column_selector(dtype_include="number")(X)]
        if any(not (isinstance(X.dtypes.iloc[i], pd.CategoricalDtype) or str(X.dtypes.iloc[i]) == "object") for i in cat) \
                or any(str(X.dtypes.iloc[i]) not in cls.NUM for i in num) or len(set(cat) | set(num)) < len(cat) + len(num):
            return None  # string / nullable boolean extension dtypes, float32, ...: sklearn
        self = cls()
        self.columns, self.dtypes, self.cat, self.num = list(X.columns), list(X.dtypes), cat, num
        self.categories = []
        for i in cat:
            cats = self._categories(X.iloc[:, i])
            if cats is None:
                return None
            self.categories.append(cats)
        self.mean = _masked_mean(X.iloc[:, num].to_numpy(dtype=np.float64)) if num else None
        return self

    @staticmethod
    def _values(col, fit=False):
        """The column as OrdinalEncoder sees it (category columns as their values), and its mask of values encoded
        as missing (-1). NaN is missing; None is a category of its own when ``fit`` saw it (left to sklearn), and
        missing when it first appears after ``fit``, as in OrdinalEncoder."""
        v = np.asarray(col)
        if v.dtype == object:
            none = np.array([x is None for x in v], dtype=bool)
            if fit and none.any():
                return None, None
            miss = none | np.array([isinstance(x, float) and x != x for x in v], dtype=bool)
            if not all(isinstance(x, str) for x in v[~miss]):
                return None, None  # numbers, pd.NA or mixed objects: sklearn's rules
            return v, miss
        if v.dtype.kind in "biuf":
            return v, (np.isnan(v) if v.dtype.kind == "f" else np.zeros(len(v), dtype=bool))
        return None, None

    def _categories(self, col):
        v, miss = self._values(col, fit=True)
        if v is None:
            return None
        present = v[~miss]
        return np.array(sorted(set(present.tolist())), dtype=object) if v.dtype == object else np.unique(present)

    def transform(self, X):
        import pandas as pd

        if list(X.columns) != self.columns or list(X.dtypes) != self.dtypes:
            return None
        parts = []
        if self.cat:
            codes = np.empty((len(X), len(self.cat)), dtype=np.int64)
            for j, (i, cats) in enumerate(zip(self.cat, self.categories)):
                v, miss = self._values(X.iloc[:, i])
                if v is None:
                    return None
                c = pd.Categorical(v[~miss], categories=cats).codes.astype(np.int64)  # unseen -> -1
                codes[:, j] = -1
                codes[~miss, j] = c
            parts.append(codes)
        if self.num:
            x = X.iloc[:, self.num].to_numpy(dtype=np.float64, copy=True)
            nan = np.isnan(x)
            if nan.any():
                x[nan] = np.broadcast_to(self.mean, x.shape)[nan]
            parts.append(x)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.hstack(parts)


class UniqueFeatureFilter:
    """Drop features with a single unique training value; keep column 0 if all are constant."""

    def fit(self, X, y=None):
        X = _check(X)
        self.n_features_in_ = X.shape[1]
        if X.shape[0] <= 1:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            differs = X != X[:1]  # more than one unique value, NaN equal to NaN as in np.unique
            if X.dtype.kind in "fc":
                differs &= ~(np.isnan(X) & np.isnan(X[:1]))
            self.features_to_keep_ = differs.any(0)
            if not self.features_to_keep_.any():
                self.features_to_keep_[0] = True
        return self

    def transform(self, X):
        return _check(X, self.n_features_in_)[:, self.features_to_keep_]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class OutlierRemover:
    """Two-stage z-score bounds (nanstd ddof=1, floor 1e-6) with log1p soft clipping."""

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _check(X)
        self.n_features_in_ = X.shape[1]
        ddof = 1 if X.shape[0] > 1 else 0
        means = np.nanmean(X, axis=0)
        stds = np.maximum(np.nanstd(X, axis=0, ddof=ddof), 1e-6)
        X_clean = X.copy()
        lower, upper = means - self.threshold * stds, means + self.threshold * stds
        X_clean[np.logical_or(X < lower[np.newaxis, :], X > upper[np.newaxis, :])] = np.nan
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.maximum(np.nanstd(X_clean, axis=0, ddof=ddof), 1e-6)
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_
        return self

    def transform(self, X):
        X = _check(X, self.n_features_in_)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        return np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class CustomStandardScaler:
    """(x - mu) / (sigma + eps) with population std, clipped to [clip_min, clip_max]."""

    def __init__(self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6):
        self.clip_min, self.clip_max, self.epsilon = clip_min, clip_max, epsilon

    def fit(self, X, y=None):
        X = _check(X.reshape(-1, 1) if X.ndim == 1 else X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon
        return self

    def transform(self, X):
        is_vector = X.ndim == 1
        X = _check(X.reshape(-1, 1) if is_vector else X, self.n_features_in_)
        X = np.clip((X - self.mean_) / self.scale_, self.clip_min, self.clip_max)
        return X.reshape(-1) if is_vector else X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def _yeojohnson_normmax(x):
    """``scipy.stats.yeojohnson_normmax(x)`` for finite 1-d float64 ``x``: the same bounded Brent search
    (``optimize.fminbound``, same bounds and tolerance) on the same log-likelihood, minus the per-call
    argument handling of the public functions, which is most of the cost on wide tables."""
    if not np.all(np.isfinite(x)):
        raise ValueError("Yeo-Johnson input must be finite.")
    if np.all(x == 0):
        return 1.0
    n = x.shape[0]
    eps = np.finfo(x.dtype).eps
    pos = x >= 0
    all_pos, all_neg = bool(np.all(pos)), bool(np.all(~pos))
    log1p_x = np.log1p(x) if all_pos else None
    log1p_neg = np.log1p(-x) if all_neg else None
    sign_log = np.sum(np.sign(x) * np.log1p(np.abs(x)), axis=0)

    def neg_llf(lmb):
        # scipy.stats._morestats._yeojohnson_llf, then yeojohnson_normmax's _neg_llf
        if all_pos:
            if abs(lmb) < eps:
                logvar = np.log(np.var(log1p_x, axis=0))
            else:
                logvar = _log_var(lmb * log1p_x, np, 0) - 2 * math.log(abs(lmb))
        elif all_neg:
            if abs(lmb - 2) < eps:
                logvar = np.log(np.var(log1p_neg, axis=0))
            else:
                logvar = _log_var((2 - lmb) * log1p_neg, np, 0) - 2 * math.log(abs(2 - lmb))
        else:
            sigma = np.var(_yeojohnson_transform(x, lmb, xp=np), axis=0)
            logvar = np.log(sigma) if sigma >= np.finfo(sigma.dtype).smallest_normal else -np.inf
        llf = np.asarray(-n / 2 * logvar + (lmb - 1) * sign_log)
        llf[np.isinf(llf)] = -np.inf
        return -llf

    log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
    log_eps = np.log(np.finfo(x.dtype).eps)
    log_tiny_float = (np.log(np.finfo(x.dtype).tiny) - log_eps) / 2
    log_max_float = (np.log(np.finfo(x.dtype).max) + log_eps) / 2
    lb, ub = log_tiny_float / log1p_max_x, log_max_float / log1p_max_x
    if all_neg:
        lb, ub = 2 - ub, 2 - lb
    elif not all_pos:
        lb, ub = max(2 - ub, lb), min(2 - lb, ub)
    return optimize.fminbound(neg_llf, lb, ub, xtol=1.48e-08)


def _yeojohnson_torch(A: torch.Tensor, pos: torch.Tensor, lmb: torch.Tensor) -> torch.Tensor:
    """``scipy.stats._morestats._yeojohnson_transform`` per column, from ``A = log1p(|x|)`` and the
    sign mask ``pos = x >= 0``, with one lambda per column."""
    eps = torch.finfo(A.dtype).eps
    lp, ln = lmb[None, :], (2 - lmb)[None, :]
    y_pos = torch.where(lp.abs() < eps, A, torch.expm1(lp * A) / lp)
    y_neg = torch.where(ln.abs() > eps, -torch.expm1(ln * A) / ln, -A)
    return torch.where(pos, y_pos, y_neg)


def _mixed_bounds(x: torch.Tensor):
    """``_yeojohnson_normmax``'s search bounds for mixed-sign columns, per column."""
    finfo = np.finfo(np.float64)
    log_eps = np.log(finfo.eps)
    log1p_max_x = torch.log1p(20 * x.abs().amax(0))
    lb = (np.log(finfo.tiny) - log_eps) / 2 / log1p_max_x
    ub = (np.log(finfo.max) + log_eps) / 2 / log1p_max_x
    return torch.maximum(2 - ub, lb), torch.minimum(2 - lb, ub)


TRITON_YJ_MAX_ROWS = 32768  # up to here one Triton program per column beats the batched torch search


def _yeojohnson_lambdas_gpu(X, device) -> np.ndarray:
    """``_yeojohnson_normmax`` for the finite mixed-sign columns of a float64 matrix on a CUDA device:
    ``kernels.yeojohnson_lambdas`` (the whole search in one launch) for up to ``TRITON_YJ_MAX_ROWS``
    rows, ``_yeojohnson_normmax_mixed`` (parallel over rows too) for taller columns."""
    x = torch.as_tensor(X, dtype=torch.float64, device=device)
    if kernels.triton is not None and x.shape[0] <= TRITON_YJ_MAX_ROWS:
        return kernels.yeojohnson_lambdas(x, *_mixed_bounds(x)).cpu().numpy()
    return _yeojohnson_normmax_mixed(x, device)


def _yeojohnson_normmax_mixed(X: np.ndarray, device) -> np.ndarray:
    """``_yeojohnson_normmax`` for every column of a float64 matrix whose columns are finite and of
    mixed sign, all columns at once in torch: scipy's bounded Brent (``_minimize_scalar_bounded``,
    same bounds, ``xatol`` and ``maxfun``) run per column with ``torch.where`` for its branches, on
    scipy's mixed-sign log-likelihood."""
    x = torch.as_tensor(X, dtype=torch.float64, device=device)
    n = x.shape[0]
    A, pos = torch.log1p(x.abs()), x >= 0
    sign_log = (torch.sign(x) * A).sum(0)
    tiny = torch.finfo(torch.float64).smallest_normal

    def neg_llf(lmb):
        sigma = _yeojohnson_torch(A, pos, lmb).var(0, correction=0)
        logvar = torch.where(sigma >= tiny, sigma.log(), -math.inf)
        llf = -n / 2 * logvar + (lmb - 1) * sign_log
        return -torch.where(torch.isinf(llf), -math.inf, llf)

    a, b = _mixed_bounds(x)

    xatol, maxfun = 1.48e-08, 500
    sqrt_eps, golden_mean = math.sqrt(2.2e-16), 0.5 * (3.0 - math.sqrt(5.0))
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc.clone(), fulc.clone()
    rat, e = torch.zeros_like(a), torch.zeros_like(a)
    fx = neg_llf(xf)
    num = 1
    ffulc, fnfc = fx.clone(), fx.clone()
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * xf.abs() + xatol / 3.0
    tol2 = 2.0 * tol1
    active = (xf - xm).abs() > (tol2 - 0.5 * (b - a))
    while bool(active.any()):
        # parabolic step where scipy takes one, golden-section step otherwise
        r = (xf - nfc) * (fx - ffulc)
        q = (xf - fulc) * (fx - fnfc)
        p = (xf - fulc) * q - (xf - nfc) * r
        q = 2.0 * (q - r)
        p = torch.where(q > 0.0, -p, p)
        q = q.abs()
        parabolic = ((e.abs() > tol1) & (p.abs() < (0.5 * q * e).abs())
                     & (p > q * (a - xf)) & (p < q * (b - xf)))
        rat_par = (p + 0.0) / q
        x_par = xf + rat_par
        near_edge = ((x_par - a) < tol2) | ((b - x_par) < tol2)
        rat_par = torch.where(near_edge, tol1 * (torch.sign(xm - xf) + ((xm - xf) == 0)), rat_par)
        e_gold = torch.where(xf >= xm, a - xf, b - xf)
        e_new = torch.where(parabolic, rat, e_gold)
        rat_new = torch.where(parabolic, rat_par, golden_mean * e_gold)
        si = torch.sign(rat_new) + (rat_new == 0)
        u = xf + si * torch.maximum(rat_new.abs(), tol1)
        fu = neg_llf(u)
        num += 1

        better = fu <= fx
        c1 = (fu <= fnfc) | (nfc == xf)
        c2 = ~c1 & ((fu <= ffulc) | (fulc == xf) | (fulc == nfc))
        a_new = torch.where(better, torch.where(u >= xf, xf, a), torch.where(u < xf, u, a))
        b_new = torch.where(better, torch.where(u >= xf, b, xf), torch.where(u < xf, b, u))
        fulc_new = torch.where(better | c1, nfc, torch.where(c2, u, fulc))
        ffulc_new = torch.where(better | c1, fnfc, torch.where(c2, fu, ffulc))
        nfc_new = torch.where(better, xf, torch.where(c1, u, nfc))
        fnfc_new = torch.where(better, fx, torch.where(c1, fu, fnfc))
        xf_new, fx_new = torch.where(better, u, xf), torch.where(better, fu, fx)

        def upd(old, new):
            return torch.where(active, new, old)

        a, b, e, rat = upd(a, a_new), upd(b, b_new), upd(e, e_new), upd(rat, rat_new)
        fulc, ffulc, nfc, fnfc = upd(fulc, fulc_new), upd(ffulc, ffulc_new), upd(nfc, nfc_new), upd(fnfc, fnfc_new)
        xf, fx = upd(xf, xf_new), upd(fx, fx_new)
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0
        tol2 = 2.0 * tol1
        active = active & ((xf - xm).abs() > (tol2 - 0.5 * (b - a)))
        if num >= maxfun:
            break
    return xf.cpu().numpy()


MIN_BATCHED_COLUMNS = 4


class YeoJohnson:
    """``PowerTransformer(method="yeo-johnson", standardize=True)`` (scikit-learn 1.9) for float input.

    Same steps: per non-constant column, lambda by scipy's Yeo-Johnson MLE and the column transformed;
    constant columns keep lambda 1 in ``fit`` (untransformed) but are transformed in ``transform``, as
    upstream; then a ``StandardScaler``, which raises ``ValueError`` on non-finite output exactly where
    the scikit-learn version does. For float64 input, mixed-sign NaN-free columns (after the centring
    scaler, every non-constant one) get their lambdas from ``_yeojohnson_lambdas_gpu`` and the
    transforms run for all columns at once, in torch, on a CUDA ``device`` with at least
    ``MIN_BATCHED_COLUMNS`` such columns (below that, or on CPU, the scalar scipy path is faster); the
    rest use ``_yeojohnson_normmax``. Lambdas agree with the scalar path to the search tolerance
    (~1e-8; up to ~1e-6 on columns with a very flat likelihood), transformed values to ~1e-7.
    """

    def __init__(self, device=None):
        self.device = device

    def _device(self):
        device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        return device if device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    def fit_transform(self, X):
        X = check_array(X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan", copy=True)
        n_samples = X.shape[0]
        mean = np.mean(X, axis=0, dtype=np.float64)
        var = np.var(X, axis=0, dtype=np.float64)
        constant = np.array([_is_constant_feature(var[i], mean[i], n_samples) for i in range(X.shape[1])], dtype=bool)
        self.lambdas_ = np.ones(X.shape[1], dtype=X.dtype)
        batched = ~constant
        if X.dtype == np.float64 and self._device().type == "cuda":
            batched &= ~np.isnan(X).any(0) & (X >= 0).any(0) & (X < 0).any(0)
        if not (X.dtype == np.float64 and self._device().type == "cuda" and batched.sum() >= MIN_BATCHED_COLUMNS):
            batched[:] = False
        with np.errstate(invalid="ignore"):
            for i in np.flatnonzero(~constant & ~batched):
                col = X[:, i]
                self.lambdas_[i] = _yeojohnson_normmax(col[~np.isnan(col)])
                X[:, i] = _yeojohnson_transform(X[:, i], self.lambdas_[i], xp=np)
        if batched.any():
            self.lambdas_[batched] = _yeojohnson_lambdas_gpu(X[:, batched], self._device())
            X[:, batched] = self._transform_columns(X[:, batched], self.lambdas_[batched])
        self.scaler_ = StandardScaler(copy=False)
        return self.scaler_.fit_transform(X)

    def _transform_columns(self, X, lambdas, device=None):
        x = torch.as_tensor(X, dtype=torch.float64, device=device or self._device())
        lmb = torch.as_tensor(lambdas, dtype=torch.float64, device=x.device)
        return _yeojohnson_torch(torch.log1p(x.abs()), x >= 0, lmb).cpu().numpy()

    def transform(self, X, device=None):
        return self.scaler_.transform(self.power(X, device))

    def power(self, X, device=None):
        """The per-column Yeo-Johnson step of ``transform`` (before its closing scaler, which raises on inf)."""
        device = torch.device(device) if device is not None else self._device()
        cuda = X.dtype == np.float64 and device.type == "cuda" and X.shape[1] >= MIN_BATCHED_COLUMNS
        X = check_array(X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan", copy=not cuda)
        with np.errstate(invalid="ignore", over="ignore"):
            if cuda:
                X = self._transform_columns(X, self.lambdas_, device)  # NaN stays NaN, as scipy's
            else:
                for i, lmbda in enumerate(self.lambdas_):
                    X[:, i] = _yeojohnson_transform(X[:, i], lmbda, xp=np)
        return X


GPU_MIN_CELLS = 200_000  # rows x features from which PreprocessingPipeline runs in torch on a CUDA device
GPU_TABLE_COPIES = 10  # float64 copies of the table the torch pipeline holds at its peak


def _nan_mean_var(x: torch.Tensor, ddof: int = 0):
    """numpy's ``nanmean`` and ``nanvar(ddof=ddof)`` per column: mean of the non-NaN values, then the
    summed squared deviations over ``count - ddof``."""
    valid = ~torch.isnan(x)
    count = valid.sum(0)
    mean = torch.where(valid, x, 0.0).sum(0) / count
    dev = torch.where(valid, x - mean, 0.0)
    return mean, (dev * dev).sum(0) / (count - ddof)


class _TorchPipeline:
    """``PreprocessingPipeline``'s steps on a float64 tensor, as the numpy classes compute them:
    ``CustomStandardScaler``, then for ``power`` ``YeoJohnson`` (lambdas by the batched Brent for
    the mixed-sign NaN-free columns, by ``_yeojohnson_normmax`` on the host for the rare others; the
    closing ``StandardScaler`` with scikit-learn's float64 mean/variance, constant-feature rule and
    ``ValueError`` on infinite values), then ``OutlierRemover``. Reductions differ from numpy's only
    in summation order."""

    def __init__(self, power: bool, threshold: float, clip_min=-100.0, clip_max=100.0, epsilon=1e-6):
        self.power, self.threshold = power, threshold
        self.clip_min, self.clip_max, self.epsilon = clip_min, clip_max, epsilon

    PARAMS = ("mean_", "scale_", "x_min_", "x_max_", "lambdas_", "yj_mean_", "yj_scale_", "lower_", "upper_")

    def _on(self, device):
        """This pipeline with its fitted vectors on ``device`` (a shallow copy; they are kept on the CPU)."""
        p = copy.copy(self)
        for k in self.PARAMS:
            if hasattr(self, k):
                setattr(p, k, getattr(self, k).to(device))
        return p

    def fit(self, x: torch.Tensor) -> "_TorchPipeline":
        n = x.shape[0]
        self.mean_, self.scale_ = x.mean(0), x.std(0, correction=0) + self.epsilon
        x = self._scale(x)
        if self.power:
            self.x_min_, self.x_max_ = x.amin(0), x.amax(0)
            x = self._yeojohnson_fit(x, n)
        ddof = 1 if n > 1 else 0
        means, var = _nan_mean_var(x, ddof)
        stds = torch.clamp_min(var.sqrt(), 1e-6)
        lower, upper = means - self.threshold * stds, means + self.threshold * stds
        clean = torch.where((x < lower) | (x > upper), math.nan, x)
        means, var = _nan_mean_var(clean, ddof)
        stds = torch.clamp_min(var.sqrt(), 1e-6)
        self.lower_, self.upper_ = means - self.threshold * stds, means + self.threshold * stds
        self.__dict__.update(self._on("cpu").__dict__)  # tiny; the fitted state does not pin a device
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._on(x.device)._transform(x)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self._scale(x)
        if self.power:
            x = self._yeojohnson(x)
        return self._clip_outliers(x)

    def _scale(self, x):
        return torch.clamp((x - self.mean_) / self.scale_, self.clip_min, self.clip_max)

    def _clip_outliers(self, x):
        x = torch.maximum(-torch.log1p(x.abs()) + self.lower_, x)
        return torch.minimum(torch.log1p(x.abs()) + self.upper_, x)

    def _yeojohnson_fit(self, x, n):
        mean, var = x.mean(0), x.var(0, correction=0)
        eps = torch.finfo(torch.float64).eps
        constant = var <= n * eps * var + (n * mean * eps) ** 2  # sklearn's _is_constant_feature
        batched = ~constant & ~torch.isnan(x).any(0) & (x >= 0).any(0) & (x < 0).any(0)
        self.lambdas_ = torch.ones(x.shape[1], dtype=x.dtype, device=x.device)
        y = x.clone()
        other = torch.nonzero(~constant & ~batched).squeeze(1)
        if other.numel():  # same-sign or NaN columns (after the centring scaler, rarely any)
            cols = x[:, other].cpu().numpy()
            lambdas = np.ones(cols.shape[1])
            with np.errstate(invalid="ignore"):
                for j in range(cols.shape[1]):
                    col = cols[:, j]
                    lambdas[j] = _yeojohnson_normmax(col[~np.isnan(col)])
                    cols[:, j] = _yeojohnson_transform(col, lambdas[j], xp=np)
            self.lambdas_[other] = torch.from_numpy(lambdas).to(x.device)
            y[:, other] = torch.from_numpy(cols).to(x.device)
        if bool(batched.any()):
            xb = x[:, batched]
            lambdas = torch.from_numpy(_yeojohnson_lambdas_gpu(xb, x.device)).to(x.device)
            self.lambdas_[batched] = lambdas
            y[:, batched] = _yeojohnson_torch(torch.log1p(xb.abs()), xb >= 0, lambdas)
        _check_finite(y)
        valid = ~torch.isnan(y)
        count = valid.sum(0)
        total = torch.where(valid, y, 0.0).sum(0)
        dev = torch.where(valid, y - total / count, 0.0)
        # sklearn's _incremental_mean_and_var (first batch): corrected two-pass variance
        var = ((dev * dev).sum(0) - dev.sum(0) ** 2 / count) / count
        self.yj_mean_ = total / count
        constant = var <= count * eps * var + (count * self.yj_mean_ * eps) ** 2
        self.yj_scale_ = torch.where(constant, 1.0, var.sqrt())
        return (y - self.yj_mean_) / self.yj_scale_

    def _yeojohnson(self, x):
        y = _yeojohnson_torch(torch.log1p(x.abs()), x >= 0, self.lambdas_)
        over = torch.isinf(y).any(1)
        if bool(over.any()):
            # rare: an outlier absent from fit() overflows; clip that row (only) to the train range, so a row's
            # output never depends on the other rows passed with it (exact edits rely on that)
            xc = torch.minimum(torch.maximum(x[over], self.x_min_), self.x_max_)
            y[over] = _yeojohnson_torch(torch.log1p(xc.abs()), xc >= 0, self.lambdas_)
            _check_finite(y)  # still infinite after the clip: raise as upstream (no inf without an overflow)
        return (y - self.yj_mean_) / self.yj_scale_


def _check_finite(y):
    if bool(torch.isinf(y).any()):  # StandardScaler's check_array(ensure_all_finite="allow-nan")
        raise ValueError("Input X contains infinity or a value too large for dtype('float64').")


class PreprocessingPipeline:
    """Standard scaling -> ``none``/``power`` normalisation -> outlier removal.

    ``power`` is ``PowerTransformer(yeo-johnson, standardize=True)``; a row it would overflow at transform time
    (an outlier absent from ``fit``) is clipped to the scaled training range first. Upstream clips the whole input
    in that case; per row, a row's output never depends on the rows passed with it (exact edits rely on that).
    float64 tables of at least ``GPU_MIN_CELLS`` cells with a CUDA ``device`` run as ``_TorchPipeline``
    on that device: ``transform``'s output is then a float64 tensor there. Nothing of the train rows is kept.
    """

    def __init__(self, normalization_method: str = "power", outlier_threshold: float = 4.0, random_state=None,
                 device=None):
        if normalization_method not in ("none", "power"):
            raise NotImplementedError(f"normalization method {normalization_method!r} is not ported")
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.device = device

    def _torch_device(self, X):
        device = torch.device(self.device) if self.device is not None else None
        numeric = X.dtype == np.float64 or X.dtype.kind in "iub"  # numpy computes these in float64 too
        if device is None or device.type != "cuda" or not torch.cuda.is_available() or not numeric:
            return None
        # about ten float64 copies of the table: within an eighth of the device memory, else numpy
        fits = X.size * 8 * GPU_TABLE_COPIES <= torch.cuda.get_device_properties(device).total_memory // 8
        return device if GPU_MIN_CELLS <= X.size and fits else None

    def fit(self, X, y=None):
        X = _check(X)
        self.n_features_in_ = X.shape[1]
        if X.dtype == np.float16:
            X = X.astype(np.float32)
        device = self._torch_device(X)
        if device is not None:
            self.torch_ = _TorchPipeline(self.normalization_method == "power", self.outlier_threshold)
            self.torch_.fit(torch.from_numpy(np.asarray(X, np.float64)).to(device))
            return self
        self.torch_ = None
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)
        if self.normalization_method == "power":
            self.normalizer_ = YeoJohnson(device=self.device)
            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold).fit(X_normalized)
        return self

    def transform(self, X, device=None):
        """``X`` transformed; torch steps run on ``device`` (the model's; default CPU)."""
        X = _check(X, self.n_features_in_)  # no copy: every step below writes to new arrays
        device = torch.device(device) if device is not None else torch.device("cpu")
        if X.dtype == np.float16:
            X = X.astype(np.float32)
        if self.torch_ is not None:
            return self.torch_.transform(torch.from_numpy(np.asarray(X, np.float64)).to(device))
        X = self.standard_scaler_.transform(X)
        if self.normalizer_ is not None:
            # rare: an outlier absent from fit() overflows; clip that row (only) to the train range, so a row's
            # output never depends on the other rows passed with it (exact edits rely on that)
            P = self.normalizer_.power(X, device)
            over = np.isinf(P).any(1)
            if over.any():
                P[over] = self.normalizer_.power(np.clip(X[over], self.X_min_, self.X_max_), device)
            X = self.normalizer_.scaler_.transform(P)
        return self.outlier_remover_.transform(X)


class Shuffler:
    """Permutation patterns: ``none``, ``shift`` (all circular shifts), ``latin`` (random Latin
    square from ``random.Random(random_state)``; random permutations above ``max_elements_for_latin``)."""

    def __init__(self, n_elements: int, method: str = "latin", max_elements_for_latin: int = 4000, random_state=None):
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int):
        self.rng_ = random.Random(self.random_state)
        indices = list(range(self.n_elements))
        method = self.method
        if self.n_elements > self.max_elements_for_latin and method == "latin":
            method = "random"
        if method == "none" or n_estimators == 1:
            return [indices]
        if method == "shift":
            return [indices[-i:] + indices[:-i] for i in range(self.n_elements)]
        if method == "random":  # reached only via the latin fallback (n_elements > 4000)
            return [self.rng_.sample(indices, self.n_elements) for _ in range(n_estimators)]
        if method == "latin":
            return self._latin_squares()
        raise NotImplementedError(f"shuffle method {method!r} is not ported")

    def _latin_squares(self):
        """Upstream's random Latin square (recursive construction, then its rows and its columns
        shuffled), as a ``_LatinColumns`` sequence that builds each returned column on access.

        Upstream builds the whole n x n square and returns all n columns, but an ensemble uses only a
        few. The random draws are the same calls in the same order: one ``choice`` per size n..2,
        then the row shuffle and the column shuffle, whose calls depend only on the length.
        """
        symbols = list(range(self.n_elements))
        inserted = []  # the symbol put on the diagonal when growing to size n, for n = n_elements..2
        for _ in range(self.n_elements, 1, -1):
            sym = self.rng_.choice(symbols)
            symbols.remove(sym)
            inserted.append(sym)
        rows, cols = list(range(self.n_elements)), list(range(self.n_elements))
        self.rng_.shuffle(rows)
        self.rng_.shuffle(cols)
        return _LatinColumns(inserted, symbols[0], rows, cols)


class _LatinColumns:
    """Upstream's shuffled Latin square, column ``k`` = column ``cols[k]`` of the row-shuffled square.

    The size-n square is the size-(n-1) square with its row 0 appended again and symbol ``s_n``
    inserted at position i of row i. Unrolling that recursion, entry (i, j) of the size-n square is
    ``seq[(j - i) mod n]`` with ``seq = [s_n, s_(n-1), ..., s_2, last]``: a cyclic square.
    """

    def __init__(self, inserted, last, rows, cols):
        self.seq = np.asarray(list(inserted) + [last])
        self.rows, self.cols = np.asarray(rows), cols

    def __len__(self):
        return len(self.cols)

    def __getitem__(self, k):
        return self.seq[(self.cols[k] - self.rows) % len(self.rows)].tolist()


class EnsembleGenerator:
    """Ensemble of dataset views (upstream ``EnsembleGenerator``, default inference path).

    ``fit`` drops constant features, draws the feature shuffles (latin) and, for classification,
    the class shuffles (all circular shifts), shuffles their product with
    ``random.Random(random_state)``, pairs it with ``norm_methods`` and keeps the first
    ``n_estimators`` members; then fits one ``PreprocessingPipeline`` per used norm method.

    Members are grouped by norm method in ``norm_methods`` order (upstream iterates
    ``list(set(...))``, whose order is hash-dependent; within a group, member order is identical).

    Fitted attributes (``OrderedDict`` keyed by norm method, in ``norm_methods`` order):
    ``ensemble_configs_[m]`` = list of ``(feat_shuffle, class_shuffle_or_None)``; member j of group m uses
    ``X[:, feat_shuffle]`` and labels ``np.array(class_shuffle)[y]``.
    """

    def __init__(self, n_estimators: int = 8, norm_methods=("none", "power"), feat_shuffle_method: str = "latin",
                 class_shuffle_method: str = "shift", outlier_threshold: float = 4.0, random_state=42,
                 classification: bool = True, device=None):
        if feat_shuffle_method != "latin":
            raise NotImplementedError(f"feat_shuffle_method {feat_shuffle_method!r} is not ported")
        if class_shuffle_method != "shift":
            raise NotImplementedError(f"class_shuffle_method {class_shuffle_method!r} is not ported")
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.classification = classification
        self.device = device  # torch device for the batched Yeo-Johnson fit

    def fit(self, X, y):
        check_X_y(X, y)
        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        elif isinstance(self.norm_methods, str):
            self.norm_methods_ = [self.norm_methods]
        else:
            self.norm_methods_ = list(self.norm_methods)
        for m in self.norm_methods_:
            if m not in ("none", "power"):
                raise NotImplementedError(f"normalization method {m!r} is not ported")
        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)
        self.n_features_in_ = X.shape[1]
        if self.classification:
            self.n_classes_ = len(np.unique(y))
        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_ = self._generate_ensemble()
        self.preprocessors_ = {}
        for m in self.ensemble_configs_:
            self.preprocessors_[m] = PreprocessingPipeline(
                normalization_method=m, outlier_threshold=self.outlier_threshold, random_state=self.random_state,
                device=self.device,
            ).fit(X)
        return self

    def _generate_ensemble(self):
        X_shuffles = Shuffler(self.n_features_in_, self.feat_shuffle_method,
                              random_state=self.random_state).shuffle(self.n_estimators)
        if self.classification:
            y_patterns = Shuffler(self.n_classes_, self.class_shuffle_method,
                                  random_state=self.random_state).shuffle(self.n_estimators)
        else:
            y_patterns = [None]
        # Shuffle (feature, class) index pairs instead of the patterns: the same random calls, and only
        # the members kept below build their feature permutation.
        shuffle_configs = list(itertools.product(range(len(X_shuffles)), range(len(y_patterns))))
        self.rng_.shuffle(shuffle_configs)
        members = itertools.islice(itertools.product(shuffle_configs, self.norm_methods_), self.n_estimators)
        members = [((X_shuffles[a], y_patterns[b]), m) for (a, b), m in members]

        configs = OrderedDict()
        for method in dict.fromkeys(self.norm_methods_):  # fixed order instead of upstream's list(set(...))
            group = [c for c, m in members if m == method]
            if group:
                configs[method] = group
        return configs
