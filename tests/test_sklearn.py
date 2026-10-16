"""scikit-learn behaviour of CombClassifier (CPU, tiny model)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comb import CombClassifier
from tiny import tiny_checkpoint


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("w") / "tiny.ckpt")


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    return X, np.where(X[:, 0] > 0, "yes", "no")


def test_predict_proba_is_a_distribution_and_labels_round_trip(ckpt):
    X, y = _data()
    clf = CombClassifier(ckpt, n_estimators=3, device="cpu").fit(X, y)
    p = clf.predict_proba(X[:20])
    assert p.shape == (20, 2) and np.allclose(p.sum(1), 1) and (p >= 0).all()
    assert set(clf.predict(X[:20])) <= {"no", "yes"} and list(clf.classes_) == ["no", "yes"]


def test_clone_pipeline_and_cross_val(ckpt):
    X, y = _data()
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu")
    assert clone(clf).get_params() == clf.get_params()
    scores = cross_val_score(make_pipeline(StandardScaler(), clf), X, y, cv=3)
    assert len(scores) == 3 and np.isfinite(scores).all()


def test_dataframe_with_categories_and_missing_values(ckpt):
    X, y = _data()
    df = pd.DataFrame(X, columns=list("abcde"))
    df["colour"] = pd.Categorical(np.random.default_rng(0).choice(["red", "green", None], len(df)))
    df.loc[::7, "a"] = np.nan
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(df, y)
    new = df.iloc[:4].copy()
    new["colour"] = pd.Categorical(["blue", "red", None, "green"])  # unseen and missing categories
    assert clf.predict_proba(new).shape == (4, 2)
    clf.add(df.iloc[:10], y[:10]).forget(df.iloc[:10], y[:10])


def test_wrong_feature_count_raises(ckpt):
    X, y = _data()
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def test_cell_budget_override(ckpt):
    X, y = _data(300)
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu", cell_m=32).fit(X, y)
    assert all(b.anchors.shape[1] == 32 for b in clf.batches_)  # the tiny checkpoint's own budget is 16


def test_moving_keeps_predictions(ckpt):
    import torch

    X, y = _data()
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(X, y)
    before = clf.predict_proba(X[:10])
    assert clf.to("cpu") is clf and clf.model_ is clf.to("cpu").model_  # already there: nothing rebuilt
    clf._place(torch.device("cpu"))  # what to() does on a real move: new network, moved state, summaries rebuilt
    assert np.array_equal(clf.predict_proba(X[:10]), before)


def test_many_classes_respect_the_temperature(ckpt):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(240, 4)), np.arange(240) % 14  # more classes than the head: hierarchical path
    clf = CombClassifier(ckpt, n_estimators=1, device="cpu", softmax_temperature=0.5).fit(X, y)
    # the hierarchy returns T log p with the caller's T, so the final softmax(. / T) gives back p (as upstream)
    assert np.allclose(np.exp(clf._member_logits(X[:5])[0] / 0.5).sum(1), 1, atol=1e-3)


def test_input_errors(ckpt):
    from sklearn.exceptions import NotFittedError

    X, y = _data()
    with pytest.raises(NotFittedError):
        CombClassifier(ckpt, device="cpu").predict(X)
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu")
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        clf.fit(X, y[:-1])
    clf.fit(X, y)
    for edit in (clf.add, clf.forget):
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            edit(X[:12], y[:10])
    assert clf.n_rows_ == len(y)


def test_rejects_inf_and_sparse(ckpt):
    import scipy.sparse as sp

    X, y = _data()
    X[3, 1] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        CombClassifier(ckpt, device="cpu").fit(X, y)
    with pytest.raises(TypeError, match="sparse"):
        CombClassifier(ckpt, device="cpu").fit(sp.csr_matrix(_data()[0]), y)


def test_string_arrays_are_categorical(ckpt):
    X, y = _data()
    Xs = np.c_[X[:, :3], np.where(X[:, 3] > 0, "a", "b")].astype(object)
    p = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(Xs, y).predict_proba(Xs[:10])
    assert np.isfinite(p).all()


def test_large_inputs_stream_in_chunks(ckpt, monkeypatch):
    import comb.sklearn as cs

    X, y = _data(3000)
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(X[:500], y[:500])
    ref = clf.predict_proba(X[500:])
    monkeypatch.setattr(cs, "CPU_BUDGET", 2**16)  # 1024-row chunks
    assert len(clf._row_chunks(2500, 5)) == 3
    assert np.allclose(clf.predict_proba(X[500:]), ref, atol=1e-6)


def test_kmeans_iters_is_a_parameter(ckpt):
    X, y = _data(n=200)
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu", kmeans_iters=8)
    assert clone(clf).get_params()["kmeans_iters"] == 8
    a8 = clf.fit(X, y).batches_[0].anchors
    a0 = CombClassifier(ckpt, n_estimators=2, device="cpu", kmeans_iters=0).fit(X, y).batches_[0].anchors
    assert not np.allclose(a8.numpy(), a0.numpy())  # N > M (16): the steps move the anchors
    with pytest.raises(ValueError, match="kmeans_iters"):
        CombClassifier(ckpt, device="cpu", kmeans_iters=-1).fit(X, y)


def test_dataframe_columns_by_name(ckpt):
    X, y = _data()
    df = pd.DataFrame(X, columns=list("abcde"))
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(df, y)
    assert list(clf.feature_names_in_) == list("abcde")
    np.testing.assert_allclose(clf.predict_proba(df[list("edcba")]), clf.predict_proba(df))  # reordered: same
    with pytest.raises(ValueError, match="different columns"):
        clf.predict_proba(df.rename(columns={"a": "z"}))


def test_zero_rows_and_small_cell_budget_raise(ckpt):
    X, y = _data()
    clf = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="0 sample"):
        clf.predict_proba(X[:0])
    y3 = np.arange(len(X)) % 3
    with pytest.raises(ValueError, match="below the number of classes"):
        CombClassifier(ckpt, n_estimators=2, device="cpu", cell_m=2).fit(X, y3)
