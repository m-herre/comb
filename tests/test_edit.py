"""Exact editing: every edit equals building the final row set in one go on the same frozen state (CPU, tiny model)."""
import pickle

import numpy as np
import pytest

from comb import CombClassifier
from tiny import tiny_checkpoint


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("w") / "tiny.ckpt")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(700, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1)
    return [(X[i:i + 100], y[i:i + 100]) for i in range(0, 600, 100)], X[600:]


def _fit(ckpt, X, y, **kw):
    return CombClassifier(ckpt, n_estimators=4, device="cpu", **kw).fit(X, y)


def _stack(chunks):
    return np.vstack([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


def test_fit_is_deterministic(ckpt, data):
    chunks, Xt = data
    a, b = _fit(ckpt, *chunks[0]), _fit(ckpt, *chunks[0])
    assert np.array_equal(a._member_logits(Xt), b._member_logits(Xt))


def test_streamed_adds_equal_one_shot(ckpt, data):
    chunks, Xt = data
    streamed = _fit(ckpt, *chunks[0])
    for c in chunks[1:]:
        streamed.add(*c)
    oneshot = _fit(ckpt, *chunks[0]).add(*_stack(chunks[1:]))
    assert np.array_equal(streamed._member_logits(Xt), oneshot._member_logits(Xt))
    assert streamed.n_rows_ == oneshot.n_rows_ == 600


def test_forget_undoes_add(ckpt, data):
    chunks, Xt = data
    base = _fit(ckpt, *chunks[0]).add(*chunks[1])
    edited = _fit(ckpt, *chunks[0]).add(*chunks[1]).add(*chunks[2]).forget(*chunks[2])
    assert np.array_equal(edited._member_logits(Xt), base._member_logits(Xt))
    for b, e in zip(base.batches_, edited.batches_):
        assert np.array_equal(b.count.numpy(), e.count.numpy())


def test_forget_rows_from_fit(ckpt, data):
    chunks, Xt = data
    full = _fit(ckpt, *_stack(chunks[:2]))
    reference = full.empty_like().add(*chunks[1])
    assert np.abs(full.forget(*chunks[0])._member_logits(Xt) - reference._member_logits(Xt)).max() < 1e-5


def test_merge_shards_equals_one_shot(ckpt, data):
    chunks, Xt = data
    model = _fit(ckpt, *chunks[0])
    shards = [model.empty_like().add(*c) for c in chunks[1:]]
    for s in shards:
        model.merge(s)
    oneshot = _fit(ckpt, *chunks[0]).add(*_stack(chunks[1:]))
    assert np.array_equal(model._member_logits(Xt), oneshot._member_logits(Xt))


def test_merge_needs_same_frozen_state(ckpt, data):
    chunks, _ = data
    with pytest.raises(ValueError, match="same frozen state"):
        _fit(ckpt, *chunks[0]).merge(_fit(ckpt, *chunks[1]))


def test_new_class_is_rejected(ckpt, data):
    chunks, _ = data
    with pytest.raises(ValueError, match="new classes"):
        _fit(ckpt, *chunks[0]).add(chunks[1][0][:2], [7, 7])


def test_forget_more_than_present_raises_and_changes_nothing(ckpt, data):
    chunks, Xt = data
    model = _fit(ckpt, *chunks[0])
    before = model._member_logits(Xt)
    with pytest.raises(ValueError, match="negative count"):
        model.forget(*_stack([chunks[0], chunks[0]]))
    assert np.array_equal(model._member_logits(Xt), before)


def test_empty_model_cannot_predict(ckpt, data):
    chunks, Xt = data
    with pytest.raises(RuntimeError, match="no rows"):
        _fit(ckpt, *chunks[0]).empty_like().predict(Xt)


def test_many_classes_fall_back_and_refuse_edits(ckpt):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(120, 4)), np.arange(120) % 13
    model = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(X, y)
    assert model.predict_proba(X[:5]).shape == (5, 13) and not model.editable_
    with pytest.raises(NotImplementedError):
        model.add(X[:2], y[:2])


def test_pickle_keeps_edits(ckpt, data):
    chunks, Xt = data
    model = _fit(ckpt, *chunks[0]).add(*chunks[1])
    again = pickle.loads(pickle.dumps(model)).add(*chunks[2])
    assert np.abs(again._member_logits(Xt) - model.add(*chunks[2])._member_logits(Xt)).max() < 1e-6


def test_forgetting_a_class_gives_it_probability_zero(ckpt, data):
    chunks, Xt = data
    X, y = _stack(chunks[:2])
    model = _fit(ckpt, *chunks[0]).add(*chunks[1]).forget(X[y == 2], y[y == 2])
    p = model.predict_proba(Xt)
    assert p.shape[1] == 3 and np.all(p[:, 2] == 0) and np.allclose(p.sum(1), 1)
    assert np.all(_fit(ckpt, *chunks[0]).predict_proba(Xt)[:, 2] > 0)  # unedited: every class has mass


def _forced_overflow():
    """A fitted pipeline whose first column overflows Yeo-Johnson at 60 (the rare clip-to-train-range path), a row
    outside the train range that does not overflow, and one that does."""
    from comb.preprocess import PreprocessingPipeline
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    pipe = PreprocessingPipeline().fit(X)
    pipe.normalizer_.lambdas_[0] = 200.0
    return pipe, np.array([[0.0, 3 * X[:, 1].max()]]), np.array([[60.0, 0.0]])


def test_overflow_clip_is_per_row():
    pipe, outside, overflow = _forced_overflow()
    alone = pipe.transform(outside)
    assert np.array_equal(pipe.transform(np.vstack([outside, overflow]))[:1], alone)
    assert np.isfinite(pipe.transform(overflow)).all()


def test_overflow_clip_is_per_row_torch():
    """The torch pipeline (``fit`` needs CUDA; ``transform`` runs anywhere) on a hand-set fitted state."""
    import torch
    from comb.preprocess import _TorchPipeline
    f64 = dict(dtype=torch.float64)
    pipe = _TorchPipeline(power=True, threshold=4.0)
    pipe.mean_, pipe.scale_ = torch.zeros(2, **f64), torch.ones(2, **f64)
    pipe.x_min_, pipe.x_max_ = torch.full((2,), -3.0, **f64), torch.full((2,), 3.0, **f64)
    pipe.lambdas_ = torch.tensor([200.0, 1.0], **f64)
    pipe.yj_mean_, pipe.yj_scale_ = torch.zeros(2, **f64), torch.ones(2, **f64)
    pipe.lower_, pipe.upper_ = torch.full((2,), -1e9, **f64), torch.full((2,), 1e9, **f64)
    outside = torch.tensor([[0.0, 9.0]], **f64)  # outside the train range in column 1, no overflow
    overflow = torch.tensor([[60.0, 0.0]], **f64)
    alone = pipe.transform(outside)
    assert torch.equal(pipe.transform(torch.cat([outside, overflow]))[:1], alone)
    assert torch.isfinite(pipe.transform(overflow)).all()


def test_merge_self_raises_and_growth_warns(ckpt):
    import warnings

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    model = CombClassifier(ckpt, n_estimators=2, device="cpu").fit(X[:10], y[:10])
    with pytest.raises(ValueError, match="merge"):
        model.merge(model)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.add(X[10:12], y[10:12])
        assert not w
        model.add(X[12:], y[12:])  # 40 rows > max(M = 16, 4 x 10 fit rows)? 40 = 40: not yet
        assert not w
        model.add(X[:1], y[:1])
        assert len(w) == 1 and "refit" in str(w[0].message)
        model.add(X[1:2], y[1:2])
        assert len(w) == 1  # once per model


def test_pickle_refuses_other_weights(ckpt, tmp_path):
    from tiny import tiny_checkpoint

    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    path = tiny_checkpoint(tmp_path / "w.ckpt", seed=0)
    blob = pickle.dumps(CombClassifier(path, n_estimators=2, device="cpu").fit(X, y))
    tiny_checkpoint(tmp_path / "w.ckpt", seed=1)  # the same file, other weights
    with pytest.raises(ValueError, match="changed"):
        pickle.loads(blob)
    (tmp_path / "w.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="COMB_CHECKPOINT"):
        pickle.loads(blob)


def test_released_weights_must_match_the_version(tmp_path, monkeypatch):
    import sys
    import types

    from comb import checkpoint
    from tiny import tiny_checkpoint

    path = tiny_checkpoint(tmp_path / "w.ckpt", seed=0)
    hub = types.SimpleNamespace(snapshot_download=lambda *a, **k: str(path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("COMB_CHECKPOINT", raising=False)
    checkpoint._released.cache_clear()
    try:
        with pytest.raises(ValueError, match="not this version's"):
            checkpoint.resolve(None)
        monkeypatch.setattr(checkpoint, "SHA256", checkpoint.weights_digest(path))
        checkpoint._released.cache_clear()
        assert str(checkpoint.resolve(None)) == path
    finally:
        checkpoint._released.cache_clear()
