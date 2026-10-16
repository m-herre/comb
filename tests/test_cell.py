"""Cell-ICL (Comb's ICL stage, comb/cell.py): CPU checks.

    python -m pytest tests/test_cell.py -q
"""
import torch
import pytest

from comb.cell import allocate, fit_anchors, nearest, segment_sums
from comb.model import TabICL

from tiny import TINY_CELL


def _model(cell_m, seed=0):
    """A tiny model, weights widened so differences are visible."""
    torch.manual_seed(seed)
    m = TabICL(**dict(TINY_CELL, cell_m=cell_m))
    for p in m.parameters():
        p.data.normal_(0, 0.2)
    return m.eval()


def _table(B=2, n_train=40, n_test=9, H=5, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n_train + n_test, H, generator=g)
    y = torch.randint(0, classes, (B, n_train), generator=g)
    y[:, :classes] = torch.arange(classes)
    return X, y


def test_one_cell_per_row_equals_exact_attention():
    """N <= M runs exact attention over the rows; the same rows as count-1 cells give the same outputs."""
    m = _model(cell_m=64)
    X, y = _table(n_train=40)
    ref = m(X, y)
    R = m.row_interactor(m.col_embedder(X, y))
    with torch.no_grad():
        (Xc, bias, n), count = m.icl_predictor.cells(R[:, :40], y, (R[:, :40].float(), y.long()))
        assert torch.equal(count, torch.ones_like(count)) and n == 40
        out = m.icl_predictor.decode(R[:, 40:], m.icl_predictor.summary_from_cells(Xc, bias, n))
    assert (out[..., :3] - ref).abs().max() < 1e-4


def test_duplicated_rows_equal_count_weighted_cells():
    """Every train row twice: exact attention over 2N rows (SSMax n = 2N) == N cells of count 2. Pins the
    log-count key bias and SSMax n = rows."""
    m = _model(cell_m=128)  # 2N = 80 <= M: the reference runs exact attention
    X, y = _table(n_train=40)
    Xd = torch.cat([X[:, :40], X[:, :40], X[:, 40:]], 1)
    yd = torch.cat([y, y], 1)
    ref = m(Xd, yd)
    R = m.row_interactor(m.col_embedder(Xd, yd))  # the column stage sees the 2N rows in both
    with torch.no_grad():
        anchors = (R[:, :40].float(), yd[:, :40].long())  # one anchor per distinct row
        (Xc, bias, n), count = m.icl_predictor.cells(R[:, :80], yd, anchors)
        assert torch.equal(count, torch.full_like(count, 2)) and n == 80
        out = m.icl_predictor.decode(R[:, 80:], m.icl_predictor.summary_from_cells(Xc, bias, n))
    assert (out[..., :3] - ref).abs().max() < 1e-4


def test_compressed_forward_uses_at_most_m_cells_and_is_row_order_invariant():
    cell = _model(cell_m=12)
    X, y = _table(n_train=60)
    icl = cell.icl_predictor
    icl.diag = {}
    out = cell(X, y)
    assert icl.diag["exact"] == 0.0 and icl.diag["cells"] <= 12
    icl.diag = None
    perm = torch.randperm(60, generator=torch.Generator().manual_seed(1))
    Xp = torch.cat([X[:, perm], X[:, 60:]], 1)
    assert (cell(Xp, y[:, perm]) - out).abs().max() < 1e-4


def test_no_attention_is_n_by_n():
    """Compressed (N > M): every ICL attention has <= M keys; no train row ever attends to the train rows."""
    cell = _model(cell_m=12)
    X, y = _table(n_train=60, n_test=20)
    shapes = []
    orig = type(cell.icl_predictor.tf_icl.blocks[0].attn).attend

    def spy(self, q, k, v, rope=None, *args, **kw):
        shapes.append((q.shape[-2], k.shape[-2]))
        return orig(self, q, k, v, rope, *args, **kw)

    for blk in cell.icl_predictor.tf_icl.blocks:
        blk.attn.attend = spy.__get__(blk.attn)
    cell(X, y)
    assert shapes and all(k <= 12 for _, k in shapes), shapes


def test_cached_path_equals_forward():
    for m in (12, 64):  # compressed and exact
        cell = _model(cell_m=m)
        X, y = _table(n_train=40)
        full = cell(X, y)
        with torch.no_grad():
            hid = cell.col_embedder.train_hiddens(X[:, :40], y)
            R = cell.row_interactor(cell.col_embedder.embed_rows(X[:, :40], hid, y))
            (Xc, bias, n), _ = cell.icl_predictor.cells(R, y)
            cache = {"col": hid, "slots": cell.icl_predictor.summary_from_cells(Xc.to(R.dtype), bias, n),
                     "num_classes": len(torch.unique(y[0]))}
        assert (cell.decode(cache, X[:, 40:]) - full).abs().max() < 1e-5


def test_assignment_and_sums_are_chunking_invariant():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 700, 8, generator=g)
    y = torch.randint(0, 3, (2, 700), generator=g)
    a, cls = fit_anchors(x, y, 40)
    full = nearest(x, a, y, cls)
    assert bool((full >= 0).all()) and torch.equal(cls.gather(1, full), y)  # every row in a cell of its class
    for s, e in [(0, 1), (3, 10), (17, 350), (350, 700)]:
        assert torch.equal(full[:, s:e], nearest(x[:, s:e], a, y[:, s:e], cls))
    cnt, sums = segment_sums(x, full, a.shape[1])
    ref = torch.zeros_like(sums)
    for b in range(2):
        ref[b].index_add_(0, full[b], x[b])
    assert int(cnt.sum()) == 1400 and (sums - ref).abs().max() < 1e-4


def test_anchors_respect_classes_and_budget():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 300, 8, generator=g)
    y = torch.randint(0, 4, (3, 300), generator=g)
    a, cls = fit_anchors(x, y, 32)
    assert a.shape[1] <= 32
    for b in range(3):
        live = cls[b][cls[b] >= 0]
        assert set(live.tolist()) == set(y[b].tolist())  # every present class has >= 1 cell
    assert allocate([5, 0, 300], 32) == [1, 0, 31]
    assert allocate([5, 0, 3], 32) == [5, 0, 3]  # N <= M: one cell per row
    with pytest.raises(ValueError):
        allocate([4] * 10, 5)  # a budget below the class count would drop classes


def test_gradients_reach_every_parameter():
    cell = _model(cell_m=12)
    cell.train()
    X, y = _table(n_train=60)
    out = cell(X, y)
    out.float().logsumexp(-1).mean().backward()
    missing = [n for n, p in cell.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.abs().sum() > 0)]
    assert not missing, missing


def test_many_classes_run():
    cell = _model(cell_m=12)
    X, y = _table(n_train=80, classes=14)
    out = cell(X, y)
    assert out.shape[-1] == 14 and torch.isfinite(out).all()
