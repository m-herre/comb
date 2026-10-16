"""Cell-ICL, Comb's in-context stage: TabICLv2's ICL stage over an exactly editable weighted coreset of the train rows.

TabICLv2's ICL stage runs 12 blocks; every train row attends to the train rows and every test row attends
to the train rows. Cell-ICL keeps that stage -- the same 12 blocks, label encoder, norm and decoder, under
the same parameter names -- but replaces the N train rows by at most M pseudo-rows, one per **cell** of a
class-conditional partition of the label-free row representations R:

    anchors   per table and class c, k_c ~ M n_c / N k-means centroids of that class's rows (detached)
    cell p    the rows of class c whose nearest class-c anchor is p
    stats     count n_p and sum s_p of the member rows R                <- the fitted state: exact sums
    pseudo    x_p = s_p / n_p + y_emb(c_p); every attention reading the cells adds log n_p to their logits

With the log-count bias a softmax over the cells equals the softmax over the rows when each cell's
members share one representation (the weighted-coreset approximation of attention); SSMax uses n = N, the
number of rows. A table with N <= M gets one cell per row with count 1: the stage **is** TabICLv2's.

Given frozen anchors, adding rows adds to (n_p, s_p), deleting subtracts and shards merge by adding; the
12 blocks then re-run over <= M pseudo-rows (O(L M^2), independent of N). Build O(N M d + L M^2), test
row O(L M). Distances and sums run in fixed row blocks with deterministic reductions, so a row's cell does
not depend on how many rows are processed with it (streamed == one-shot; nano ledger, BA guard G1).
"""
from __future__ import annotations

import contextlib
import threading
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

KMEANS_ITERS = 8  # Lloyd steps in training
INFER_KMEANS_ITERS = 4  # CombClassifier's default (kmeans_iters): same Elo in official TabArena runs, less fit time
KMEANS_ROWS_PER_CELL = 8  # Lloyd steps run on at most this many rows per anchor (class-stratified sample)
ASSIGN_BLOCK = 1024  # rows per distance block: fixed, so a row's distances do not depend on its neighbours
_PROJ_SEED = 1234


def allocate(counts: List[int], m: int) -> List[int]:
    """Cells per class: >= 1 for present classes, <= n_c, sum <= m, proportional to the class counts with the
    budget a class cannot use (k_c capped at n_c) handed to the others (water-filling)."""
    n = sum(counts)
    if n <= m:
        return list(counts)
    if m < sum(c > 0 for c in counts):
        raise ValueError(f"cell budget m={m} is below the number of classes present ({sum(c > 0 for c in counts)})")
    k = [0] * len(counts)
    free, active = m, [j for j, c in enumerate(counts) if c > 0]
    while active:
        tot = sum(counts[j] for j in active)
        share = {j: free * counts[j] / tot for j in active}
        capped = [j for j in active if share[j] >= counts[j]]
        if not capped:
            for j in active:
                k[j] = max(1, int(round(share[j])))
            break
        for j in capped:
            k[j] = counts[j]
            free -= counts[j]
        active = [j for j in active if j not in capped]
    while sum(k) > m:
        i = max(range(len(k)), key=lambda j: (k[j], counts[j]))
        k[i] -= 1
    return k


def _projection(dim: int, device) -> Tensor:
    g = torch.Generator(device="cpu").manual_seed(_PROJ_SEED)
    return torch.randn(dim, generator=g).to(device)


def nearest(x: Tensor, c: Tensor, y: Tensor, cls: Tensor) -> Tensor:
    """Per table, the index of the nearest anchor of each row's own class: x (B, N, d), c (B, P, d), row labels
    y (B, N), anchor classes cls (B, P) (-1: padding) -> (B, N) long, -1 where the row's class has no anchor.
    Distances run in full fp32 (TF32 off: training and inference must agree on near-ties) in fixed blocks of
    ASSIGN_BLOCK rows (the last padded), so a row's result does not depend on the other rows; the class mask is
    built per block (no (B, N, P) tensor)."""
    B, n, _ = x.shape
    if n == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=x.device)
    out = []
    with _no_tf32(x.is_cuda):
        c2 = (c * c).sum(-1)[:, None, :]  # argmin_j |x - c_j|^2 = argmin_j (|c_j|^2 - 2 x.c_j)
        for s in range(0, n, ASSIGN_BLOCK):
            xb, yb = x[:, s:s + ASSIGN_BLOCK], y[:, s:s + ASSIGN_BLOCK]
            pad = ASSIGN_BLOCK - xb.shape[1]
            if pad:  # only the last block is padded (no copy of x)
                xb = torch.cat([xb, xb[:, -1:].expand(B, pad, xb.shape[-1])], 1)
                yb = torch.cat([yb, yb[:, -1:].expand(B, pad)], 1)
            d = torch.baddbmm(c2, xb, c.transpose(1, 2), alpha=-2.0)
            d = d.masked_fill(yb[..., None] != cls[:, None, :], float("inf"))
            a = d.argmin(-1)
            out.append(torch.where(torch.isfinite(d.gather(-1, a[..., None]))[..., 0], a, torch.full_like(a, -1)))
    return torch.cat(out, 1)[:, :n]


_TF32_LOCK = threading.RLock()


@contextlib.contextmanager
def _no_tf32(cuda: bool):
    """TF32 off for the enclosed matmuls. The flag is process-wide, so threads take turns: saving and restoring it
    under a lock keeps one thread from restoring another's temporary value."""
    if not cuda:
        yield
        return
    with _TF32_LOCK:
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32


def segment_sums(x: Tensor, assign: Tensor, p: int, dtype: Optional[torch.dtype] = None) -> Tuple[Tensor, Tensor]:
    """Per-cell (counts (B, P), sums (B, P, d)) of the rows x (B, N, d) by cell index (B, N) in [-1, P), summed in
    ``dtype`` (default x's). Sorted segment sums: deterministic (no atomics) and differentiable in x; rows with -1
    are dropped."""
    B, n, d = x.shape
    key = torch.where(assign < 0, torch.full_like(assign, p), assign)
    counts = torch.zeros(B, p + 1, dtype=torch.long, device=x.device).scatter_add_(1, key, torch.ones_like(key))
    order = key.argsort(dim=1, stable=True)
    xs = torch.gather(x, 1, order[..., None].expand(B, n, d)).to(dtype or x.dtype)  # one copy at the sum precision
    sums = torch.segment_reduce(xs, "sum", lengths=counts, axis=1)
    return counts[:, :p], sums[:, :p]


@torch.no_grad()
def fit_anchors(t: Tensor, y: Tensor, m: int, iters: Optional[int] = None,
                n_classes: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Class-conditional k-means anchors of every table: t (B, N, d) label-free rows, y (B, N) labels ->
    anchors (B, P, d) fp32 and their class (B, P) long, -1 for padding (tables can get different cell counts).
    Init: per class, the rows at k_c evenly spaced ranks of the projection onto a fixed direction; Lloyd steps
    assign each row to the nearest anchor of its own class. The Lloyd steps see at most KMEANS_ROWS_PER_CELL
    rows per anchor of each class (evenly spaced in that projection order), so their cost does not grow with N
    beyond that. Deterministic and row-order invariant (up to projection ties)."""
    iters = KMEANS_ITERS if iters is None else iters  # read at call time (the sweep varies it)
    t = t.float()
    B, n, d = t.shape
    y = y.long()
    n_cls = n_classes if n_classes is not None else (int(y.max().item()) + 1 if n else 1)  # given: no host sync
    counts = torch.zeros(B, n_cls, dtype=torch.long, device=t.device).scatter_add_(1, y, torch.ones_like(y)).cpu()
    ks = [allocate(row.tolist(), m) for row in counts]
    if all(sum(k) == int(row.sum()) for k, row in zip(ks, counts)):
        iters = 0  # every row is an anchor (N <= M, no compression): Lloyd steps have nothing to move
    P = max(sum(k) for k in ks)
    # rows grouped by class, ascending projection within a class
    proj = t @ _projection(d, t.device)
    order = proj.argsort(dim=1, stable=True)
    order = torch.gather(order, 1, torch.gather(y, 1, order).argsort(dim=1, stable=True))
    idx = torch.zeros(B, P, dtype=torch.long)
    cls = torch.full((B, P), -1, dtype=torch.long)
    samples, sample_cls = [], []
    starts = torch.cumsum(counts, 1) - counts
    for b, k in enumerate(ks):
        pos, rows, rcls = 0, [], []
        for c, kc in enumerate(k):
            if kc == 0:
                continue
            nc, s0 = int(counts[b, c]), int(starts[b, c])
            idx[b, pos:pos + kc] = s0 + torch.linspace(0, nc - 1, kc).long()
            cls[b, pos:pos + kc] = c
            pos += kc
            ns = min(nc, KMEANS_ROWS_PER_CELL * kc)
            rows.append(s0 + torch.linspace(0, nc - 1, ns).long())
            rcls.append(torch.full((ns,), c, dtype=torch.long))
        samples.append(torch.cat(rows))
        sample_cls.append(torch.cat(rcls))
    S = max(len(r) for r in samples)
    sidx = torch.zeros(B, S, dtype=torch.long)
    scls = torch.full((B, S), -1, dtype=torch.long)  # padding rows match no anchor (dropped from the sums)
    for b, (r, rc) in enumerate(zip(samples, sample_cls)):
        sidx[b, :len(r)], scls[b, :len(r)] = r, rc
    idx, cls, sidx, scls = (v.to(t.device) for v in (idx, cls, sidx, scls))
    anchors = torch.gather(t, 1, torch.gather(order, 1, idx)[..., None].expand(B, P, d))
    ts = torch.gather(t, 1, torch.gather(order, 1, sidx)[..., None].expand(B, S, d))
    for _ in range(iters):
        a = nearest(ts, anchors, scls, cls)  # sample padding (-1) matches no anchor
        cnt, sums = segment_sums(ts, a, P)
        anchors = torch.where((cnt > 0)[..., None], sums / cnt.clamp_min(1)[..., None].float(), anchors)
    return anchors, cls


class CellICL(nn.Module):
    """TabICLv2's in-context stage (``tf_icl``, ``ln``, ``y_encoder``, ``decoder``; upstream parameter names) and its
    ``forward(R, y_train)``. ``num_cells`` is the cell budget M (the trainer resamples it per step)."""

    def __init__(self, out_dim, max_classes, d_model, num_blocks, nhead, dim_feedforward, dropout, activation,
                 bias_free_ln, zero_init, num_cells: int = 1024):  # fmt: skip
        super().__init__()
        from .model import Encoder, OneHotAndLinear

        if max_classes <= 0:
            raise ValueError("Cell-ICL partitions the rows by class: classification only")
        self.max_classes = max_classes
        self.num_cells = int(num_cells)
        self.tf_icl = Encoder(
            num_blocks, d_model, nhead, dim_feedforward, dropout=dropout, activation=activation,
            bias_free_ln=bias_free_ln, ssmax=True, zero_init=zero_init,
        )  # fmt: skip
        self.ln = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.y_encoder = OneHotAndLinear(max_classes, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, out_dim))
        self.diag: Optional[Dict[str, float]] = None  # set to {} to collect cell stats on the next forward

    # -- cells ------------------------------------------------------------------------------------------

    def label(self, y: Tensor) -> Tensor:
        return self.y_encoder(y.float())

    def cells(self, R_train: Tensor, y_train: Tensor, anchors: Optional[Tuple[Tensor, Tensor]] = None):
        """Pseudo-rows (B, P, d) with their label embedding, key bias (B, P) and the row count N, from the
        label-free train representations R_train (B, N, d). ``anchors`` (frozen editing contract) or fitted."""
        # fp32 throughout (sums over thousands of rows; fp16 autocast at inference would lose them)
        with torch.autocast(device_type=R_train.device.type, enabled=False):
            fitted = anchors is None
            if fitted:
                anchors = fit_anchors(R_train.detach(), y_train, self.num_cells, n_classes=self.max_classes)
            a, cls = anchors
            assign = nearest(R_train.detach().float(), a.float(), y_train.long(), cls)
            count, total = segment_sums(R_train.float(), assign, a.shape[1])
            # fitted anchors cover every class present, so every row is in a cell: n = N without a host sync
            n = R_train.shape[1] if fitted else None
            return self.cells_from_stats(count, total, cls, n), count

    def cells_from_stats(self, count: Tensor, total: Tensor, cls: Tensor, n: Optional[int] = None):
        """(pseudo-rows (B, P, d), key bias (B, P), n rows) from exact per-cell stats (editing path too); n is
        the SSMax row count (default: the largest table's count total, a host sync)."""
        live = count > 0
        mean = total / count.clamp_min(1)[..., None].to(total.dtype)
        x = mean + self.label(cls.clamp_min(0)).to(total.dtype)
        x = torch.where(live[..., None], x, torch.zeros_like(x))
        bias = torch.where(live, count.clamp_min(1).to(torch.float32).log(),
                           torch.full_like(count, float("-inf"), dtype=torch.float32))
        if n is None:
            n = int(count.sum(1).max().item())
        return x, bias, n

    # -- forward -----------------------------------------------------------------------------------------

    def forward(self, R: Tensor, y_train: Tensor) -> Tensor:
        N = y_train.shape[1]
        if N <= self.num_cells:  # one cell per row, count 1: exactly TabICLv2's ICL stage
            R[:, :N] = R[:, :N] + self.label(y_train)  # in place
            out = self.tf_icl(R, train_size=N, last_query_start=N)
            if self.diag is not None:
                self.diag.update(cells=float(N), exact=1.0)
            return self.decoder(self.ln(out))
        (X, bias, n), count = self.cells(R[:, :N], y_train)
        if self.diag is not None:
            self._record(count, N)
        P = X.shape[1]
        src = torch.cat([X.to(R.dtype), R[:, N:]], 1)
        out = self.tf_icl(src, train_size=P, last_query_start=P, key_bias=bias, ssmax_n=n)
        return self.decoder(self.ln(out))

    def _record(self, count: Tensor, N: int) -> None:
        c = count.detach().float()
        live = (c > 0).sum(1).float()
        p = c / c.sum(1, keepdim=True).clamp_min(1)
        ent = (-(p * p.clamp_min(1e-12).log()).sum(1)).exp()
        self.diag.update(cells=float(live.mean()), exact=0.0, rows_per_cell=float(N / live.mean().clamp_min(1)),
                         max_count=float(c.max()), cell_entropy_eff=float(ent.mean()))

    # -- inference cache: build once, decode any number of test rows --------------------------------------

    def summary_from_cells(self, X: Tensor, bias: Optional[Tensor], n: int) -> Dict:
        states = []
        P = X.shape[1]
        for block in self.tf_icl.blocks[:-1]:
            states.append(X)
            X = block(q=X, train_size=P, key_bias=bias, ssmax_n=n)
        states.append(X)
        return {"states": states, "bias": bias, "n": n}

    def decode(self, Q: Tensor, summary: Dict) -> Tensor:
        bias, n = summary["bias"], summary["n"]
        for block, S in zip(self.tf_icl.blocks, summary["states"]):
            S = S.to(Q.dtype)
            Q = block(q=Q, k=S, v=S, key_bias=bias, ssmax_n=n)
        return self.decoder(self.ln(Q))
