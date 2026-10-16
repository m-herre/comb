# Ported from tabicl 2.2.0 (BSD-3-Clause, Copyright (c) 2025, Soda team @ Inria); see LICENSE-tabicl.
"""Optimizer and LR schedule, ported from upstream tabicl 2.2.0 (U/train/_muon.py,
U/train/_optim.py) to be bit-equal without importing tabicl.

Muon here only implements the ``use_muon=True`` branch: upstream's AdamW branch is dead
for this project because every parameter is placed in a single ``use_muon=True`` group
(see U/train/_run.py:340-366), so it is intentionally dropped.
"""

import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR


# after https://github.com/KellerJordan/Muon/tree/master
def zeropower_via_newtonschulz5_batched(X, steps):
    """``zeropower_via_newtonschulz5`` for a stack (n, rows, cols) of wide (rows <= cols) matrices: the same
    iteration per matrix, with ``b * A + c * A @ A`` and ``a * X + B @ X`` as one batched GEMM each, in IEEE
    fp32. Upstream's per-matrix GEMMs run in TF32 when the trainer enables it (2e-3 from fp64 after five
    steps); IEEE fp32 is ~1e-6 and still cheap (a few ms per optimizer step), and batched TF32 GEMMs would
    round differently from the per-matrix ones and be slightly less accurate."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        X = X / (torch.linalg.vector_norm(X, dim=(1, 2), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = torch.bmm(X, X.transpose(1, 2))
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, B, X, beta=a)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return X


# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape  # assumes 2D shape given (!)
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    This is a single-group, ``use_muon=True``-only port: unlike upstream's ``Muon``, there is no
    AdamW branch and no ``param_groups``/``use_muon`` routing -- all parameters passed in are
    optimized with Muon, reshaped to 2D before the Newton-Schulz step.

    Arguments:
        params: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`.
        weight_decay: Decoupled weight decay coefficient. (0.01 is the v2 recipe default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        momentum: The momentum used by the internal SGD. (0.9 is the v2 recipe default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
    """

    def __init__(self,
                 params,
                 lr,
                 weight_decay=0.01,
                 momentum=0.9,
                 nesterov=True,
                 ns_steps=5,
                 matched_adamw_rms=0.2):

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Upstream's per-parameter update, batched: parameters whose Newton-Schulz input has the same
        (wide) 2D shape run the iteration together (``zeropower_via_newtonschulz5_batched``), and the
        momentum, weight decay and update are ``torch._foreach`` ops, elementwise the same as upstream's."""
        for group in self.param_groups:
            self._muon_group(group)

    def _muon_group(self, group):
        lr = group["lr"]
        ns_steps = group["ns_steps"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        matched_adamw_rms = group["matched_adamw_rms"]
        params = [p for p in group["params"] if p.grad is not None]  # unused this step: skipped
        if not params:
            return
        grads = [p.grad for p in params]
        bufs = []
        for p, g in zip(params, grads):
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            bufs.append(state["momentum_buffer"])
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        ns_inputs = torch._foreach_add(grads, bufs, alpha=momentum) if group["nesterov"] else bufs
        ns_inputs = [x.reshape(len(g), -1) for x, g in zip(ns_inputs, grads)]

        by_shape = {}
        for i, x in enumerate(ns_inputs):
            by_shape.setdefault(tuple(x.shape), []).append(i)
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        for shape, idx in by_shape.items():
            wide = shape[0] <= shape[1]
            X = torch.stack([ns_inputs[i] if wide else ns_inputs[i].T for i in idx])
            X = zeropower_via_newtonschulz5_batched(X, ns_steps)
            if not wide:
                X = X.transpose(1, 2)
            updates = [X[j].view(grads[i].shape) for j, i in enumerate(idx)]
            adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, shape)
            torch._foreach_add_([params[i] for i in idx], updates, alpha=-adjusted_lr)


def _get_cosine_with_restarts_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    amplitude_decay: float,
    lr_end: float = 0.0,
    lr_init: float = 1.0,
):
    """
    Compute the learning rate factor for a cosine schedule with warmup, hard restarts, and amplitude scaling.
    """
    if current_step < num_warmup_steps:
        # Warmup phase: Linearly increase learning rate
        return float(current_step) / float(max(1, num_warmup_steps))

    # After warmup: Apply cosine schedule with hard restarts and amplitude scaling
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init

    # Determine which cycle the current step is in
    cycle_progress = (float(num_cycles) * progress) % 1.0
    current_cycle = int(float(num_cycles) * progress)
    amplitude = amplitude_decay**current_cycle  # Exponentially decay amplitude per cycle

    # Calculate the current learning rate with proper scaling
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    current_lr = lr_end + (lr_init - lr_end) * cosine_factor * amplitude
    return current_lr / lr_init  # as LambdaLR multiplies by lr_init


def cosine_with_warmup(optimizer, warmup_steps: float, total_steps: int, lr_end: float) -> LambdaLR:
    """Cosine schedule with linear warmup, fixed to a single cycle (num_cycles=1) and no
    amplitude decay (amplitude_decay=1), matching upstream's ``get_cosine_with_restarts``
    called with those two arguments pinned.
    """
    lr_init = optimizer.defaults["lr"]
    if lr_end > lr_init:
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_cosine_with_restarts_lr_lambda,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=1,
        amplitude_decay=1,
        lr_end=lr_end,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda)
