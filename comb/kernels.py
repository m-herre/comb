"""Fused CUDA kernels (Triton) for inference, each equal within fp16 rounding to the PyTorch ops it
replaces. Every function falls back to those PyTorch ops when Triton or CUDA is unavailable.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    # Sizes are not specialised on, so a new table shape never triggers a recompile.
    @triton.jit(do_not_specialize=["n_rows", "rows_per_batch"])
    def _layer_norm_fp16_kernel(X, W, B, Y, n_rows, rows_per_batch, stride_xb, stride_xr, eps,
                                D: tl.constexpr, BLOCK_ROWS: tl.constexpr, HAS_BIAS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # > 2**31 elements
        cols = tl.arange(0, D)
        mask = rows < n_rows
        x_off = (rows // rows_per_batch) * stride_xb + (rows % rows_per_batch) * stride_xr
        x = tl.load(X + x_off[:, None] + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt_rn(var + eps)
        y = xc * rstd[:, None] * tl.load(W + cols)[None, :]
        if HAS_BIAS:
            y = y + tl.load(B + cols)[None, :]
        tl.store(Y + rows[:, None] * D + cols[None, :], y.to(tl.float16), mask=mask[:, None])


def _as_batched_rows(x: Tensor):
    """(x3, rows_per_batch, stride_batch, stride_row) viewing x as (batch, rows, D) with unit last stride."""
    if x.stride(-1) != 1 or x.dim() < 2:
        x = x.contiguous()
    lead = x.shape[:-2]
    try:
        x3 = x.view(-1, x.shape[-2], x.shape[-1]) if lead else x.unsqueeze(0)
    except RuntimeError:  # leading dims not mergeable without a copy
        x = x.contiguous()
        x3 = x.view(-1, x.shape[-2], x.shape[-1])
    return x3, x3.shape[1], x3.stride(0), x3.stride(1)


def layer_norm_fp16(x: Tensor, norm: torch.nn.LayerNorm) -> Tensor:
    """fp16 ``norm(x)`` computed in fp32: what CUDA autocast's fp32 layer norm followed by the fp16
    cast of the next linear produces, in one pass and without the fp32 intermediate."""
    D = x.shape[-1]
    if triton is None or not x.is_cuda or D & (D - 1) or norm.weight is None:
        with torch.autocast("cuda", enabled=False):
            return F.layer_norm(x.float(), norm.normalized_shape, norm.weight, norm.bias, norm.eps).half()
    x3, rows_per_batch, sb, sr = _as_batched_rows(x)
    n_rows = x3.shape[0] * rows_per_batch
    y = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    block_rows = max(1, 4096 // D)
    grid = (triton.cdiv(n_rows, block_rows),)
    bias = norm.bias if norm.bias is not None else norm.weight
    _layer_norm_fp16_kernel[grid](x3, norm.weight, bias, y, n_rows, rows_per_batch, sb, sr, norm.eps,
                                  D=D, BLOCK_ROWS=block_rows, HAS_BIAS=norm.bias is not None)
    return y


if triton is not None:

    @triton.jit(do_not_specialize=["n_rows", "L"])
    def _rope_fp16_kernel(T, COS, SIN, Y, n_rows, L, NH, s_n, s_l, s_h,
                          HALF: tl.constexpr, BLOCK: tl.constexpr, OUT_FP16: tl.constexpr = True,
                          NEG_SIN: tl.constexpr = False):
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # row = (n, l, head)
        mask = rows < n_rows
        h = rows % NH
        pos = (rows // NH) % L
        n = rows // (NH * L)
        c = tl.arange(0, HALF)
        src = (n * s_n + pos * s_l + h * s_h)[:, None] + c[None, :]
        x1 = tl.load(T + src, mask=mask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(T + src + HALF, mask=mask[:, None], other=0.0).to(tl.float32)
        cs = pos[:, None] * HALF + c[None, :]
        cos = tl.load(COS + cs, mask=mask[:, None], other=0.0)
        sin = tl.load(SIN + cs, mask=mask[:, None], other=0.0)
        if NEG_SIN:  # the backward: rotation by -angle
            sin = -sin
        dst = rows[:, None] * (2 * HALF) + c[None, :]
        y1, y2 = x1 * cos + (-x2) * sin, x2 * cos + x1 * sin
        if OUT_FP16:
            y1, y2 = y1.to(tl.float16), y2.to(tl.float16)
        tl.store(Y + dst, y1, mask=mask[:, None])
        tl.store(Y + dst + HALF, y2, mask=mask[:, None])


def rope_fp16(t: Tensor, freqs: Tensor) -> Tensor:
    """Non-interleaved RoPE on fp16 ``t`` of shape (..., L, heads, head_dim), rotated along L by angles
    ``freqs`` (L, head_dim / 2), computed in fp32 like ``model.apply_rotary_emb``; returns a contiguous
    fp16 tensor of ``t``'s shape. Needs unit stride on the last dim and mergeable leading dims."""
    *lead, L, nh, hd = t.shape
    half = hd // 2
    t4 = t.view(-1, L, nh, hd)
    y = torch.empty(t.shape, device=t.device, dtype=torch.float16)
    n_rows = t4.shape[0] * L * nh
    block = 256
    _rope_fp16_kernel[(triton.cdiv(n_rows, block),)](
        t4, freqs.cos().contiguous(), freqs.sin().contiguous(), y, n_rows, L, nh,
        t4.stride(0), t4.stride(1), t4.stride(2), HALF=half, BLOCK=block)
    return y


def _rope_rows(t: Tensor, cos: Tensor, sin: Tensor, out_dtype, neg_sin: bool) -> Tensor:
    *lead, L, nh, hd = t.shape
    t4 = t.view(-1, L, nh, hd)
    y = torch.empty(t.shape, device=t.device, dtype=out_dtype)
    n_rows = t4.shape[0] * L * nh
    _rope_fp16_kernel[(triton.cdiv(n_rows, 256),)](
        t4, cos, sin, y, n_rows, L, nh, t4.stride(0), t4.stride(1), t4.stride(2), HALF=hd // 2, BLOCK=256,
        OUT_FP16=out_dtype == torch.float16, NEG_SIN=neg_sin)
    return y


class _RopeTrain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rope_rows(t, cos, sin, torch.float32, False)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        dy = dy if dy.stride(-1) == 1 and _mergeable(dy) else dy.contiguous()
        return _rope_rows(dy, cos, sin, torch.float32, True), None, None


def _mergeable(t: Tensor) -> bool:
    try:
        t.view(-1, *t.shape[-3:])
    except RuntimeError:
        return False
    return True


def rope_train(t: Tensor, freqs: Tensor) -> Tensor:
    """Differentiable fp32 RoPE on ``t`` (..., L, heads, head_dim), as ``model.apply_rotary_emb`` on its
    (..., heads, L, head_dim) transpose; returns a contiguous tensor of ``t``'s shape. The backward is the
    rotation by the negated angles."""
    return _RopeTrain.apply(t, freqs.cos().contiguous(), freqs.sin().contiguous())


def can_rope_train(t: Tensor) -> bool:
    hd = t.shape[-1]
    return (triton is not None and t.is_cuda and t.dtype == torch.float32 and t.stride(-1) == 1
            and not hd & (hd - 1) and _mergeable(t))


def can_rope_fp16(t: Tensor) -> bool:
    hd = t.shape[-1]
    if triton is None or not t.is_cuda or t.dtype != torch.float16 or t.stride(-1) != 1 or hd & (hd - 1):
        return False
    try:
        t.view(-1, *t.shape[-3:])
    except RuntimeError:
        return False
    return True


if triton is not None:

    @triton.jit
    def _attention_step(q, k_base, v_base, offs_n, offs_d, skn, svn, Lk, m_i, l_i, acc, qk_scale,
                        MASK: tl.constexpr, FOLD_SCALE: tl.constexpr, IEEE: tl.constexpr):
        # one key block of the online softmax (fp32, base 2); MASK only for the ragged last block
        if MASK:
            in_k = offs_n < Lk
            k = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None], mask=in_k[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :], mask=in_k[:, None], other=0.0)
        else:
            k = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None])
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :])
        if IEEE:  # fp32 inputs: exact fp32 FMA dots, no TF32
            s = tl.dot(q, k, input_precision="ieee")
        else:
            s = tl.dot(q, k)
        if FOLD_SCALE:  # max of the raw scores, the scale folded into one FMA before exp2
            if MASK:
                s = tl.where(in_k[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
            p = tl.math.exp2(s * qk_scale - m_new[:, None])
        else:
            s = s * qk_scale
            if MASK:
                s = tl.where(in_k[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if IEEE:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        return m_new, l_i, acc

    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attention_kernel(Q, K, V, Out, qk_scale, Lq, Lk,
                          sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som,
                          H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          FOLD_SCALE: tl.constexpr, IEEE: tl.constexpr):
        # FlashAttention-style forward: one program per (batch * head, query block), online softmax in
        # fp32 (base 2); fp16 tensor-core dots, or exact fp32 dots for fp32 inputs; no padding of the
        # head dim. The full key blocks run unmasked, the ragged last one after the loop.
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :],
                    mask=offs_m[:, None] < Lq, other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        k_base, v_base = K + b * skb + h * skh, V + b * svb + h * svh
        n_full = (Lk // BLOCK_N) * BLOCK_N
        for start in range(0, n_full, BLOCK_N):
            m_i, l_i, acc = _attention_step(q, k_base, v_base, start + tl.arange(0, BLOCK_N), offs_d, skn, svn, Lk,
                                            m_i, l_i, acc, qk_scale, False, FOLD_SCALE, IEEE)
        if n_full < Lk:
            m_i, l_i, acc = _attention_step(q, k_base, v_base, n_full + tl.arange(0, BLOCK_N), offs_d, skn, svn, Lk,
                                            m_i, l_i, acc, qk_scale, True, FOLD_SCALE, IEEE)
        acc = acc / l_i[:, None]
        tl.store(Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :], acc.to(Out.dtype.element_ty),
                 mask=offs_m[:, None] < Lq)


ATTENTION_HEAD_DIMS = (16,)  # where this kernel beats FlashAttention-2 on the RTX PRO 6000 (hd 64: it does not)


def can_attention(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return (triton is not None and q.is_cuda and q.dtype == k.dtype == v.dtype
            and q.dtype in (torch.float16, torch.float32) and q.dim() == 4 and q.shape[-1] in ATTENTION_HEAD_DIMS
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1 and k.shape[-2] > 0)


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``F.scaled_dot_product_attention(q, k, v)`` (no mask, default scale) for fp16 or fp32 (N, heads, L,
    hd) inputs with a small head dim (fp32: exact fp32 dots, as accurate as SDPA's fp32 kernels);
    returns (N, heads, Lq, hd) in q's dtype, laid out as (N, Lq, heads, hd) in memory."""
    N, H, Lq, D = q.shape
    Lk = k.shape[-2]
    o = torch.empty(N, Lq, H, D, device=q.device, dtype=q.dtype).transpose(1, 2)
    # Fixed tiles by shape class (measured on the RTX PRO 6000; no autotuning, so the few compiled
    # variants do not depend on the data): the CLS-only queries, short rows, long rows, the rest.
    ieee = q.dtype == torch.float32
    if ieee:
        block_m, block_n, warps, stages, fold = (16, 64, 4, 2, True) if Lq <= 16 else (64, 64, 4, 2, True)
    elif Lq <= 16:
        block_m, block_n, warps, stages, fold = 16, 256, 4, 2, False
    elif Lq <= 64:
        block_m, block_n, warps, stages, fold = 64, 32 if Lk <= 32 else 64, 4, 3, True
    elif Lq >= 1024 and Lk >= 1024:
        block_m, block_n, warps, stages, fold = 256, 64, 8, 3, True
    else:
        block_m, block_n, warps, stages, fold = 128, 64, 4, 3, True
    grid = (N * H, triton.cdiv(Lq, block_m))
    _attention_kernel[grid](q, k, v, o, 1.4426950408889634 / math.sqrt(D), Lq, Lk,
                            *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3],
                            H=H, D=D, BLOCK_M=block_m, BLOCK_N=block_n, FOLD_SCALE=fold, IEEE=ieee,
                            num_warps=warps, num_stages=stages)
    return o


if triton is not None:

    @triton.jit(do_not_specialize=["n_rows", "rows_per_batch"])
    def _ffn_residual_kernel(X, LNW, LNB, W1, B1, W2, B2, Y, n_rows, rows_per_batch, stride_xb, stride_xr, eps,
                             D: tl.constexpr, F: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr,
                             HAS_LN_BIAS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = rows < n_rows
        cols = tl.arange(0, D)
        x_off = (rows // rows_per_batch) * stride_xb + (rows % rows_per_batch) * stride_xr
        x_in = tl.load(X + x_off[:, None] + cols[None, :], mask=mask[:, None], other=0.0)
        x = x_in.to(tl.float32)
        # norm2, in fp32, cast to fp16 as autocast's linear1 would
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt_rn(tl.sum(xc * xc, axis=1) / D + eps)
        xn = xc * rstd[:, None] * tl.load(LNW + cols)[None, :]
        if HAS_LN_BIAS:
            xn = xn + tl.load(LNB + cols)[None, :]
        xn = xn.to(tl.float16)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        fr = tl.arange(0, BLOCK_F)
        for f0 in range(0, F, BLOCK_F):
            # linear1 chunk (W1 is (F, D)), rounded to fp16 like its fp16 output; GELU (erf) on that,
            # rounded to fp16; then its share of linear2 (W2 is (D, F)), accumulated in fp32
            w1 = tl.load(W1 + (f0 + fr)[None, :] * D + cols[:, None])
            h = (tl.dot(xn, w1) + tl.load(B1 + f0 + fr)[None, :].to(tl.float32)).to(tl.float16).to(tl.float32)
            g = (0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))).to(tl.float16)
            w2 = tl.load(W2 + cols[None, :] * F + (f0 + fr)[:, None])
            acc = tl.dot(g, w2, acc)
        ff = (acc + tl.load(B2 + cols)[None, :].to(tl.float32)).to(tl.float16)  # linear2's fp16 output
        y = x_in + ff.to(x_in.dtype)  # the residual in x's dtype, as `x + ff` does
        tl.store(Y + rows[:, None] * D + cols[None, :], y, mask=mask[:, None])


def _fp16_params(module: torch.nn.Module) -> tuple:
    """(weight, bias) of a linear layer in fp16, cast once and reused while the parameters are unchanged."""
    key = (module.weight._version, module.bias._version, module.weight.data_ptr())
    cached = getattr(module, "_fp16_params", None)
    if cached is None or cached[0] != key:
        cached = (key, module.weight.detach().half().contiguous(), module.bias.detach().half().contiguous())
        module._fp16_params = cached
    return cached[1], cached[2]


FFN_DIMS = {(128, 256)}  # (d_model, dim_feedforward) of the column and row stages


def can_ffn_residual(x: Tensor, block) -> bool:
    return (triton is not None and x.is_cuda and x.dtype in (torch.float16, torch.float32)
            and (block.linear1.in_features, block.linear1.out_features) in FFN_DIMS
            and block.linear1.bias is not None and block.linear2.bias is not None and block.norm2.weight is not None
            and block.activation is F.gelu and x.shape[-1] == block.linear1.in_features)


def ffn_residual(x: Tensor, block) -> Tensor:
    """``x + block._ff_block(block.norm2(x))`` under fp16 autocast (``TransformerEncoderLayer`` with erf
    GELU, eval mode) in one kernel: the 4x wider hidden layer never leaves the chip. Rounds where
    autocast does (fp16 after norm2, linear1, GELU and linear2); returns x's dtype, contiguous."""
    D, Fdim = block.linear1.in_features, block.linear1.out_features
    x3, rows_per_batch, sb, sr = _as_batched_rows(x)
    n_rows = x3.shape[0] * rows_per_batch
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    w1, b1 = _fp16_params(block.linear1)
    w2, b2 = _fp16_params(block.linear2)
    ln = block.norm2
    block_m, block_f, warps = (128, 64, 8) if x.dtype == torch.float16 else (64, 32, 4)  # tuned on sm_120
    _ffn_residual_kernel[(triton.cdiv(n_rows, block_m),)](
        x3, ln.weight, ln.bias if ln.bias is not None else ln.weight, w1, b1, w2, b2, y,
        n_rows, rows_per_batch, sb, sr, ln.eps, D=D, F=Fdim, BLOCK_M=block_m, BLOCK_F=block_f,
        HAS_LN_BIAS=ln.bias is not None, num_warps=warps, num_stages=2)
    return y


if triton is not None:

    @triton.jit(do_not_specialize=["n_rows", "rows_per_batch", "L"])
    def _ln_linear_kernel(X, LNW, LNB, W, B, COS, SIN, Y, n_rows, rows_per_batch, stride_xb, stride_xr,
                          L, eps, D: tl.constexpr, N_OUT: tl.constexpr, N_ROPE: tl.constexpr, HD: tl.constexpr,
                          BLOCK_M: tl.constexpr, HAS_LN_BIAS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = rows < n_rows
        cols = tl.arange(0, D)
        x_off = (rows // rows_per_batch) * stride_xb + (rows % rows_per_batch) * stride_xr
        x = tl.load(X + x_off[:, None] + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt_rn(tl.sum(xc * xc, axis=1) / D + eps)
        xn = xc * rstd[:, None] * tl.load(LNW + cols)[None, :]
        if HAS_LN_BIAS:
            xn = xn + tl.load(LNB + cols)[None, :]
        xn = xn.to(tl.float16)  # the fp16 input autocast's linear sees
        pos = rows % L  # position along the sequence (rows are (..., L) flattened)
        for n0 in tl.static_range(0, N_OUT, D):
            w = tl.load(W + (n0 + cols)[None, :] * D + cols[:, None])  # (D, D) block of W^T
            y = (tl.dot(xn, w) + tl.load(B + n0 + cols)[None, :].to(tl.float32)).to(tl.float16)
            if n0 < N_ROPE:
                # RoPE: rotate_half(y) per head of the rounded y, in registers (halves swapped, the
                # moved second half negated); then the fp32 maths of apply_rotary_emb
                y4 = tl.permute(tl.reshape(y, (BLOCK_M, D // HD, 2, HD // 2)), (0, 1, 3, 2))
                first, second = tl.split(y4)
                r = tl.reshape(tl.permute(tl.join(-second, first), (0, 1, 3, 2)), (BLOCK_M, D))
                cs = pos[:, None] * HD + (cols % HD)[None, :]
                cos = tl.load(COS + cs, mask=mask[:, None], other=0.0)
                sin = tl.load(SIN + cs, mask=mask[:, None], other=0.0)
                y = (y.to(tl.float32) * cos + r.to(tl.float32) * sin).to(tl.float16)
            tl.store(Y + rows[:, None] * N_OUT + n0 + cols[None, :], y, mask=mask[:, None])

    @triton.jit(do_not_specialize=["n_rows", "rows_a", "rows_r"])
    def _linear_residual_kernel(A, W, B, R, Y, n_rows, rows_a, stride_ab, stride_ar, rows_r, stride_rb, stride_rr,
                                D: tl.constexpr, BLOCK_M: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = rows < n_rows
        cols = tl.arange(0, D)
        a = tl.load(A + ((rows // rows_a) * stride_ab + (rows % rows_a) * stride_ar)[:, None] + cols[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float16)
        w = tl.load(W + cols[None, :] * D + cols[:, None])
        o = (tl.dot(a, w) + tl.load(B + cols)[None, :].to(tl.float32)).to(tl.float16)  # out_proj's fp16 output
        r = tl.load(R + ((rows // rows_r) * stride_rb + (rows % rows_r) * stride_rr)[:, None] + cols[None, :],
                    mask=mask[:, None], other=0.0)
        tl.store(Y + rows[:, None] * D + cols[None, :], r + o.to(r.dtype), mask=mask[:, None])


def attention_block_params(attn) -> dict:
    """fp16 copies of an attention module's projections, cached per module while its parameters are
    unchanged."""
    w, b, wo, bo = attn.in_proj_weight, attn.in_proj_bias, attn.out_proj.weight, attn.out_proj.bias
    key = tuple(t._version for t in (w, b, wo, bo)) + (w.data_ptr(),)
    cached = getattr(attn, "_fused_params", None)
    if cached is None or cached[0] != key:
        params = {"w": w.detach().half().contiguous(), "b": b.detach().half().contiguous(),
                  "wo": wo.detach().half().contiguous(), "bo": bo.detach().half().contiguous()}
        cached = (key, params)
        attn._fused_params = cached
    return cached[1]


def ln_linear_fp16(x: Tensor, norm, weight: Tensor, bias: Tensor, n_rope: int = 0, angles: Tensor | None = None) -> Tensor:
    """fp16 ``F.linear(norm(x), weight, bias)`` under autocast (norm in fp32, fp16 GEMM input and output),
    with RoPE on the first ``n_rope`` output columns (per head, positions along x's dim -2, ``angles``
    (L, head_dim / 2) as ``RotaryEmbedding.angles``) computed in fp32 like ``apply_rotary_emb``; one pass."""
    D, n_out = x.shape[-1], weight.shape[0]
    x3, rows_per_batch, sb, sr = _as_batched_rows(x)
    n_rows = x3.shape[0] * rows_per_batch
    y = torch.empty(*x.shape[:-1], n_out, device=x.device, dtype=torch.float16)
    if n_rope:
        cos, sin = angles.cos(), angles.sin()
        cos, sin = torch.cat([cos, cos], -1).contiguous(), torch.cat([sin, sin], -1).contiguous()
        hd = cos.shape[-1]
    else:
        cos = sin = weight
        hd = D
    block_m = 32  # 64 rows spill registers at 4 warps (3.5x slower); tuned on sm_120
    _ln_linear_kernel[(triton.cdiv(n_rows, block_m),)](
        x3, norm.weight, norm.bias if norm.bias is not None else norm.weight, weight, bias, cos, sin, y, n_rows, rows_per_batch, sb, sr, x.shape[-2] if x.dim() > 1 else 1, norm.eps,
        D=D, N_OUT=n_out, N_ROPE=n_rope, HD=hd, BLOCK_M=block_m, HAS_LN_BIAS=norm.bias is not None,
        num_warps=4, num_stages=2)
    return y


def linear_residual(a: Tensor, weight: Tensor, bias: Tensor, residual: Tensor) -> Tensor:
    """``residual + F.linear(a, weight, bias)`` under autocast (fp16 linear output, sum in the residual's
    dtype) in one pass; returns residual's dtype and shape, contiguous."""
    D = a.shape[-1]
    a3, rows_a, sab, sar = _as_batched_rows(a)
    r3, rows_r, srb, srr = _as_batched_rows(residual)
    n_rows = a3.shape[0] * rows_a
    y = torch.empty(residual.shape, device=a.device, dtype=residual.dtype)
    block_m = 64
    _linear_residual_kernel[(triton.cdiv(n_rows, block_m),)](
        a3, weight, bias, r3, y, n_rows, rows_a, sab, sar, rows_r, srb, srr, D=D, BLOCK_M=block_m,
        num_warps=4, num_stages=2)
    return y


if triton is not None:
    from triton.language.extra import libdevice

    @triton.jit
    def _f64(value: tl.constexpr):
        return tl.full([], value, tl.float64)

    @triton.jit
    def _yj_neg_llf(S, n, lmb, sign_log, BLOCK: tl.constexpr):
        # scipy's mixed-sign Yeo-Johnson negative log-likelihood at lmb for one column, in float64:
        # the transform as _yeojohnson_torch, its population variance by Chan's merge of per-block
        # two-pass statistics (accurate like numpy's two-pass var)
        eps = _f64(2.220446049250313e-16)
        count = _f64(0.0)
        mean = _f64(0.0)
        m2 = _f64(0.0)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < n
            s = tl.load(S + offs, mask=valid, other=0.0)
            a = tl.abs(s)
            pos = s >= 0
            lp = lmb
            ln = _f64(2.0) - lmb
            e = libdevice.expm1(tl.where(pos, lp, ln) * a)
            y_pos = tl.where(tl.abs(lp) < eps, a, e / lp)
            y_neg = tl.where(tl.abs(ln) > eps, -e / ln, -a)
            y = tl.where(valid, tl.where(pos, y_pos, y_neg), 0.0)
            cb = tl.sum(valid.to(tl.float64), 0)
            mb = tl.sum(y, 0) / cb
            d = tl.where(valid, y - mb, 0.0)
            m2b = tl.sum(d * d, 0)
            delta = mb - mean
            tot = count + cb
            mean = mean + delta * cb / tot
            m2 = m2 + m2b + delta * delta * count * cb / tot
            count = tot
        sigma = m2 / count
        logvar = tl.where(sigma >= _f64(2.2250738585072014e-308), tl.log(sigma), -float("inf"))
        llf = -count / _f64(2.0) * logvar + (lmb - _f64(1.0)) * sign_log
        llf = tl.where(tl.abs(llf) == float("inf"), -float("inf"), llf)
        return -llf

    @triton.jit
    def _yj_brent_kernel(S, LB, UB, SIGN_LOG, OUT, n, BLOCK: tl.constexpr):
        # one program per column: scipy.optimize._minimize_scalar_bounded (xatol 1.48e-8, maxfun 500)
        col = tl.program_id(0).to(tl.int64)
        S = S + col * n
        a = tl.load(LB + col)
        b = tl.load(UB + col)
        sign_log = tl.load(SIGN_LOG + col)
        # Python float literals would be fp32 constants in Triton: every constant is spelled in fp64
        zero, half, two = _f64(0.0), _f64(0.5), _f64(2.0)
        xatol_3 = _f64(1.48e-08 / 3.0)
        sqrt_eps = _f64(1.4832396974191326e-08)  # sqrt(2.2e-16)
        golden_mean = _f64(0.3819660112501051)  # 0.5 * (3 - sqrt(5))
        fulc = a + golden_mean * (b - a)
        nfc = fulc
        xf = fulc
        rat = zero
        e = zero
        fx = _yj_neg_llf(S, n, xf, sign_log, BLOCK)
        num = 1
        ffulc = fx
        fnfc = fx
        xm = half * (a + b)
        tol1 = sqrt_eps * tl.abs(xf) + xatol_3
        tol2 = two * tol1
        while (tl.abs(xf - xm) > (tol2 - half * (b - a))) & (num < 500):
            golden = True
            if tl.abs(e) > tol1:
                golden = False
                r = (xf - nfc) * (fx - ffulc)
                q = (xf - fulc) * (fx - fnfc)
                p = (xf - fulc) * q - (xf - nfc) * r
                q = two * (q - r)
                if q > zero:
                    p = -p
                q = tl.abs(q)
                r = e
                e = rat
                if (tl.abs(p) < tl.abs(half * q * r)) & (p > q * (a - xf)) & (p < q * (b - xf)):
                    rat = (p + zero) / q
                    x_par = xf + rat
                    if ((x_par - a) < tol2) | ((b - x_par) < tol2):
                        dm = xm - xf
                        rat = tol1 * ((dm > 0).to(tl.float64) - (dm < 0).to(tl.float64) + (dm == 0).to(tl.float64))
                else:
                    golden = True
            if golden:
                if xf >= xm:
                    e = a - xf
                else:
                    e = b - xf
                rat = golden_mean * e
            si = (rat > 0).to(tl.float64) - (rat < 0).to(tl.float64) + (rat == 0).to(tl.float64)
            x = xf + si * tl.maximum(tl.abs(rat), tol1)
            fu = _yj_neg_llf(S, n, x, sign_log, BLOCK)
            num += 1
            if fu <= fx:
                if x >= xf:
                    a = xf
                else:
                    b = xf
                fulc = nfc
                ffulc = fnfc
                nfc = xf
                fnfc = fx
                xf = x
                fx = fu
            else:
                if x < xf:
                    a = x
                else:
                    b = x
                if (fu <= fnfc) | (nfc == xf):
                    fulc = nfc
                    ffulc = fnfc
                    nfc = x
                    fnfc = fu
                elif (fu <= ffulc) | (fulc == xf) | (fulc == nfc):
                    fulc = x
                    ffulc = fu
            xm = half * (a + b)
            tol1 = sqrt_eps * tl.abs(xf) + xatol_3
            tol2 = two * tol1
        tl.store(OUT + col, xf)


def yeojohnson_lambdas(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """Yeo-Johnson MLE lambdas of the columns of a finite, mixed-sign float64 CUDA matrix: scipy's
    bounded Brent between per-column bounds ``lb``/``ub`` on the mixed-sign log-likelihood, one
    program per column."""
    n = x.shape[0]
    A = torch.log1p(x.abs())
    S = torch.where(x >= 0, A, -A).t().contiguous()  # signed log1p|x|, one row per column
    sign_log = (torch.sign(x) * A).sum(0)
    out = torch.empty(x.shape[1], device=x.device, dtype=torch.float64)
    _yj_brent_kernel[(x.shape[1],)](S, lb.contiguous(), ub.contiguous(), sign_log, out, n, BLOCK=1024,
                                    num_warps=8)
    return out


if triton is not None:

    @triton.jit(do_not_specialize=["n_rows", "L"])
    def _qassmax_kernel(Q, W1, B1, W2, B2, BASE, Y, n_rows, L, sqn, sqh, sql,
                        H: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr, BLOCK_M: tl.constexpr,
                        BIAS1_SEPARATE: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = rows < n_rows
        n, h, pos = rows // (H * L), (rows // L) % H, rows % L
        d = tl.arange(0, HD)
        f = tl.arange(0, NH)
        q = tl.load(Q + (n * sqn + h * sqh + pos * sql)[:, None] + d[None, :], mask=mask[:, None], other=0.0)
        h1 = tl.dot(q, tl.load(W1 + f[None, :] * HD + d[:, None]))  # W1 (NH, HD) as (HD, NH)
        if BIAS1_SEPARATE:  # linear on a non-contiguous 4-d input: fp16 matmul, then the fp16 bias add
            h1 = h1.to(tl.float16).to(tl.float32)
        h1 = (h1 + tl.load(B1 + f)[None, :].to(tl.float32)).to(tl.float16).to(tl.float32)
        g = (0.5 * h1 * (1.0 + tl.math.erf(h1 * 0.7071067811865476))).to(tl.float16)
        m = tl.dot(g, tl.load(W2 + d[None, :] * NH + f[:, None]))  # W2 (HD, NH) as (NH, HD)
        m = (m + tl.load(B2 + d)[None, :].to(tl.float32)).to(tl.float16).to(tl.float32)
        mod = (1.0 + libdevice.tanh(m).to(tl.float16).to(tl.float32)).to(tl.float16)
        base = tl.load(BASE + h[:, None] * HD + d[None, :], mask=mask[:, None], other=0.0)
        s = (base.to(tl.float32) * mod.to(tl.float32)).to(tl.float16)
        y = (q.to(tl.float32) * s.to(tl.float32)).to(tl.float16)
        tl.store(Y + rows[:, None] * HD + d[None, :], y, mask=mask[:, None])


def can_qassmax_fp16(q: Tensor, layer) -> bool:
    lin1, lin2 = layer.query_mlp[0], layer.query_mlp[2]
    return (triton is not None and q.is_cuda and q.dtype == torch.float16 and q.dim() == 4 and q.stride(-1) == 1
            and q.shape[-1] in (16, 32, 64, 128) and lin1.out_features in (16, 32, 64, 128)
            and lin1.bias is not None and lin2.bias is not None and isinstance(layer.query_mlp[1], torch.nn.GELU)
            and layer.query_mlp[1].approximate == "none")


def qassmax_fp16(q: Tensor, base_scales: Tensor, layer) -> Tensor:
    """``q * (base_scales * (1 + tanh(query_mlp(q))))`` for fp16 (N, heads, L, head_dim) ``q`` under autocast,
    in one pass, rounding where the eager ops do (fp16 after each op; the first linear's bias added
    separately when ``q`` is not contiguous, as ``F.linear`` then does); returns a contiguous tensor."""
    N, H, L, HD = q.shape
    w1, b1 = _fp16_params(layer.query_mlp[0])
    w2, b2 = _fp16_params(layer.query_mlp[2])
    y = torch.empty(N, H, L, HD, device=q.device, dtype=torch.float16)
    n_rows = N * H * L
    block_m = 64
    _qassmax_kernel[(triton.cdiv(n_rows, block_m),)](
        q, w1, b1, w2, b2, base_scales.reshape(H, HD).contiguous(), y, n_rows, L, q.stride(0), q.stride(1),
        q.stride(2), H=H, HD=HD, NH=w1.shape[0], BLOCK_M=block_m, BIAS1_SEPARATE=not q.is_contiguous(),
        num_warps=4, num_stages=2)
    return y


# ---------------------------------------------------------------------------------------------------
# Training attention: FlashAttention-2 forward + backward for fp32 inputs with 3xTF32 tensor-core dots, at
# least as accurate against fp64 as SDPA's fp32 kernel on the trainer's real calls (plain TF32: 1e-3). The
# forward saves the per-row log-sum-exp; the backward runs two kernels (dK/dV per key block, dQ per query
# block): no atomics, deterministic. A single pass with atomic dQ was measured no faster.
# ---------------------------------------------------------------------------------------------------

if triton is not None:

    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attn_train_fwd(Q, K, V, Out, LSE, qk_scale, Lq, Lk,
                        sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som,
                        H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        PREC: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        in_q = offs_m < Lq
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=in_q[:, None], other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        k_base, v_base = K + b * skb + h * skh, V + b * svb + h * svh
        for start in range(0, Lk, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            in_k = offs_n < Lk
            k = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None], mask=in_k[None, :], other=0.0)
            s = tl.where(in_k[None, :], tl.dot(q, k, input_precision=PREC) * qk_scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :], mask=in_k[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=PREC)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :], acc, mask=in_q[:, None])
        tl.store(LSE + bh * Lq + offs_m, m_i + tl.math.log2(l_i), mask=in_q)  # base 2, scaled scores

    @triton.jit(do_not_specialize=["Lq"])
    def _attn_train_delta(Out, DO, DELTA, Lq, sob, soh, som, H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        in_q = offs_m < Lq
        off = b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :]
        o = tl.load(Out + off, mask=in_q[:, None], other=0.0)
        do = tl.load(DO + off, mask=in_q[:, None], other=0.0)
        tl.store(DELTA + bh * Lq + offs_m, tl.sum(o * do, 1), mask=in_q)

    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attn_train_bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, qk_scale, sm_scale, Lq, Lk,
                             sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som, sgb, sgh, sgl,
                             H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             PREC: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        in_k = offs_n < Lk
        k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=in_k[:, None], other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=in_k[:, None], other=0.0)
        dk = tl.zeros([BLOCK_N, D], tl.float32)
        dv = tl.zeros([BLOCK_N, D], tl.float32)
        q_base, do_base = Q + b * sqb + h * sqh, DO + b * sob + h * soh
        for start in range(0, Lq, BLOCK_M):
            offs_m = start + tl.arange(0, BLOCK_M)
            in_q = offs_m < Lq
            q = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :], mask=in_q[:, None], other=0.0)
            do = tl.load(do_base + offs_m[:, None] * som + offs_d[None, :], mask=in_q[:, None], other=0.0)
            lse = tl.load(LSE + bh * Lq + offs_m, mask=in_q, other=float("inf"))
            delta = tl.load(DELTA + bh * Lq + offs_m, mask=in_q, other=0.0)
            s = tl.dot(q, tl.trans(k), input_precision=PREC) * qk_scale  # (M, N)
            p = tl.where(in_k[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p), do, input_precision=PREC)
            dp = tl.dot(do, tl.trans(v), input_precision=PREC)
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds), q, input_precision=PREC)
        off = b * sgb + h * sgh + offs_n[:, None] * sgl + offs_d[None, :]  # DK, DV: strides (sgb, sgh, sgl)
        tl.store(DK + off, dk * sm_scale, mask=in_k[:, None])
        tl.store(DV + off, dv, mask=in_k[:, None])

    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attn_train_bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, qk_scale, sm_scale, Lq, Lk,
                           sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som, sgb, sgh, sgl,
                           H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           PREC: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        in_q = offs_m < Lq
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=in_q[:, None], other=0.0)
        do = tl.load(DO + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :], mask=in_q[:, None], other=0.0)
        lse = tl.load(LSE + bh * Lq + offs_m, mask=in_q, other=float("inf"))
        delta = tl.load(DELTA + bh * Lq + offs_m, mask=in_q, other=0.0)
        dq = tl.zeros([BLOCK_M, D], tl.float32)
        k_base, v_base = K + b * skb + h * skh, V + b * svb + h * svh
        for start in range(0, Lk, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            in_k = offs_n < Lk
            k = tl.load(k_base + offs_n[:, None] * skn + offs_d[None, :], mask=in_k[:, None], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :], mask=in_k[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k), input_precision=PREC) * qk_scale
            p = tl.where(in_k[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v), input_precision=PREC)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds, k, input_precision=PREC)
        tl.store(DQ + b * sgb + h * sgh + offs_m[:, None] * sgl + offs_d[None, :], dq * sm_scale, mask=in_q[:, None])


    # fp16-split variant (head dim 64, large attention): every fp32 operand x (scaled by a power of two
    # into [2^7, 2^8), or per tile for dS) is hi = fp16(x) plus lo = fp16((x - hi) * 2^11), and each dot is
    # hi.hi + (hi.lo + lo.hi) * 2^-11: three fp16 MMAs (twice TF32's rate) with fp32 accumulation, 22 bits
    # of each operand, as accurate against fp64 as 3xTF32 and SDPA's fp32 kernel.
    @triton.jit
    def _split16(x):
        hi = x.to(tl.float16)
        lo = ((x - hi.to(tl.float32)) * 2048.0).to(tl.float16)
        return hi, lo


    @triton.jit
    def _dot16x3(a_hi, a_lo, b_hi, b_lo):
        return tl.dot(a_hi, b_hi) + (tl.dot(a_hi, b_lo) + tl.dot(a_lo, b_hi)) * (1.0 / 2048.0)


    @triton.jit
    def _pow2_scale16(t):
        # power of two bringing max|t| (>0) into [2^7, 2^8)
        e = tl.math.floor(tl.math.log2(tl.maximum(t, 1e-30)))
        return tl.math.exp2(7.0 - e)


    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attn16_fwd(QH, QL, KH, KL, VH, VL, Out, LSE, SC, qk_scale, Lq, Lk, sob, soh, som,
                   H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # QH/QL (N*H, Lq, D) fp16, KH.. (N*H, Lk, D); SC = [sq, sk, sv] power-of-two scales applied
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        in_q = offs_m < Lq
        qoff = bh * Lq * D + offs_m[:, None] * D + offs_d[None, :]
        qh = tl.load(QH + qoff, mask=in_q[:, None], other=0.0)
        ql = tl.load(QL + qoff, mask=in_q[:, None], other=0.0)
        sq, sk, sv = tl.load(SC), tl.load(SC + 1), tl.load(SC + 2)
        s_scale = qk_scale / (sq * sk)
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        for start in range(0, Lk, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            in_k = offs_n < Lk
            koff = bh * Lk * D + offs_n[None, :] * D + offs_d[:, None]
            kh = tl.load(KH + koff, mask=in_k[None, :], other=0.0)
            kl = tl.load(KL + koff, mask=in_k[None, :], other=0.0)
            s = tl.where(in_k[None, :], _dot16x3(qh, ql, kh, kl) * s_scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            voff = bh * Lk * D + offs_n[:, None] * D + offs_d[None, :]
            vh = tl.load(VH + voff, mask=in_k[:, None], other=0.0)
            vl = tl.load(VL + voff, mask=in_k[:, None], other=0.0)
            ph, pl = _split16(p)
            acc = acc * alpha[:, None] + _dot16x3(ph, pl, vh, vl)
            m_i = m_new
        acc = acc / (l_i[:, None] * sv)
        tl.store(Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :], acc, mask=in_q[:, None])
        tl.store(LSE + bh * Lq + offs_m, m_i + tl.math.log2(l_i), mask=in_q)


    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attn16_bwd_dkdv(QH, QL, KH, KL, VH, VL, DOH, DOL, LSE, DELTA, DK, DV, SC, qk_scale, sm_scale, Lq, Lk,
                    sgb, sgh, sgl, H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        in_k = offs_n < Lk
        sq, sk, sv, sdo = tl.load(SC), tl.load(SC + 1), tl.load(SC + 2), tl.load(SC + 3)
        s_scale = qk_scale / (sq * sk)
        dp_scale = 1.0 / (sdo * sv)
        koff = bh * Lk * D + offs_n[:, None] * D + offs_d[None, :]
        kh = tl.load(KH + koff, mask=in_k[:, None], other=0.0)
        kl = tl.load(KL + koff, mask=in_k[:, None], other=0.0)
        vh = tl.load(VH + koff, mask=in_k[:, None], other=0.0)
        vl = tl.load(VL + koff, mask=in_k[:, None], other=0.0)
        dk = tl.zeros([BLOCK_N, D], tl.float32)
        dv = tl.zeros([BLOCK_N, D], tl.float32)
        for start in range(0, Lq, BLOCK_M):
            offs_m = start + tl.arange(0, BLOCK_M)
            in_q = offs_m < Lq
            qoff = bh * Lq * D + offs_m[:, None] * D + offs_d[None, :]
            qh = tl.load(QH + qoff, mask=in_q[:, None], other=0.0)
            ql = tl.load(QL + qoff, mask=in_q[:, None], other=0.0)
            doh = tl.load(DOH + qoff, mask=in_q[:, None], other=0.0)
            dol = tl.load(DOL + qoff, mask=in_q[:, None], other=0.0)
            lse = tl.load(LSE + bh * Lq + offs_m, mask=in_q, other=float("inf"))
            delta = tl.load(DELTA + bh * Lq + offs_m, mask=in_q, other=0.0)
            s = _dot16x3(qh, ql, tl.trans(kh), tl.trans(kl)) * s_scale
            p = tl.where(in_k[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
            ph, pl = _split16(p)
            dv += _dot16x3(tl.trans(ph), tl.trans(pl), doh, dol)
            dp = _dot16x3(doh, dol, tl.trans(vh), tl.trans(vl)) * dp_scale
            ds = p * (dp - delta[:, None])
            c = _pow2_scale16(tl.max(tl.max(tl.abs(ds), 1), 0))
            dsh, dsl = _split16(ds * c)
            dk += _dot16x3(tl.trans(dsh), tl.trans(dsl), qh, ql) * (1.0 / c)
        goff = (bh // H) * sgb + (bh % H) * sgh + offs_n[:, None] * sgl + offs_d[None, :]
        tl.store(DK + goff, dk * (sm_scale / sq), mask=in_k[:, None])
        tl.store(DV + goff, dv * (1.0 / sdo), mask=in_k[:, None])


    @triton.jit(do_not_specialize=["Lq", "Lk"])
    def _attn16_bwd_dq(QH, QL, KH, KL, VH, VL, DOH, DOL, LSE, DELTA, DQ, SC, qk_scale, sm_scale, Lq, Lk,
                  sgb, sgh, sgl, H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        in_q = offs_m < Lq
        sq, sk, sv, sdo = tl.load(SC), tl.load(SC + 1), tl.load(SC + 2), tl.load(SC + 3)
        s_scale = qk_scale / (sq * sk)
        dp_scale = 1.0 / (sdo * sv)
        qoff = bh * Lq * D + offs_m[:, None] * D + offs_d[None, :]
        qh = tl.load(QH + qoff, mask=in_q[:, None], other=0.0)
        ql = tl.load(QL + qoff, mask=in_q[:, None], other=0.0)
        doh = tl.load(DOH + qoff, mask=in_q[:, None], other=0.0)
        dol = tl.load(DOL + qoff, mask=in_q[:, None], other=0.0)
        lse = tl.load(LSE + bh * Lq + offs_m, mask=in_q, other=float("inf"))
        delta = tl.load(DELTA + bh * Lq + offs_m, mask=in_q, other=0.0)
        dq = tl.zeros([BLOCK_M, D], tl.float32)
        for start in range(0, Lk, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            in_k = offs_n < Lk
            koff = bh * Lk * D + offs_n[:, None] * D + offs_d[None, :]
            kh = tl.load(KH + koff, mask=in_k[:, None], other=0.0)
            kl = tl.load(KL + koff, mask=in_k[:, None], other=0.0)
            vh = tl.load(VH + koff, mask=in_k[:, None], other=0.0)
            vl = tl.load(VL + koff, mask=in_k[:, None], other=0.0)
            s = _dot16x3(qh, ql, tl.trans(kh), tl.trans(kl)) * s_scale
            p = tl.where(in_k[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
            dp = _dot16x3(doh, dol, tl.trans(vh), tl.trans(vl)) * dp_scale
            ds = p * (dp - delta[:, None])
            c = _pow2_scale16(tl.max(tl.max(tl.abs(ds), 1), 0))
            dsh, dsl = _split16(ds * c)
            dq += _dot16x3(dsh, dsl, kh, kl) * (1.0 / c)
        goff = (bh // H) * sgb + (bh % H) * sgh + offs_m[:, None] * sgl + offs_d[None, :]
        tl.store(DQ + goff, dq * (sm_scale / sk), mask=in_q[:, None])


    @triton.jit(do_not_specialize=["L"])
    def _split16_rows(X, HI, LO, SC, L, sxb, sxh, sxl, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        # x * SC[0] (strided (N, H, L, D) fp32) -> hi, lo * 2^11 (contiguous (N*H, L, D) fp16)
        bh = tl.program_id(0).to(tl.int64)
        b, h = bh // H, bh % H
        offs_l = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offs_d = tl.arange(0, D)
        m = (offs_l < L)[:, None]
        x = tl.load(X + b * sxb + h * sxh + offs_l[:, None] * sxl + offs_d[None, :], mask=m, other=0.0) * tl.load(SC)
        hi, lo = _split16(x)
        off = bh * L * D + offs_l[:, None] * D + offs_d[None, :]
        tl.store(HI + off, hi, mask=m)
        tl.store(LO + off, lo, mask=m)

TRAIN_ATTENTION_HEAD_DIMS = (16, 64)


def _grad_like(t: Tensor) -> Tensor:
    """Gradient buffer for (N, heads, L, hd) ``t`` in the (N, L, heads, hd) layout of the model's q/k/v views,
    so that the backward of their transpose and view needs no copy."""
    N, H, L, D = t.shape
    return torch.empty(N, L, H, D, device=t.device, dtype=t.dtype).transpose(1, 2)
TRAIN_ATTENTION_PRECISION = "tf32x3"
TRAIN_ATTENTION_STAGES = 2


def _train_attn_tiles(D, Lq, Lk):
    # (fwd BLOCK_M, BLOCK_N, warps), (bwd BLOCK_M, BLOCK_N, warps)
    if D <= 16:
        if max(Lq, Lk) <= 16:  # row stage: features + CLS tokens
            return (16, 16, 1), (16, 16, 1)
        if max(Lq, Lk) <= 32:
            return (32, 32, 2), (32, 32, 2)
        return (64 if Lq <= 64 else 128, 64, 4), (64, 64, 4)
    # head dim 64, small attention (ICL of stage 1): the backward's 3xTF32 dots spill with larger tiles
    return (64, 32, 4), (32, 32, 4)


class _TrainAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, prec):
        N, H, Lq, D = q.shape
        Lk = k.shape[-2]
        o = torch.empty(N, Lq, H, D, device=q.device, dtype=q.dtype).transpose(1, 2)
        lse = torch.empty(N * H, Lq, device=q.device, dtype=torch.float32)
        (fm, fn, fw), _ = _train_attn_tiles(D, Lq, Lk)
        qk_scale = 1.4426950408889634 / math.sqrt(D)
        _attn_train_fwd[(N * H, triton.cdiv(Lq, fm))](
            q, k, v, o, lse, qk_scale, Lq, Lk, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3],
            H=H, D=D, BLOCK_M=fm, BLOCK_N=fn, PREC=prec, num_warps=fw, num_stages=TRAIN_ATTENTION_STAGES)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.prec = prec
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        N, H, Lq, D = q.shape
        Lk = k.shape[-2]
        do = do if do.stride() == o.stride() else torch.empty_like(o).copy_(do)
        delta = torch.empty(N * H, Lq, device=q.device, dtype=torch.float32)
        _attn_train_delta[(N * H, triton.cdiv(Lq, 64))](o, do, delta, Lq, *o.stride()[:3], H=H, D=D, BLOCK_M=64,
                                                         num_warps=4)
        dq, dk, dv = _grad_like(q), _grad_like(k), _grad_like(v)
        _, (bm, bn, bw) = _train_attn_tiles(D, Lq, Lk)
        sm_scale = 1.0 / math.sqrt(D)
        qk_scale = sm_scale * 1.4426950408889634
        args = (qk_scale, sm_scale, Lq, Lk, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3])
        _attn_train_bwd_dkdv[(N * H, triton.cdiv(Lk, bn))](q, k, v, do, lse, delta, dk, dv, *args, *dk.stride()[:3],
                                                           H=H, D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=ctx.prec, num_warps=bw,
                                                           num_stages=TRAIN_ATTENTION_STAGES)
        _attn_train_bwd_dq[(N * H, triton.cdiv(Lq, bm))](q, k, v, do, lse, delta, dq, *args, *dq.stride()[:3],
                                                         H=H, D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=ctx.prec, num_warps=bw,
                                                         num_stages=TRAIN_ATTENTION_STAGES)
        return dq, dk, dv, None


TRAIN_ATTENTION_F16X3_MIN = 4e6  # N * Lq * Lk from which head dim 64 takes the fp16-split kernels
TRAIN_ATTENTION_F16X3_TILES = ((64, 32, 4, 3), (32, 32, 4, 2))  # (BLOCK_M, BLOCK_N, warps, stages): fwd, bwd


def _pow2_scale(x: Tensor) -> Tensor:
    """Power of two bringing max|x| into [2^7, 2^8), as a 0-d device tensor (no host sync)."""
    m = torch.linalg.vector_norm(x, ord=float("inf")).clamp_min(1e-30)
    return torch.exp2(7.0 - torch.floor(torch.log2(m)))


def _split16_host(x: Tensor, sc: Tensor):
    N, H, L, D = x.shape
    hi = torch.empty(N * H, L, D, device=x.device, dtype=torch.float16)
    lo = torch.empty_like(hi)
    _split16_rows[(N * H, triton.cdiv(L, 64))](x, hi, lo, sc, L, *x.stride()[:3], H=H, D=D, BLOCK=64,
                                                 num_warps=4)
    return hi, lo


class _TrainAttention16x3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        N, H, Lq, D = q.shape
        Lk = k.shape[-2]
        sc = torch.stack([_pow2_scale(q), _pow2_scale(k), _pow2_scale(v), torch.ones((), device=q.device)])
        (qh, ql), (kh, kl), (vh, vl) = (_split16_host(t, sc[i:]) for i, t in enumerate((q, k, v)))
        o = torch.empty(N, Lq, H, D, device=q.device, dtype=q.dtype).transpose(1, 2)
        lse = torch.empty(N * H, Lq, device=q.device, dtype=torch.float32)
        (bm, bn, w, st), _ = TRAIN_ATTENTION_F16X3_TILES
        _attn16_fwd[(N * H, triton.cdiv(Lq, bm))](qh, ql, kh, kl, vh, vl, o, lse, sc, 1.4426950408889634 / math.sqrt(D),
                                                  Lq, Lk, *o.stride()[:3], H=H, D=D, BLOCK_M=bm, BLOCK_N=bn,
                                                  num_warps=w, num_stages=st)
        ctx.save_for_backward(q, k, v, o, lse, sc)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, sc = ctx.saved_tensors
        N, H, Lq, D = q.shape
        Lk = k.shape[-2]
        do = do if do.stride() == o.stride() else torch.empty_like(o).copy_(do)
        delta = torch.empty(N * H, Lq, device=q.device, dtype=torch.float32)
        _attn_train_delta[(N * H, triton.cdiv(Lq, 64))](o, do, delta, Lq, *o.stride()[:3], H=H, D=D, BLOCK_M=64,
                                                         num_warps=4)
        sc = torch.cat([sc[:3], _pow2_scale(do)[None]])
        (qh, ql), (kh, kl), (vh, vl), (doh, dol) = (_split16_host(t, sc[i:]) for i, t in enumerate((q, k, v, do)))
        dq, dk, dv = _grad_like(q), _grad_like(k), _grad_like(v)
        _, (bm, bn, w, st) = TRAIN_ATTENTION_F16X3_TILES
        sm_scale = 1.0 / math.sqrt(D)
        args = (sc, sm_scale * 1.4426950408889634, sm_scale, Lq, Lk)
        _attn16_bwd_dkdv[(N * H, triton.cdiv(Lk, bn))](qh, ql, kh, kl, vh, vl, doh, dol, lse, delta, dk, dv, *args,
                                                       *dk.stride()[:3], H=H, D=D, BLOCK_M=bm, BLOCK_N=bn, num_warps=w,
                                                       num_stages=st)
        _attn16_bwd_dq[(N * H, triton.cdiv(Lq, bm))](qh, ql, kh, kl, vh, vl, doh, dol, lse, delta, dq, *args,
                                                     *dq.stride()[:3], H=H, D=D, BLOCK_M=bm, BLOCK_N=bn, num_warps=w,
                                                     num_stages=st)
        return dq, dk, dv


def can_attention_train(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return (triton is not None and q.is_cuda and q.dtype == k.dtype == v.dtype == torch.float32 and q.dim() == 4
            and q.shape[-1] in TRAIN_ATTENTION_HEAD_DIMS and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
            and k.shape[-2] > 0 and q.shape[-2] > 0)


@torch.compiler.disable
def attention_train(q: Tensor, k: Tensor, v: Tensor, precision: str | None = None) -> Tensor:
    """Differentiable ``F.scaled_dot_product_attention(q, k, v)`` (no mask, default scale) for fp32
    (N, heads, L, hd) inputs; returns (N, heads, Lq, hd) laid out as (N, Lq, heads, hd)."""
    if (precision is None and q.shape[-1] == 64
            and q.shape[0] * q.shape[-2] * k.shape[-2] >= TRAIN_ATTENTION_F16X3_MIN):
        return _TrainAttention16x3.apply(q, k, v)
    return _TrainAttention.apply(q, k, v, precision or TRAIN_ATTENTION_PRECISION)


# ---------------------------------------------------------------------------------------------------
# Training layer norm (fp32 in, fp32 out, fp32 statistics): forward saves mean and rstd; the backward
# computes dx per row and deterministic per-program partial sums of dweight/dbias, reduced by torch.
# 1.4-1.6x torch's kernels (its GammaBeta backward is slow for many short rows), same accuracy vs fp64.
# ---------------------------------------------------------------------------------------------------

if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _ln_train_fwd(X, W, B, Y, MEAN, RSTD, M, L, sxs, sxr, eps, D: tl.constexpr, BLOCK_R: tl.constexpr, HAS_W: tl.constexpr,
                HAS_B: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, D)
        m = (rows < M)[:, None]
        # X: (M / L) sets of L rows, strides sxs (set) and sxr (row); Y contiguous
        x = tl.load(X + ((rows // L) * sxs + (rows % L) * sxr)[:, None] + cols[None, :], mask=m, other=0.0)
        mean = tl.sum(x, 1) / D
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / D + eps)
        y = xc * rstd[:, None]
        if HAS_W:
            y = y * tl.load(W + cols)[None, :]
        if HAS_B:
            y = y + tl.load(B + cols)[None, :]
        tl.store(Y + rows[:, None] * D + cols[None, :], y, mask=m)
        tl.store(MEAN + rows, mean, mask=rows < M)
        tl.store(RSTD + rows, rstd, mask=rows < M)


    @triton.jit(do_not_specialize=["M"])
    def _ln_train_bwd(X, W, DY, MEAN, RSTD, DX, DWP, DBP, M, L, sxs, sxr, D: tl.constexpr, BLOCK_R: tl.constexpr, ROWS: tl.constexpr,
                HAS_W: tl.constexpr, HAS_B: tl.constexpr):
        # program: ROWS rows, BLOCK_R at a time; dx per row, partial dw/db of its rows to DWP/DBP[pid]
        pid = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, D)
        if HAS_W:
            w = tl.load(W + cols)
        dw = tl.zeros([D], tl.float32)
        db = tl.zeros([D], tl.float32)
        for r0 in range(0, ROWS, BLOCK_R):
            rows = pid * ROWS + r0 + tl.arange(0, BLOCK_R)
            m = (rows < M)[:, None]
            off = rows[:, None] * D + cols[None, :]
            x = tl.load(X + ((rows // L) * sxs + (rows % L) * sxr)[:, None] + cols[None, :], mask=m, other=0.0)
            dy = tl.load(DY + off, mask=m, other=0.0)
            mean = tl.load(MEAN + rows, mask=rows < M, other=0.0)
            rstd = tl.load(RSTD + rows, mask=rows < M, other=0.0)
            xhat = (x - mean[:, None]) * rstd[:, None]
            g = dy * w[None, :] if HAS_W else dy
            c1 = tl.sum(xhat * g, 1) / D
            c2 = tl.sum(g, 1) / D
            dx = (g - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + off, dx, mask=m)
            dw += tl.sum(dy * xhat, 0)
            db += tl.sum(dy, 0)
        if HAS_W:
            tl.store(DWP + pid * D + cols, dw)
        if HAS_B:
            tl.store(DBP + pid * D + cols, db)



LAYER_NORM_TRAIN_CFG = {128: (16, 256, 4), 512: (4, 64, 4)}  # D: (BLOCK_R, rows per backward program, warps)
LAYER_NORM_TRAIN_MIN_ELEMS = 2**22  # below, torch's kernels (fewer Python launches) are as fast


def _row_sets(x: Tensor) -> Tensor:
    """``x`` (..., L, D) as a (sets, L, D) view with unit stride on D (e.g. a row slice of a larger tensor),
    or a contiguous copy when the leading dims do not merge."""
    x3 = x.reshape(-1, x.shape[-1])[None] if x.is_contiguous() else None
    if x3 is None and x.stride(-1) == 1 and x.dim() >= 2:
        try:
            x3 = x.view(-1, *x.shape[-2:])
        except RuntimeError:
            pass
    return x3 if x3 is not None else x.reshape(-1, x.shape[-1]).contiguous()[None]


class _LayerNormTrain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps):
        shape, D = x.shape, x.shape[-1]
        x3 = _row_sets(x)
        M = x.numel() // D
        y = torch.empty(M, D, device=x.device, dtype=x.dtype)
        mean = torch.empty(M, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        br, _, nw = LAYER_NORM_TRAIN_CFG[D]
        _ln_train_fwd[(triton.cdiv(M, br),)](x3, w, b, y, mean, rstd, M, x3.shape[1], x3.stride(0), x3.stride(1), eps,
                                             D=D, BLOCK_R=br, HAS_W=w is not None,
                                       HAS_B=b is not None, num_warps=nw)
        ctx.save_for_backward(x3, w, mean, rstd)
        ctx.has_b = b is not None
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x3, w, mean, rstd = ctx.saved_tensors
        D = x3.shape[-1]
        M = x3.shape[0] * x3.shape[1]
        dy = dy.reshape(-1, D)
        if not dy.is_contiguous():
            dy = dy.contiguous()
        br, rows, nw = LAYER_NORM_TRAIN_CFG[D]
        P = triton.cdiv(M, rows)
        dx = torch.empty(M, D, device=x3.device, dtype=x3.dtype)
        dwp = torch.empty(P, D, device=x3.device, dtype=torch.float32) if w is not None else None
        dbp = torch.empty(P, D, device=x3.device, dtype=torch.float32) if ctx.has_b else None
        _ln_train_bwd[(P,)](x3, w, dy, mean, rstd, dx, dwp, dbp, M, x3.shape[1], x3.stride(0), x3.stride(1), D=D, BLOCK_R=br, ROWS=rows, HAS_W=w is not None,
                      HAS_B=ctx.has_b, num_warps=nw)
        return (dx.view(ctx.shape), dwp.sum(0) if w is not None else None, dbp.sum(0) if ctx.has_b else None, None)



def can_layer_norm_train(x: Tensor, norm) -> bool:
    return (triton is not None and x.is_cuda and x.dtype == torch.float32 and x.shape[-1] in LAYER_NORM_TRAIN_CFG
            and x.numel() >= LAYER_NORM_TRAIN_MIN_ELEMS and norm.elementwise_affine
            and (norm.weight is None or norm.weight.dtype == torch.float32))


def layer_norm_train(x: Tensor, norm) -> Tensor:
    """Differentiable ``norm(x)`` (``nn.LayerNorm`` over the last dim) for fp32 CUDA input."""
    return _LayerNormTrain.apply(x, norm.weight, norm.bias, norm.eps)


# ---------------------------------------------------------------------------------------------------
# Training linears: the forward and the weight/input gradients are the GEMMs torch's addmm autograd runs
# (addmm(b, x, W^T); dx = g W; dW = g^T x); the bias gradient is a Triton column sum (deterministic
# per-program partials), and for linear1 -> GELU the GELU backward and the bias sum are one pass.
# ---------------------------------------------------------------------------------------------------

if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _colsum(G, P, M, N, ROWS: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
        # P[pid] = sum of G's rows [pid * ROWS, (pid + 1) * ROWS); columns in blocks of BLOCK_N (grid axis 1)
        pid = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        in_n = cols < N
        acc = tl.zeros([BLOCK_N], tl.float32)
        for r0 in range(0, ROWS, BLOCK_R):
            rows = pid * ROWS + r0 + tl.arange(0, BLOCK_R)
            g = tl.load(G + rows[:, None] * N + cols[None, :], mask=(rows < M)[:, None] & in_n[None, :], other=0.0)
            acc += tl.sum(g, 0)
        tl.store(P + pid * N + cols, acc, mask=in_n)

    @triton.jit(do_not_specialize=["M"])
    def _gelu_bwd_colsum(DH, PRE, DPRE, P, M, N: tl.constexpr, ROWS: tl.constexpr, BLOCK_R: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, N)
        acc = tl.zeros([N], tl.float32)
        for r0 in range(0, ROWS, BLOCK_R):
            rows = pid * ROWS + r0 + tl.arange(0, BLOCK_R)
            m = (rows < M)[:, None]
            off = rows[:, None] * N + cols[None, :]
            dh = tl.load(DH + off, mask=m, other=0.0)
            x = tl.load(PRE + off, mask=m, other=0.0)
            # torch's exact gelu backward, dh * (cdf + x * pdf), with libdevice's erf and exp (CUDA's erff, expf)
            cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
            pdf = 0.3989422804014327 * libdevice.exp(x * x * -0.5)
            d = dh * (cdf + x * pdf)
            tl.store(DPRE + off, d, mask=m)
            acc += tl.sum(d, 0)
        tl.store(P + pid * N + cols, acc)


COLSUM_ROWS = 256


def _colsum_rows(N: int) -> int:
    return 16 if N <= 256 else 8


def colsum(g: Tensor) -> Tensor:
    """``g.sum(0)`` for a contiguous fp32 (M, N) CUDA tensor."""
    M, N = g.shape
    P = triton.cdiv(M, COLSUM_ROWS)
    part = torch.empty(P, N, device=g.device, dtype=torch.float32)
    bn = min(256, triton.next_power_of_2(N))
    _colsum[(P, triton.cdiv(N, bn))](g, part, M, N, ROWS=COLSUM_ROWS, BLOCK_R=16, BLOCK_N=bn, num_warps=4)
    return part.sum(0)


def _flat(t: Tensor) -> Tensor:
    t = t.reshape(-1, t.shape[-1])
    return t if t.is_contiguous() else t.contiguous()


class _LinearTrain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, gelu):
        x2 = _flat(x)
        out = torch.addmm(b, x2, w.t())
        if gelu:
            pre, out = out, F.gelu(out)
            ctx.save_for_backward(x2, w, pre)
        else:
            ctx.save_for_backward(x2, w)
        ctx.gelu, ctx.shape = gelu, x.shape
        return out.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x2, w = ctx.saved_tensors[:2]
        g2 = _flat(g)
        M, N = g2.shape
        if ctx.gelu:
            pre = ctx.saved_tensors[2]
            P = triton.cdiv(M, COLSUM_ROWS)
            part = torch.empty(P, N, device=g2.device, dtype=torch.float32)
            dpre = torch.empty_like(g2)
            _gelu_bwd_colsum[(P,)](g2, pre, dpre, part, M, N=N, ROWS=COLSUM_ROWS, BLOCK_R=_colsum_rows(N),
                                   num_warps=4)
            g2, db = dpre, part.sum(0)
        else:
            db = colsum(g2)
        dx = g2.mm(w).view(ctx.shape) if ctx.needs_input_grad[0] else None
        dw = g2.t().mm(x2)
        return dx, dw, db, None


def can_linear_train(x: Tensor, w: Tensor, b: Optional[Tensor], gelu: bool = False) -> bool:
    N = w.shape[0]
    return (triton is not None and x.is_cuda and x.dtype == w.dtype == torch.float32 and b is not None
            and x.numel() > 0 and (not gelu or (not N & (N - 1) and N <= 1024)))


def linear_train(x: Tensor, w: Tensor, b: Tensor, gelu: bool = False) -> Tensor:
    """Differentiable ``F.linear(x, w, b)`` (and exact ``F.gelu`` of it when ``gelu``) for fp32 CUDA input."""
    return _LinearTrain.apply(x, w, b, gelu)
