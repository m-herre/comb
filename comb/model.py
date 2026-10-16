# Ported from tabicl 2.2.0 (BSD-3-Clause, Copyright (c) 2025, Soda team @ Inria); see LICENSE-tabicl.
"""Comb's network: column embedding -> row interaction -> in-context learning over cells (``cell.py``).

The column and row stages are TabICLv2's default path (feature grouping "same", non-interleaved RoPE, elementwise
QASSMax, pre-norm); module attribute names follow upstream TabICL.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .kernels import (attention, attention_block_params, attention_train, can_attention, can_attention_train,
                      can_ffn_residual, can_layer_norm_train, can_qassmax_fp16, can_rope_fp16, can_rope_train,
                      can_linear_train, ffn_residual, layer_norm_fp16, layer_norm_train, linear_residual, linear_train,
                      ln_linear_fp16, qassmax_fp16, rope_fp16, rope_train, triton)

SKIP_VALUE = -100.0


# ---------------------------------------------------------------------------------------------
# Small layers
# ---------------------------------------------------------------------------------------------


class OneHotAndLinear(nn.Linear):
    """One-hot encode class indices, then apply a linear projection."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot.float(), self.weight, self.bias)


class SkippableLinear(nn.Linear):
    """Linear layer whose output is ``skip_value`` wherever every input channel is ``skip_value``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = SKIP_VALUE):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value
        return out


class QASSMaxMLP(nn.Module):
    """Elementwise query-aware scalable softmax: ``q * base_mlp(log n) * (1 + tanh(query_mlp(q)))``."""

    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base_mlp = nn.Sequential(nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, num_heads * head_dim))
        self.query_mlp = nn.Sequential(nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, head_dim))
        nn.init.zeros_(self.query_mlp[-1].weight)  # initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].bias)

    def base_scales(self, q: Tensor, n: int) -> Tensor:
        if torch.is_grad_enabled():
            return self._base_scales(q, n)
        # no grad: a function of (n, dtype, device, autocast, weights) only, so computed once per key (each call cost
        # a host-to-device copy and three launches)
        key = (n, q.dtype, q.device, torch.is_autocast_enabled("cuda"),
               tuple(w._version for w in self.base_mlp.parameters()))  # an optimizer step invalidates it
        cache = self.__dict__.setdefault("_scales", {})
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[key] = self._base_scales(q, n)
        return cache[key]

    def _base_scales(self, q: Tensor, n: int) -> Tensor:
        logn = torch.tensor(math.log(max(n, 1)), device=q.device, dtype=q.dtype).reshape(1, 1)
        return self.base_mlp(logn).view(1, self.num_heads, 1, self.head_dim)

    def forward(self, q: Tensor, n: int) -> Tensor:
        base_scales = self.base_scales(q, n)
        if not torch.is_grad_enabled() and _fp16_autocast() and can_qassmax_fp16(q, self):
            return qassmax_fp16(q, base_scales, self)  # inference: the elementwise MLP in one pass
        modulation = 1 + torch.tanh(self.query_mlp(q))  # (bs, n_heads, seq_len, head_dim)
        return q * (base_scales * modulation)


# ---------------------------------------------------------------------------------------------
# Rotary position embedding (non-interleaved: halves [0:d/2] and [d/2:d] are rotated together)
# ---------------------------------------------------------------------------------------------


def rotate_half_contiguous(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


@torch.autocast("cuda", enabled=False)
def apply_rotary_emb(freqs: Tensor, t: Tensor) -> Tensor:
    """Rotate the last dim of ``t`` (fully, ``rot_dim == head_dim``) by angles ``freqs`` of shape (L, d/2)."""
    dtype = t.dtype
    assert freqs.shape[-1] * 2 == t.shape[-1], "RoPE must rotate the full head dimension"
    cos_freq = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin_freq = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return (t * cos_freq + rotate_half_contiguous(t) * sin_freq).type(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE over the sequence dim (-2); ``freqs`` is a non-trainable parameter, as upstream."""

    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def angles(self, seq_len: int, dtype: torch.dtype, device) -> Tensor:
        """(seq_len, dim / 2) rotation angles for positions 0..seq_len-1 held in ``dtype``."""
        seq = torch.arange(seq_len, device=device, dtype=dtype)  # positions in t's dtype
        return torch.einsum("..., f -> ... f", seq.type(self.freqs.dtype), self.freqs)

    def rotate_queries_or_keys(self, t: Tensor) -> Tensor:
        return apply_rotary_emb(self.angles(t.shape[-2], t.dtype, t.device), t)


# ---------------------------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------------------------


SDPA_MAX_BATCH = 32768  # flattened batch per CUDA SDPA call (grid limit 65535)
COL_CHUNK_ELEMS = 2**22  # default (member, column, row) elements per column-stage chunk at inference


class MultiheadAttention(nn.MultiheadAttention):
    """Batch-first MHA over arbitrary leading batch dims, with optional RoPE (q, k) and SSMax (q)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, ssmax: bool = False):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)
        self.ssmax_layer = QASSMaxMLP(num_heads, embed_dim // num_heads) if ssmax else None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, rope: Optional[RotaryEmbedding] = None,
                key_bias: Optional[Tensor] = None, ssmax_n: Optional[int] = None) -> Tensor:
        q, k, v = self._in_projection(query, key, value)
        return _linear(self.attend(q, k, v, rope, key_bias, ssmax_n), self.out_proj.weight, self.out_proj.bias)

    def attend(self, q: Tensor, k: Tensor, v: Tensor, rope: Optional[RotaryEmbedding] = None,
               key_bias: Optional[Tensor] = None, ssmax_n: Optional[int] = None) -> Tensor:
        """Attention of projected q (..., Lq, E) and k, v (..., Lk, E), with RoPE on q and k when given
        and SSMax on q; returns (..., Lq, E), contiguous, before the output projection.

        ``key_bias`` (..., Lk): an additive logit bias per key (Cell-ICL: log of the cell's row count, -inf for
        an empty or padded cell), the same for every query and head. ``ssmax_n``: the n of SSMax's log n scale
        (default: the number of keys; Cell-ICL passes the number of rows the cells summarise)."""
        *batch_shape, tgt_len, embed_dim = q.shape
        nh = self.num_heads
        hd = embed_dim // nh
        src_len = k.shape[-2]
        q = q.view(*batch_shape, tgt_len, nh, hd)
        k = k.view(*batch_shape, src_len, nh, hd)
        v = v.view(*batch_shape, src_len, nh, hd).transpose(-3, -2)
        if (rope is not None and not torch.is_grad_enabled() and _fp16_autocast()
                and can_rope_fp16(q) and can_rope_fp16(k)):
            # Inference only (the kernel has no backward): fused kernel on the (..., L, heads, head_dim)
            # layout; same fp32 maths and angles
            q = rope_fp16(q, rope.angles(tgt_len, q.dtype, q.device)).transpose(-3, -2)
            k = rope_fp16(k, rope.angles(src_len, k.dtype, k.device)).transpose(-3, -2)
        elif rope is not None and torch.is_grad_enabled() and can_rope_train(q) and can_rope_train(k):
            # training (fp32): one Triton pass each way instead of the eager cos/sin/cat/mul/add chain
            q = rope_train(q, rope.angles(tgt_len, q.dtype, q.device)).transpose(-3, -2)
            k = rope_train(k, rope.angles(src_len, k.dtype, k.device)).transpose(-3, -2)
        else:
            q, k = q.transpose(-3, -2), k.transpose(-3, -2)
            if rope is not None:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)

        # Flatten the batch dims for SDPA; SSMax scales q by the number of keys.
        q_shape = q.shape
        q = q.reshape(-1, *q.shape[-3:])
        k = k.reshape(-1, *k.shape[-3:])
        v = v.reshape(-1, *v.shape[-3:])
        if self.ssmax_layer is not None:
            q = self.ssmax_layer(q, k.size(-2) if ssmax_n is None else ssmax_n)
        dropout_p = self.dropout if self.training else 0.0
        if key_bias is not None:
            # biased keys (Cell-ICL): SDPA with an additive mask; the fused kernels take no bias. The ICL stage
            # over <= M cells is small, so this path is not a throughput concern.
            mask = key_bias.reshape(-1, 1, 1, key_bias.shape[-1]).to(q.dtype)
            out = torch.cat([F.scaled_dot_product_attention(qc, kc, vc, mc, dropout_p)
                             for qc, kc, vc, mc in zip(q.split(SDPA_MAX_BATCH), k.split(SDPA_MAX_BATCH),
                                                       v.split(SDPA_MAX_BATCH), mask.split(SDPA_MAX_BATCH))]
                            ).view(q_shape)
        elif not torch.is_grad_enabled() and dropout_p == 0.0 and can_attention(q, k, v):
            # Inference only (the kernel has no backward): Triton kernel for the head dim 16 of the column
            # and row stages (fp16: 1.5-3.8x FlashAttention-2 here; fp32: 1.4-3.4x SDPA's fp32 kernel);
            # its grid has no 65535 limit on the batch.
            out = attention(q, k, v).reshape(q_shape)
        elif torch.is_grad_enabled() and dropout_p == 0.0 and can_attention_train(q, k, v):
            # Training (fp32): FlashAttention-2 forward and backward in Triton with split-precision dots
            # (3xTF32, or fp16 hi/lo for large head-dim-64 ICL attention), at least as accurate against fp64
            # as SDPA's fp32 memory-efficient kernels and 2-5x (head dim 16) or ~2x (large ICL) faster.
            out = attention_train(q, k, v).reshape(q_shape)
        elif q.is_cuda and q.shape[0] > SDPA_MAX_BATCH:
            # The fused CUDA SDPA kernels fail ("invalid configuration argument") once the flattened batch
            # exceeds the 65535 grid limit, e.g. the row interaction over 8 members x 50k rows. Batch
            # elements are independent, so splitting the batch is exact.
            out = torch.cat([F.scaled_dot_product_attention(qc, kc, vc, None, dropout_p)
                             for qc, kc, vc in zip(q.split(SDPA_MAX_BATCH), k.split(SDPA_MAX_BATCH),
                                                   v.split(SDPA_MAX_BATCH))]).view(q_shape)
        else:
            out = F.scaled_dot_product_attention(q, k, v, None, dropout_p).view(q_shape)

        return out.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)

    def _in_projection(self, query: Tensor, key: Tensor, value: Tensor):
        """``F._in_projection_packed`` (same GEMMs on the same weight slices), returning views of the
        packed projection instead of its contiguous copy."""
        w, b, E = self.in_proj_weight, self.in_proj_bias, query.shape[-1]
        if key is value:
            if query is key:
                return _linear(query, w, b).chunk(3, dim=-1)
            w_q, w_kv = w.split([E, E * 2])
            b_q, b_kv = b.split([E, E * 2])
            return (_linear(query, w_q, b_q), *_linear(key, w_kv, b_kv).chunk(2, dim=-1))
        w_q, w_k, w_v = w.chunk(3)
        b_q, b_k, b_v = b.chunk(3)
        return _linear(query, w_q, b_q), _linear(key, w_k, b_k), _linear(value, w_v, b_v)


def _fp16_autocast() -> bool:
    """CUDA autocast to fp16, the precision the fused inference kernels reproduce (the trainer autocasts to
    float32, which must keep the eager ops)."""
    return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16


def _norm_for_linear(norm: nn.LayerNorm, x: Tensor) -> Tensor:
    """``norm(x)`` for an output that only feeds linear layers.

    Under CUDA autocast, layer norm returns fp32 and the next linear casts it to fp16. For fp16 or
    fp32 input, ``layer_norm_fp16`` writes that fp16 result directly (fp32 statistics), which skips a
    full-size fp32 tensor and the cast pass. Inference only (the kernel has no backward, and the trainer's
    float32 autocast must keep fp32 norms). Training with fp32 input uses ``layer_norm_train`` (fp32 out,
    with backward); any other case is ``norm(x)``.
    """
    if x.dtype in (torch.float16, torch.float32) and x.is_cuda and not torch.is_grad_enabled() and _fp16_autocast():
        return layer_norm_fp16(x, norm)
    if torch.is_grad_enabled() and can_layer_norm_train(x, norm):
        return layer_norm_train(x, norm)  # training (fp32): Triton forward and backward
    return norm(x)


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    """Pre-norm attention + feedforward block. With ``train_size``, keys/values are the first
    ``train_size`` (normalised) positions of ``q``."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias_free_ln: bool = False,
        ssmax: bool = False,
        zero_init: bool = True,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation=activation, norm_first=True, batch_first=True)
        if bias_free_ln:
            self.norm1 = nn.LayerNorm(d_model, bias=False)
            self.norm2 = nn.LayerNorm(d_model, bias=False)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout, ssmax)
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def _ff_block(self, x: Tensor) -> Tensor:
        if (torch.is_grad_enabled() and self.activation is F.gelu and self.dropout.p == 0.0
                and can_linear_train(x, self.linear1.weight, self.linear1.bias, gelu=True)
                and can_linear_train(x, self.linear2.weight, self.linear2.bias)):
            # training (fp32): linear1 -> GELU -> linear2 with the GELU backward and bias sums fused
            h = linear_train(x, self.linear1.weight, self.linear1.bias, gelu=True)
            return linear_train(h, self.linear2.weight, self.linear2.bias)
        return super()._ff_block(x)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        query_start: int = 0,
        key_bias: Optional[Tensor] = None,
        ssmax_n: Optional[int] = None,
    ) -> Tensor:
        """``query_start`` (with ``train_size``, no RoPE): only rows from there on are queries, and only
        their outputs are returned (every row's output depends only on its own query and the train rows).
        ``key_bias`` / ``ssmax_n``: see ``MultiheadAttention.attend`` (Cell-ICL)."""
        if train_size is None and key_bias is None and ssmax_n is None and _same_view(k, v) and self._can_fuse(q, k):
            return self._forward_fused(q, k, rope)
        q_normed = _norm_for_linear(self.norm1, q)
        if train_size is None:
            k = q if k is None else k
            v = q if v is None else v
            k_normed = _norm_for_linear(self.norm1, k) if k is not q else q_normed
            v_normed = _norm_for_linear(self.norm1, v) if v is not k else k_normed
        else:
            assert k is None and v is None, "k and v must be None when train_size is provided"
            k_normed = v_normed = q_normed[..., :train_size, :]
            if query_start:
                assert rope is None, "query_start would shift the RoPE positions"
                q, q_normed = q[..., query_start:, :], q_normed[..., query_start:, :]
        x = q + self.dropout1(self.attn(q_normed, k_normed, v_normed, rope=rope, key_bias=key_bias, ssmax_n=ssmax_n))
        if (not torch.is_grad_enabled() and not self.training and _fp16_autocast()
                and can_ffn_residual(x, self)):
            return ffn_residual(x, self)  # norm2 -> linear1 -> GELU -> linear2 -> + x, fused
        return x + self._ff_block(_norm_for_linear(self.norm2, x))


def _linear(x: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
    """``F.linear``; in fp32 training ``linear_train`` (the same GEMMs, the bias gradient in one Triton pass)."""
    if torch.is_grad_enabled() and can_linear_train(x, w, b):
        return linear_train(x, w, b)
    return F.linear(x, w, b)


def _same_view(a: Optional[Tensor], b: Optional[Tensor]) -> bool:
    """Whether a and b are the same tensor or identical views of the same memory."""
    if a is b:
        return True
    return (a is not None and b is not None and a.data_ptr() == b.data_ptr() and a.shape == b.shape
            and a.stride() == b.stride() and a.dtype == b.dtype)


def _fusable_dtype(t: Optional[Tensor]) -> bool:
    return t is None or (t.is_cuda and t.dtype in (torch.float16, torch.float32))


def _mab_can_fuse(self, q: Tensor, k: Optional[Tensor]) -> bool:
    """Whether ``_forward_fused`` applies: inference under CUDA autocast, d_model 128 in heads of 16."""
    attn = self.attn
    return (triton is not None and not self.training and not torch.is_grad_enabled() and _fp16_autocast()
            and q.shape[-1] == 128 and attn.num_heads == 8 and attn.in_proj_bias is not None
            and self.norm1.weight is not None and _fusable_dtype(q) and _fusable_dtype(k) and attn.dropout == 0.0)


def _mab_forward_fused(self, q: Tensor, k: Optional[Tensor], rope: Optional[RotaryEmbedding]) -> Tensor:
    """``forward`` for self-attention (k None) or a shared key/value input, in fused kernels: norm1 +
    projection (+ RoPE) per input, the attention, then output projection + residual, then the fused
    feed-forward. Same rounding points as the eager path under autocast."""
    attn, E = self.attn, q.shape[-1]
    p = attention_block_params(attn)

    def project(x, rows, rope_cols):
        angles = rope.angles(x.shape[-2], torch.float16, x.device) if rope is not None else None
        return ln_linear_fp16(x, self.norm1, p["w"][rows], p["b"][rows], rope_cols if rope is not None else 0, angles)

    if k is None:
        qp, kp, vp = project(q, slice(0, 3 * E), 2 * E).chunk(3, dim=-1)
    else:
        qp = project(q, slice(0, E), E)
        kp, vp = project(k, slice(E, 3 * E), E).chunk(2, dim=-1)
    x = linear_residual(attn.attend(qp, kp, vp), p["wo"], p["bo"], q)
    if can_ffn_residual(x, self):
        return ffn_residual(x, self)
    return x + self._ff_block(_norm_for_linear(self.norm2, x))


MultiheadAttentionBlock._can_fuse = _mab_can_fuse
MultiheadAttentionBlock._forward_fused = _mab_forward_fused


class InducedSelfAttentionBlock(nn.Module):
    """ISAB: inducing points attend to the (train) rows, then the rows attend to the inducing points.
    Sets whose every value is the skip value (reserved CLS columns) pass through unchanged."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias_free_ln: bool = False,
        ssmax: bool = False,
        zero_init: bool = True,
        skip_value: float = SKIP_VALUE,
    ):
        super().__init__()
        self.skip_value = skip_value
        self.multihead_attn1 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, bias_free_ln, ssmax, zero_init
        )
        self.multihead_attn2 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, bias_free_ln, zero_init=zero_init
        )
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)
        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            train = src[..., :train_size, :]  # one object for keys and values: one norm, one packed projection
            hidden = self.multihead_attn1(ind_vectors, train, train)
        return self.multihead_attn2(src, hidden, hidden)

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        # A set is all skip values only if its first value is one, so the full check (a pass over
        # the whole activation tensor) runs only when some set passes that cheap test.
        skip_mask = src[..., 0, 0] == self.skip_value  # batch shape
        if skip_mask.any():
            skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if not skip_mask.any():
            return self.induced_attention(src, train_size)
        if skip_mask.all():
            return torch.full_like(src, self.skip_value)
        out = torch.empty_like(src)
        out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
        out[skip_mask] = self.skip_value
        return out


class SetTransformer(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, nhead: int, dim_feedforward: int, num_inds: int, **kw):
        super().__init__()
        self.blocks = nn.ModuleList(
            [InducedSelfAttentionBlock(d_model, nhead, dim_feedforward, num_inds, **kw) for _ in range(num_blocks)]
        )

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        for block in self.blocks:
            src = block(src, train_size)
        return src


class Encoder(nn.Module):
    def __init__(
        self, num_blocks: int, d_model: int, nhead: int, dim_feedforward: int, rope_base: Optional[float] = None, **kw
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.blocks = nn.ModuleList(
            [MultiheadAttentionBlock(d_model, nhead, dim_feedforward, **kw) for _ in range(num_blocks)]
        )
        self.rope = RotaryEmbedding(d_model // nhead, theta=rope_base) if rope_base is not None else None

    def forward(self, src: Tensor, train_size: Optional[int] = None, last_query_start: int = 0,
                key_bias: Optional[Tensor] = None, ssmax_n: Optional[int] = None) -> Tensor:
        """``last_query_start``: the last block computes only the rows from there on (see
        ``MultiheadAttentionBlock.forward``); the output then holds only those rows. ``key_bias`` /
        ``ssmax_n``: per-key logit bias and SSMax n of every block (Cell-ICL)."""
        for i, block in enumerate(self.blocks):
            start = last_query_start if i == len(self.blocks) - 1 else 0
            src = block(q=src, train_size=train_size, rope=self.rope, query_start=start, key_bias=key_bias,
                        ssmax_n=ssmax_n)
        return src


# ---------------------------------------------------------------------------------------------
# The three stages
# ---------------------------------------------------------------------------------------------


# -- many classes (inference only; upstream TabICL's strategy) ------------------------------------------
# More than ``max_classes`` classes: the column stage runs once per mixed-radix digit of the labels and
# averages (upstream ``ColEmbedding`` mixed-radix ensemble), and the ICL stage classifies hierarchically
# (upstream ``ICLearning._fit_hierarchical`` / ``_predict_hierarchical``): balanced class groups of at most
# ``max_classes``, the node's train rows are re-labelled with their group, and the leaf probabilities are
# chained with the group probabilities. No new weights: the pretrained model is reused per digit / node.

def mixed_radix_bases(num_classes: int, max_classes: int) -> List[int]:
    """Balanced bases [k_0 .. k_{D-1}], each <= max_classes, product >= num_classes (upstream)."""
    if num_classes <= max_classes:
        return [num_classes]
    D = math.ceil(math.log(num_classes) / math.log(max_classes))
    k = min(math.ceil(num_classes ** (1.0 / D)), max_classes)
    bases, product, idx = [k] * D, k**D, 0
    while product < num_classes and idx < D:
        if bases[idx] < max_classes:
            product = product // bases[idx] * (bases[idx] + 1)
            bases[idx] += 1
        idx += 1
    return bases


def mixed_radix_digit(y: Tensor, digit_idx: int, bases: List[int]) -> Tensor:
    """Digit ``digit_idx`` of y in the mixed-radix representation with ``bases`` (upstream)."""
    div = math.prod(bases[digit_idx + 1:])
    return torch.div(y, div, rounding_mode="floor").remainder(bases[digit_idx])


def class_groups(num_classes: int, max_classes: int) -> Tensor:
    """Group index per class: <= max_classes balanced contiguous groups (upstream ``_grouping``)."""
    num_groups = min(math.ceil(num_classes / max_classes), max_classes)
    groups = torch.zeros(num_classes, dtype=torch.long)
    pos, left = 0, num_classes
    for i in range(num_groups):
        size = math.ceil(left / (num_groups - i))
        groups[pos:pos + size] = i
        pos, left = pos + size, left - size
    return groups


class ColEmbedding(nn.Module):
    """Distribution-aware column embedding with a shared, target-aware set transformer."""

    def __init__(self, embed_dim, num_blocks, nhead, dim_feedforward, num_inds, feature_group_size, max_classes,
                 reserve_cls_tokens, dropout, activation, bias_free_ln, zero_init):  # fmt: skip
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.feature_group_size = feature_group_size
        self.max_classes = max_classes
        self.in_linear = SkippableLinear(feature_group_size, embed_dim)
        self.tf_col = SetTransformer(
            num_blocks, embed_dim, nhead, dim_feedforward, num_inds, dropout=dropout, activation=activation,
            bias_free_ln=bias_free_ln, ssmax=True, zero_init=zero_init,
        )  # fmt: skip
        self.y_encoder = OneHotAndLinear(max_classes, embed_dim)

    def feature_grouping(self, X: Tensor) -> Tensor:
        """Circular-permutation grouping ("same" mode): (B, T, H) -> (B, T, H, size)."""
        H = X.shape[-1]
        idxs = torch.arange(H, dtype=torch.long, device=X.device)
        return torch.stack([X[:, :, (idxs + 2**i) % H] for i in range(self.feature_group_size)], dim=-1)

    def _label_embedding(self, y_train: Tensor) -> Tensor:
        num_classes = int(y_train.max().item()) + 1
        if num_classes > self.max_classes:
            raise ValueError(f"Comb supports at most {self.max_classes} classes, got labels up to "
                             f"{num_classes - 1}")  # fmt: skip
        return self.y_encoder(y_train.float())

    def forward(self, X: Tensor, y_train: Tensor) -> Tensor:
        if self.training:
            return self._forward_train(X, y_train)
        num_classes = int(y_train.max().item()) + 1
        if num_classes > self.max_classes:  # many classes: average over the label digits (upstream)
            bases = mixed_radix_bases(num_classes, self.max_classes)
            y_long = y_train.long()
            out = sum(self._forward_eval(X, mixed_radix_digit(y_long, i, bases).to(y_train.dtype))
                      for i in range(len(bases)))
            return out / len(bases)
        return self._forward_eval(X, y_train)

    def _forward_eval(self, X: Tensor, y_train: Tensor) -> Tensor:
        """Inference: the training forward's values for every feature column, without its dead work.

        The reserved CLS columns are left out of the set transformer, because ``RowInteraction``
        overwrites their embeddings with its CLS tokens (sets are processed independently, so the
        feature columns are unaffected); their slots in the output are left unset. The label
        embedding is computed once; the input embedding is built per column chunk, columns leading
        ((chunk, B, T, E), contiguous, so the fused kernels need no copies), and the set transformer's
        output is written straight into a contiguous (B, T, C+H, E) output.
        """
        train_size = y_train.shape[1]
        C = self.reserve_cls_tokens
        features = self.feature_grouping(X)  # (B, T, H, size)
        label = self._label_embedding(y_train)  # (B, train, E)
        B, T, H, _ = features.shape
        chunk = max(1, COL_CHUNK_ELEMS // (B * T))
        out = None
        for start in range(0, H, chunk):
            src = self.in_linear(features[:, :, start:start + chunk].permute(2, 0, 1, 3))  # (chunk, B, T, E)
            src[..., :train_size, :] += label
            emb = self.tf_col(src, train_size=train_size)
            if out is None:
                out = torch.empty(B, T, C + H, emb.shape[-1], device=emb.device, dtype=emb.dtype)
            out[:, :, C + start:C + start + emb.shape[0]] = emb.permute(1, 2, 0, 3)
        return out

    # -- inference cache: train rows once, then any number of test-row chunks ----------------

    def _skip_sets(self, src: Tensor) -> bool:
        """Whether any set of ``src`` (..., rows, E) is all skip values (``InducedSelfAttentionBlock.forward``'s test)."""
        if not bool((src[..., 0, 0] == SKIP_VALUE).any()):
            return False
        return bool((src == SKIP_VALUE).all(dim=(-2, -1)).any())

    def train_hiddens(self, X_train: Tensor, y_train: Tensor, max_elems: int = COL_CHUNK_ELEMS) -> Optional[List[Tensor]]:
        """Inference: every set-transformer block's inducing-point states, (H, B, num_inds, E) per block, from
        the train rows alone. With them, ``embed_rows`` gives any row the embedding ``_forward_eval`` gives it:
        in the column stage a row attends only to these states, which depend only on the train rows. Returns
        None if a set of the train rows is all skip values (``forward`` would then decide skipping on the
        train and test rows together; the caller falls back to the full forward). Columns run in chunks of at
        most ``max_elems`` (column, member, row) elements (at least one column)."""
        if int(y_train.max().item()) + 1 > self.max_classes:
            return None  # many classes: the per-digit column stage is not cached (callers fall back)
        features = self.feature_grouping(X_train)  # (B, N, H, size)
        label = self._label_embedding(y_train)  # (B, N, E)
        B, N, H, _ = features.shape
        chunk = max(1, max_elems // (B * N))
        hiddens: List[List[Tensor]] = [[] for _ in self.tf_col.blocks]
        for start in range(0, H, chunk):
            src = self.in_linear(features[:, :, start:start + chunk].permute(2, 0, 1, 3)) + label  # (chunk, B, N, E)
            for j, block in enumerate(self.tf_col.blocks):
                if self._skip_sets(src):
                    return None
                ind = block.ind_vectors.expand(*src.shape[:-2], block.num_inds, src.shape[-1])
                hidden = block.multihead_attn1(ind, src, src)
                hiddens[j].append(hidden)
                src = block.multihead_attn2(src, hidden, hidden)
        return [torch.cat(h, dim=0) for h in hiddens]

    def embed_rows(self, X: Tensor, hiddens: List[Tensor], y: Optional[Tensor] = None) -> Tensor:
        """Inference: ``_forward_eval``'s (B, T, C+H, E) output for rows X (B, T, H), given ``train_hiddens``;
        ``y`` (B, T) marks the rows as train rows (their label embedding is added first). All columns run at once:
        the caller bounds the memory through the number of rows."""
        C = self.reserve_cls_tokens
        features = self.feature_grouping(X)
        B, T, H, _ = features.shape
        src = self.in_linear(features.permute(2, 0, 1, 3))  # (H, B, T, E)
        if y is not None:
            src = src + self._label_embedding(y)
        for j, block in enumerate(self.tf_col.blocks):
            src = block.multihead_attn2(src, hiddens[j], hiddens[j])
        out = torch.empty(B, T, C + H, src.shape[-1], device=src.device, dtype=src.dtype)
        out[:, :, C:] = src.permute(1, 2, 0, 3)
        return out

    def _forward_train(self, X: Tensor, y_train: Tensor) -> Tensor:
        train_size = y_train.shape[1]
        X = F.pad(self.feature_grouping(X), (0, 0, self.reserve_cls_tokens, 0), value=SKIP_VALUE)
        features = X.transpose(1, 2)  # (B, H+C, T, size)
        y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)

        src = self.in_linear(features)  # (B, H+C, T, E)
        num_classes = int(y_train.max().item()) + 1
        if num_classes > self.max_classes:
            raise ValueError(f"Comb supports at most {self.max_classes} classes, got labels up to "
                             f"{num_classes - 1}")  # fmt: skip
        y_emb = self.y_encoder(y_train.float())
        src[..., :train_size, :] = src[..., :train_size, :] + y_emb
        src = self.tf_col(src, train_size=train_size)
        return src.transpose(1, 2)  # (B, T, H+C, E)


class RowInteraction(nn.Module):
    """Row-wise transformer over the column embeddings; CLS tokens aggregate each row."""

    def __init__(self, embed_dim, num_blocks, nhead, dim_feedforward, num_cls, rope_base, dropout, activation,
                 bias_free_ln, zero_init):  # fmt: skip
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.tf_row = Encoder(
            num_blocks, embed_dim, nhead, dim_feedforward, rope_base=rope_base, dropout=dropout,
            activation=activation, bias_free_ln=bias_free_ln, zero_init=zero_init,
        )  # fmt: skip
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.out_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln)

    def forward(self, embeddings: Tensor) -> Tensor:
        B, T = embeddings.shape[:2]
        embeddings[:, :, : self.num_cls] = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        rope = self.tf_row.rope
        for block in self.tf_row.blocks[:-1]:
            embeddings = block(embeddings, rope=rope)
        # Last block: the queries are the CLS tokens only; keys/values are the full row.
        cls_out = self.tf_row.blocks[-1](q=embeddings[..., : self.num_cls, :], k=embeddings, v=embeddings, rope=rope)
        return self.out_ln(cls_out).flatten(-2)  # (B, T, C*E)


# ---------------------------------------------------------------------------------------------
# TabICL
# ---------------------------------------------------------------------------------------------


class TabICL(nn.Module):
    """Comb's network: TabICLv2's column embedding and row interaction, then the in-context stage over at most
    ``cell_m`` cells (``cell.CellICL``). Classification with up to ``max_classes`` classes (more at inference
    through the hierarchical path)."""

    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 8,
        col_num_inds: int = 128,
        col_feature_group_size: int = 3,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias_free_ln: bool = False,
        zero_init: bool = False,
        cell_m: int = 1024,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.max_classes = max_classes
        icl_dim = embed_dim * row_num_cls
        common = dict(dropout=dropout, activation=activation, bias_free_ln=bias_free_ln, zero_init=zero_init)
        self.col_embedder = ColEmbedding(
            embed_dim, col_num_blocks, col_nhead, embed_dim * ff_factor, col_num_inds, col_feature_group_size,
            max_classes, row_num_cls, **common,
        )  # fmt: skip
        self.row_interactor = RowInteraction(
            embed_dim, row_num_blocks, row_nhead, embed_dim * ff_factor, row_num_cls, row_rope_base, **common
        )
        from .cell import CellICL  # cell.py builds on this module's blocks

        self.icl_predictor = CellICL(
            max_classes, max_classes, icl_dim, icl_num_blocks, icl_nhead, icl_dim * ff_factor, num_cells=cell_m,
            **common,
        )

    def _forward(self, X: Tensor, y_train: Tensor) -> Tensor:
        assert y_train.shape[1] <= X.shape[1], "Number of training samples exceeds total samples"
        return self.icl_predictor(self.row_interactor(self.col_embedder(X, y_train)), y_train)

    def forward(self, X: Tensor, y_train: Tensor, softmax_temperature: float = 0.9) -> Tensor:
        """X: (B, T, H) with the first ``y_train.shape[1]`` rows labelled; returns (B, T - ts, out).

        ``train()`` mode returns all ``max_classes`` logits. ``eval()`` mode runs under ``no_grad`` and slices the
        logits to the number of classes in ``y_train[0]``; the caller owns precision (no autocast here). With more
        classes than the head, ``softmax_temperature`` is the caller's (see ``_forward_many_classes``).
        """
        if self.training:
            return self._forward(X, y_train)
        with torch.no_grad():
            if len(torch.unique(y_train[0])) > self.max_classes:
                return self._forward_many_classes(X, y_train, softmax_temperature)
            out = self._forward(X, y_train)
            num_classes = len(torch.unique(y_train[0]))
            assert all(len(torch.unique(yi)) == num_classes for yi in y_train), (
                "All tables must have the same number of classes"
            )
            return out[..., :num_classes]

    def _forward_many_classes(self, X: Tensor, y_train: Tensor, softmax_temperature: float = 0.9) -> Tensor:
        """Eval, more than ``max_classes`` classes: hierarchical classification (upstream). Returns
        ``T * log(p)`` (B, n_test, C) so the wrapper's ``softmax(avg / T)`` recovers the probabilities."""
        train_size = y_train.shape[1]
        R = self.row_interactor(self.col_embedder(X, y_train))
        icl, T = self.icl_predictor, softmax_temperature

        def probs(R_node, y_node, R_test):  # (n_test, k) for a node with labels 0..k-1
            k = int(y_node.max().item()) + 1
            logits = icl(torch.cat([R_node, R_test], 0)[None], y_node[None].to(y_train.dtype))[0, :, :k]
            return torch.softmax(logits.float() / T, -1)

        def node(R_node, y_node, R_test, num_classes):  # y_node: global labels -> (n_test, num_classes)
            classes = torch.unique(y_node)
            local = torch.searchsorted(classes, y_node)
            out = torch.zeros(R_test.shape[0], num_classes, device=R_test.device)
            if len(classes) <= self.max_classes:
                out[:, classes] = probs(R_node, local, R_test)
                return out
            groups = class_groups(len(classes), self.max_classes).to(y_node.device)
            g = groups[local]
            p_group = probs(R_node, g, R_test)
            for i in range(int(groups.max().item()) + 1):
                m = g == i
                out += node(R_node[m], y_node[m], R_test, num_classes) * p_group[:, i:i + 1]
            return out

        outs = []
        for b in range(X.shape[0]):
            yb = y_train[b].long()
            C = int(yb.max().item()) + 1
            p = node(R[b, :train_size], yb, R[b, train_size:], C)
            outs.append(T * torch.log(p + 1e-6))
        return torch.stack(outs)

    # -- inference: test rows against a cached summary ------------------------------------------------

    @torch.no_grad()
    def decode(self, cache: Dict, X_test: Tensor) -> Tensor:
        """Inference: outputs (B, n, out) for test rows X_test (B, n, H) in one pass, from the cache ``{"col":
        train_hiddens, "slots": the cell summary, "num_classes"}``; classification logits are sliced to the number
        of train classes, as ``forward`` in eval mode."""
        R = self.row_interactor(self.col_embedder.embed_rows(X_test, cache["col"]))
        out = self.icl_predictor.decode(R, cache["slots"])
        return out[..., : cache["num_classes"]]
