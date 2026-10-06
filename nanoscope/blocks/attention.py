"""Multi-head attention with grouped key/value heads, optional QK-norm, a positional scheme
and an optional sliding window.

Weight names follow the shipped Modern model: `q`, `k`, `v`, `proj`, `q_norm`, `k_norm`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope.blocks.norm import RMSNorm
from nanoscope.blocks.positional import NoPE
from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule, BlockSpec, build_option


@block("attention", "composite", reference="naive_causal_attention",
       features=("gqa", "qk_norm", "window"))
class Attention(BlockModule):
    """n_heads query heads share n_kv_heads key/value heads (1 = multi-query, n_heads =
    plain multi-head). `pos` is a positional block (RoPE, NoPE) built at the head size;
    `window` limits each query to that many most recent positions."""

    def __init__(self, d_model: int, context_length: int, n_heads: int,
                 n_kv_heads: int | None = None, pos: BlockSpec | nn.Module | None = None,
                 qk_norm: bool = False, window: int | None = None,
                 bias: bool = False) -> None:
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError(f"Attention: d_model {d_model} must divide by n_heads {n_heads}, "
                             f"and n_heads by n_kv_heads {n_kv_heads}")
        if window is not None and window < 1:
            raise ValueError(f"Attention window must be at least 1, got {window}")
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.q = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.q_norm = RMSNorm().build(self.head_dim, context_length) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm().build(self.head_dim, context_length) if qk_norm else nn.Identity()
        self.pos: nn.Module = (NoPE().build(self.head_dim, context_length) if pos is None
                    else build_option(pos, self.head_dim, context_length))
        self.mask: torch.Tensor | None
        if window is None:
            self.mask = None
        else:
            t = torch.arange(context_length)
            band = (t[:, None] >= t[None, :]) & (t[:, None] - t[None, :] < window)
            self.register_buffer("mask", band, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.pos(self.q_norm(q), self.k_norm(k))
        # Each key/value head serves n_heads / n_kv_heads query heads.
        groups = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        if self.mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask[:T, :T])
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))

    def flops_per_token(self, context_length: int) -> int:
        span = context_length if self.window is None else min(self.window, context_length)
        weights = sum(p.numel() for m in (self.q, self.k, self.v, self.proj)
                      for p in m.parameters())
        total = 6 * weights + 2 * 6 * self.d_model * span
        for part in (self.q_norm, self.k_norm, self.pos):
            if isinstance(part, BlockModule):
                total += part.flops_per_token(context_length)  # type: ignore[operator]
        return total
