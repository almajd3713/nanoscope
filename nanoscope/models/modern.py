"""The 2024-26 consensus decoder: RoPE, RMSNorm, SwiGLU, GQA, QK-norm, no biases,
tied embeddings and z-loss.

Every component has a switch, so a leave-one-out ablation is one keyword:
Modern(rope=False) uses learned positions instead, swiglu=False a GELU MLP,
rmsnorm=False LayerNorm, qk_norm=False no QK-norm, n_kv_heads=n_heads plain
multi-head attention, z_loss=0 no z-loss, tie_weights=False a separate output matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """LayerNorm without centering or bias: x / rms(x) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight


def _norm(dim: int, rmsnorm: bool) -> nn.Module:
    return RMSNorm(dim) if rmsnorm else nn.LayerNorm(dim, bias=False)


def rope_tables(head_dim: int, length: int, device: torch.device, base: float = 10000.0):
    """cos/sin of angle m * theta_i for position m and frequency theta_i = base^(-2i/head_dim)."""
    theta = base ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(length, device=device).float(), theta)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each pair (x[i], x[i + head_dim/2]) by its position's angle. x: (B, H, T, D)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


class Attention(nn.Module):
    """Grouped-query attention: n_heads query heads share n_kv_heads key/value heads."""

    def __init__(
        self, d_model: int, n_heads: int, n_kv_heads: int, rope: bool, qk_norm: bool,
        context_length: int = 256,
    ):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        if rope:  # computed once; buffers follow the module to its device
            cos, sin = rope_tables(self.head_dim, context_length, torch.device("cpu"))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Each key/value head serves n_heads / n_kv_heads query heads.
        groups = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """silu(x W1) * (x W3), then W2. Hidden size 8d/3 keeps the parameter count of a 4d GELU MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = 8 * ((8 * d_model // 3 + 7) // 8)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.w1(x)) * self.w3(x))


class GELUMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(
        self, d_model, n_heads, n_kv_heads, rope, swiglu, rmsnorm, qk_norm, context_length,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(d_model, rmsnorm)
        self.attn = Attention(d_model, n_heads, n_kv_heads, rope, qk_norm, context_length)
        self.norm2 = _norm(d_model, rmsnorm)
        self.mlp = SwiGLU(d_model) if swiglu else GELUMLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Modern(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        rope: bool = True,
        swiglu: bool = True,
        rmsnorm: bool = True,
        qk_norm: bool = True,
        z_loss: float = 1e-4,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.z_loss = z_loss
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = None if rope else nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, n_kv_heads, rope, swiglu, rmsnorm, qk_norm, context_length)
            for _ in range(n_layers)
        )
        self.norm = _norm(d_model, rmsnorm)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02 / math.sqrt(2 * n_layers) if name.endswith("proj.weight") else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        T = idx.size(1)
        assert T <= self.context_length, f"sequence of {T} tokens > context_length"  # noqa: SIM300
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if not self.z_loss:
            return logits
        # z-loss keeps the softmax normalizer near 1 (log Z near 0), which stabilizes training.
        z = torch.logsumexp(logits.float(), dim=-1)
        return logits, self.z_loss * z.pow(2).mean()
