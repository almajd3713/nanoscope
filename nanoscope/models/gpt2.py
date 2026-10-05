"""GPT-2-style transformer: learned positions, LayerNorm, GELU MLP, multi-head attention.

This is the control the modern block (modern.py) is measured against.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope.sizing import count_params


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying, as in GPT-2

        # GPT-2 init: N(0, 0.02), with residual projections scaled down by depth.
        for name, p in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif p.ndim == 2:
                std = 0.02 / math.sqrt(2 * n_layers) if name.endswith("proj.weight") else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def flops_per_token(self, context_length: int) -> int:
        """Training FLOPs per token (PaLM, appendix B): 6 per weight used, including the
        tied output matrix, plus 12 * layers * d_model * context for attention scores."""
        _, non_embedding = count_params(self)
        tied = self.head.weight is self.tok_emb.weight
        unembed = self.head.weight.numel() if tied else 0
        return 6 * (non_embedding + unembed) + 12 * len(self.blocks) * self.d_model * context_length

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.size(1)
        assert T <= self.context_length, f"sequence of {T} tokens > context_length"  # noqa: SIM300
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
