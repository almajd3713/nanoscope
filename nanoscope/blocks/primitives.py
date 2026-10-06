"""Primitives: the smallest blocks. Never locked, so lessons can build bigger blocks from them.

Each has `flops_per_token(context_length)`: training FLOPs per token (6 per weight used, and
6 * d_model * context for each of the two attention matmuls).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule

ACTIVATIONS = ("gelu", "silu", "relu")


@block("primitive", "primitive", reference="matmul")
class Linear(BlockModule):
    """x @ W.T (+ b), from `in_features` to `out_features` (both default to d_model)."""

    def __init__(self, d_model: int, context_length: int, in_features: int | None = None,
                 out_features: int | None = None, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features or d_model, out_features or d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def flops_per_token(self, context_length: int) -> int:
        return 6 * self.linear.weight.numel()


@block("primitive", "primitive", reference="gelu")
class Activation(BlockModule):
    """gelu (the tanh form GPT-2 uses), silu or relu."""

    def __init__(self, d_model: int, context_length: int, kind: str = "gelu") -> None:
        super().__init__()
        if kind not in ACTIVATIONS:
            raise ValueError(
                f"Activation kind must be one of {', '.join(ACTIVATIONS)}, got {kind!r}")
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gelu":
            return F.gelu(x, approximate="tanh")
        return F.silu(x) if self.kind == "silu" else F.relu(x)

    def flops_per_token(self, context_length: int) -> int:
        return 0


@block("primitive", "primitive", reference="causal_mask")
class CausalMask(BlockModule):
    """Scores (..., T, S): positions a query may not see (key after query) become -inf."""

    mask: torch.Tensor

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()
        self.register_buffer(
            "mask", torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
            persistent=False)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        T, S = scores.shape[-2:]
        return scores.masked_fill(~self.mask[:T, :S], float("-inf"))

    def flops_per_token(self, context_length: int) -> int:
        return 0


@block("primitive", "primitive", reference="naive_causal_attention")
class ScaledDotScores(BlockModule):
    """q @ k.T / sqrt(head_dim) for (B, H, T, D) queries and keys: (B, H, T, S)."""

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    def flops_per_token(self, context_length: int) -> int:
        return 6 * self.d_model * context_length


@block("primitive", "primitive", reference="softmax")
class Softmax(BlockModule):
    """exp / sum(exp) over the last dimension."""

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.softmax(-1)

    def flops_per_token(self, context_length: int) -> int:
        return 0


@block("primitive", "primitive", reference="weighted_sum")
class WeightedSum(BlockModule):
    """Attention weights (B, H, T, S) times values (B, H, S, D): (B, H, T, D)."""

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return weights @ values

    def flops_per_token(self, context_length: int) -> int:
        return 6 * self.d_model * context_length


@block("primitive", "primitive")
class SplitHeads(BlockModule):
    """(B, T, n_heads * D) to (B, n_heads, T, D)."""

    def __init__(self, d_model: int, context_length: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

    def flops_per_token(self, context_length: int) -> int:
        return 0


@block("primitive", "primitive")
class MergeHeads(BlockModule):
    """(B, n_heads, T, D) to (B, T, n_heads * D); the inverse of SplitHeads."""

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).reshape(B, T, H * D)

    def flops_per_token(self, context_length: int) -> int:
        return 0
