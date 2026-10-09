"""Positional schemes that act on queries and keys inside attention.

All take `(q, k)` of shape (B, heads, T, head_dim) and return them. Attention builds its
`pos` option at `d_model=head_dim`, so for these blocks `d_model` is the head size. A scheme
that works on the attention scores instead (ALiBi) leaves q and k alone and offers
`score_bias(n_heads, T)`, which Attention adds to the scores before the softmax.
"""

from __future__ import annotations

import torch

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule


@block("positional", "composite", reference="naive_rope", features=("rope",))
class RoPE(BlockModule):
    """Rotate each (x[i], x[i + D/2]) pair by position * theta_i, so a query-key dot product
    depends only on how far apart the two positions are."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, d_model: int, context_length: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError(f"RoPE needs an even head size, got {d_model}")
        theta = base ** (-torch.arange(0, d_model, 2).float() / d_model)
        angles = torch.outer(torch.arange(context_length).float(), theta)  # (T, D/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[: x.size(-2)], self.sin[: x.size(-2)]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotate(q), self.rotate(k)

    def flops_per_token(self, context_length: int) -> int:
        return 0  # elementwise, not counted by the 6N + attention formula


@block("positional", "primitive")
class NoPE(BlockModule):
    """No positional signal at all: the causal mask alone tells attention about order."""

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def flops_per_token(self, context_length: int) -> int:
        return 0


def alibi_slopes(n_heads: int) -> list[float]:
    """The geometric sequence of per-head slopes from the ALiBi paper: 2^(-8/n * (i + 1)) for a
    power-of-two head count; otherwise the sequence of the next lower power of two, then every
    other slope of the next higher one."""
    def power_of_two(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start * start**i for i in range(n)]

    if n_heads & (n_heads - 1) == 0:
        return power_of_two(n_heads)
    lower = 1 << (n_heads.bit_length() - 1)
    return power_of_two(lower) + power_of_two(2 * lower)[0::2][: n_heads - lower]


@block("positional", "primitive", reference="naive_alibi_attention", features=("alibi",))
class ALiBi(BlockModule):
    """Attention with Linear Biases: no position is added to q or k; instead each head
    subtracts `slope * distance` from the score of a key that far behind the query, so
    nearby tokens score higher, the more so for steeper heads. The tables are not saved.

    Primitive tier (never locked) until a lesson unlocks it as `block:ALiBi`: the lock table
    comes from the curricula, and the composite tier is exactly the lockable blocks."""

    distance: torch.Tensor

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()
        t = torch.arange(context_length)
        self.register_buffer("distance", (t[:, None] - t[None, :]).clamp(min=0).float(),
                             persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def score_bias(self, n_heads: int, T: int) -> torch.Tensor:
        """(n_heads, T, T): -slope_h * (query position - key position) for keys at or before
        the query (the entries above the diagonal are masked by attention anyway)."""
        slopes = torch.tensor(alibi_slopes(n_heads), device=self.distance.device)
        return -slopes[:, None, None] * self.distance[:T, :T]

    def flops_per_token(self, context_length: int) -> int:
        return 0
