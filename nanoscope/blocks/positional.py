"""Positional schemes that act on queries and keys inside attention.

Both take `(q, k)` of shape (B, heads, T, head_dim) and return them. Attention builds its
`pos` option at `d_model=head_dim`, so for these blocks `d_model` is the head size.
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


@block("positional", "composite")
class NoPE(BlockModule):
    """No positional signal at all: the causal mask alone tells attention about order."""

    def __init__(self, d_model: int, context_length: int) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def flops_per_token(self, context_length: int) -> int:
        return 0
