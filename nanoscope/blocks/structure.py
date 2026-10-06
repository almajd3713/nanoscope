"""A transformer layer: attention and an MLP, each wrapped in a residual and a norm.

Submodule names follow the shipped Modern model: `norm1`, `attn`, `norm2`, `mlp`.
"""

from __future__ import annotations

import torch

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule, BlockSpec, build_option

ORDERS = ("pre", "post")


@block("structure", "composite", reference="residual", features=("post_norm",))
class Block(BlockModule):
    """pre:  x = x + attn(norm1(x)); x = x + mlp(norm2(x))
    post: x = norm1(x + attn(x));   x = norm2(x + mlp(x))

    `norm` is one spec, built twice so the two norms have their own weights."""

    def __init__(self, d_model: int, context_length: int, norm: BlockSpec, attn: BlockSpec,
                 mlp: BlockSpec, order: str = "pre") -> None:
        super().__init__()
        if order not in ORDERS:
            raise ValueError(f"Block order must be one of {', '.join(ORDERS)}, got {order!r}")
        self.order = order
        self.norm1 = build_option(norm, d_model, context_length)
        self.attn = build_option(attn, d_model, context_length)
        self.norm2 = build_option(norm, d_model, context_length)
        self.mlp = build_option(mlp, d_model, context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "pre":
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.mlp(x))

    def flops_per_token(self, context_length: int) -> int:
        return sum(m.flops_per_token(context_length)
                   for m in (self.norm1, self.attn, self.norm2, self.mlp))
