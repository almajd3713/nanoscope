"""Composite blocks: blocks made of other blocks, in named slots.

    class PreNormAttention(Composite):
        SLOTS = ("norm", "attn")

        def forward(self, x):
            return self.attn(self.norm(x))

    PreNormAttention(norm=RMSNorm(), attn=Attention(n_heads=4))   # a spec

Every slot is a spec, built once per instance with the instance's `d_model` and context, so
the graph can show and swap what sits in each slot. Lesson templates are Composites with
empty slots to fill.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule, BlockSpec, build_option


class Composite(BlockModule):
    """Base for blocks with named slots. Subclasses set `SLOTS` and write `forward`."""

    SLOTS: tuple[str, ...] = ()

    def __init__(self, d_model: int, context_length: int, **slots: Any) -> None:
        super().__init__()
        unknown, missing = set(slots) - set(self.SLOTS), set(self.SLOTS) - set(slots)
        if unknown or missing:
            raise TypeError(
                f"{type(self).__name__} has the slots {', '.join(self.SLOTS)}"
                + (f"; unknown: {', '.join(sorted(unknown))}" if unknown else "")
                + (f"; missing: {', '.join(sorted(missing))}" if missing else ""))
        self.d_model = d_model
        for name in self.SLOTS:
            setattr(self, name, build_option(slots[name], d_model, context_length))

    def flops_per_token(self, context_length: int) -> int:
        total = 0
        for name in self.SLOTS:
            child = getattr(self, name)
            if hasattr(child, "flops_per_token"):
                total += child.flops_per_token(context_length)
        return total


@block("structure", "primitive", reference="residual")
class Residual(BlockModule):
    """x + inner(x)."""

    def __init__(self, d_model: int, context_length: int, inner: BlockSpec | nn.Module) -> None:
        super().__init__()
        self.inner = build_option(inner, d_model, context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inner(x)

    def flops_per_token(self, context_length: int) -> int:
        inner = getattr(self.inner, "flops_per_token", None)
        return inner(context_length) if inner else 0
