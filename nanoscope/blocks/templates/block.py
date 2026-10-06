"""A transformer layer with its residual connections fixed and its parts left open."""

from __future__ import annotations

import torch
import torch.nn as nn

from nanoscope.blocks.composite import Composite
from nanoscope.blocks.registry import block


@block("template", "primitive", reference="residual")
class BlockTemplate(Composite):
    """Pre-norm transformer layer: the residual adds are written here, the parts are slots.

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    There are two norm slots (`norm1`, `norm2`) so the two norms get separate weights, as in
    `Block`. Fill them with `RMSNorm()`, an attention block and an MLP block.
    """

    SLOTS = ("norm1", "attn", "norm2", "mlp")
    norm1: nn.Module
    attn: nn.Module
    norm2: nn.Module
    mlp: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
