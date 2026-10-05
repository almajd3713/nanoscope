"""Normalisation layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule


@block("norm", "composite", reference="layer_norm")
class LayerNorm(BlockModule, nn.LayerNorm):
    """Subtract the mean, divide by the standard deviation, scale (and shift with a bias)."""

    def __init__(self, d_model: int, context_length: int, bias: bool = True,
                 eps: float = 1e-5) -> None:
        nn.LayerNorm.__init__(self, d_model, eps=eps, bias=bias)

    def flops_per_token(self, context_length: int) -> int:
        return 6 * sum(p.numel() for p in self.parameters())


@block("norm", "composite", reference="rms_norm")
class RMSNorm(BlockModule):
    """LayerNorm without centering or bias: x / rms(x) * weight."""

    def __init__(self, d_model: int, context_length: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight

    def flops_per_token(self, context_length: int) -> int:
        return 6 * self.weight.numel()
