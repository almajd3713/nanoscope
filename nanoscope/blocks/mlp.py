"""Feed-forward blocks. Weight names follow the shipped models: `fc`/`proj` for the GELU MLP,
`w1`/`w3`/`proj` for SwiGLU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule


@block("mlp", "composite", reference="gelu")
class GELUMLP(BlockModule):
    """gelu(x W1) W2 with hidden size `hidden` (default 4 * d_model)."""

    def __init__(self, d_model: int, context_length: int, hidden: int | None = None,
                 bias: bool = False) -> None:
        super().__init__()
        hidden = hidden or 4 * d_model
        self.fc = nn.Linear(d_model, hidden, bias=bias)
        self.proj = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))

    def flops_per_token(self, context_length: int) -> int:
        return 6 * (self.fc.weight.numel() + self.proj.weight.numel())


@block("mlp", "composite", reference="swiglu", features=("swiglu",))
class SwiGLU(BlockModule):
    """silu(x W1) * (x W3), then W2. The default hidden size is 8/3 * d_model rounded up to
    a multiple of 8, which keeps the parameter count of a 4 * d_model GELU MLP."""

    def __init__(self, d_model: int, context_length: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 8 * ((8 * d_model // 3 + 7) // 8)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.w1(x)) * self.w3(x))

    def flops_per_token(self, context_length: int) -> int:
        return 6 * sum(m.weight.numel() for m in (self.w1, self.w3, self.proj))
