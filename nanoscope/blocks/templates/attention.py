"""One attention head as eight steps, each a slot."""

from __future__ import annotations

import torch
import torch.nn as nn

from nanoscope.blocks.composite import Composite
from nanoscope.blocks.registry import block


@block("template", "primitive", reference="naive_causal_attention")
class AttentionTemplate(Composite):
    """Single-head causal attention on (B, T, d_model):

        q, k, v = q(x), k(x), v(x)             three projections
        s = scores(q, k)                       how much each query likes each key
        s = mask(s)                            hide the future
        w = normalize(s)                       turn scores into weights that sum to 1
        y = mix(w, v)                          the weighted sum of the values
        return out(y)                          project back

    Fill it with `Linear()`, `ScaledDotScores()`, `CausalMask()`, `Softmax()` and
    `WeightedSum()` and it matches `naive_causal_attention`.
    """

    SLOTS = ("q", "k", "v", "scores", "mask", "normalize", "mix", "out")
    q: nn.Module
    k: nn.Module
    v: nn.Module
    scores: nn.Module
    mask: nn.Module
    normalize: nn.Module
    mix: nn.Module
    out: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        weights = self.normalize(self.mask(self.scores(q, k)))
        return self.out(self.mix(weights, v))
