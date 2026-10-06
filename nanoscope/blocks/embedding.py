"""Token and learned-position embeddings. Both are nn.Embedding tables underneath, so their
parameter is called `weight` and initialises exactly as nn.Embedding does."""

from __future__ import annotations

import torch
import torch.nn as nn

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule


@block("embedding", "primitive", reference="embed_one_hot")
class TokenEmbedding(BlockModule, nn.Embedding):
    """Token ids (B, T) to vectors (B, T, d_model): row `id` of a (vocab_size, d_model) table,
    which is the same as a one-hot vector times the table."""

    def __init__(self, d_model: int, context_length: int, vocab_size: int) -> None:
        nn.Embedding.__init__(self, vocab_size, d_model)

    def flops_per_token(self, context_length: int) -> int:
        return 0  # a lookup, no multiply-adds (the tied head counts the matmul)


@block("embedding", "primitive", reference="add_learned_position")
class LearnedPosition(BlockModule, nn.Embedding):
    """Adds a learned vector for each position: x[:, t] + table[t]."""

    def __init__(self, d_model: int, context_length: int) -> None:
        nn.Embedding.__init__(self, context_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.weight[: x.size(1)]

    def flops_per_token(self, context_length: int) -> int:
        return 0
