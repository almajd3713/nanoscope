"""The output head: hidden vectors to one logit per vocabulary entry."""

from __future__ import annotations

import torch.nn as nn

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule


@block("embedding", "primitive", reference="tied_head", features=("tie_weights",))
class Head(BlockModule, nn.Linear):
    """x @ W.T with W of shape (vocab_size, d_model). The Decoder shares W with the token
    embedding when tie_weights is on (`head.weight is tok_emb.weight`)."""

    def __init__(self, d_model: int, context_length: int, vocab_size: int) -> None:
        nn.Linear.__init__(self, d_model, vocab_size, bias=False)

    def flops_per_token(self, context_length: int) -> int:
        return 6 * self.weight.numel()  # tied or not, it is a matmul the model does
