"""Bigram language model — predicts the next token from the current token only."""

from __future__ import annotations

import torch
import torch.nn as nn


class Bigram(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx)
        return self.head(x)
