"""Lesson 1: a bigram language model.

Fill in the two TODOs and `forward`, then run:  nanoscope learn check foundations/01-bigram
"""

import torch.nn as nn


class MyBigram(nn.Module):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        # TODO 1: an embedding table: token id -> d_model numbers   (nn.Embedding)
        # TODO 2: a linear layer, no bias: d_model numbers -> one score per token
        #         (nn.Linear(d_model, vocab_size, bias=False))

    def forward(self, idx):
        # idx has shape (batch, time) and holds token ids.
        # Return scores of shape (batch, time, vocab_size).
        raise NotImplementedError("embed idx, then project the embeddings to scores")
