"""Lesson 2: an MLP over a window of tokens.

Write MyMLP, then run:  nanoscope learn check foundations/02-mlp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def window(emb, k):
    """For every position t, the embeddings of tokens t-k+1 .. t (zeros before the start of
    the text), side by side: (batch, time, d) -> (batch, time, k * d)."""
    padded = F.pad(emb, (0, 0, k - 1, 0))  # pad the time axis on the left
    return padded.unfold(1, k, 1).flatten(2).contiguous()


class MyMLP(nn.Module):
    def __init__(self, vocab_size, d_model=32, window_size=8, hidden=256):
        super().__init__()
        self.window_size = window_size
        # TODO 1: an embedding table (vocab_size rows of d_model numbers)
        # TODO 2: a linear layer from window_size * d_model numbers to `hidden`
        # TODO 3: a linear layer, no bias, from `hidden` to vocab_size

    def forward(self, idx):
        # 1. embed idx            -> (batch, time, d_model)
        # 2. window(emb, k)       -> (batch, time, window_size * d_model)
        # 3. hidden layer + torch.relu
        # 4. project to scores    -> (batch, time, vocab_size)
        raise NotImplementedError("embed, window, hidden layer, project")
