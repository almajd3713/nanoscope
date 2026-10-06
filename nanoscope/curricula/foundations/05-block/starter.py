"""Lesson 5: the transformer block.

MultiHead is given (it is your lesson 4, finished). Write MyBlock, then run:
    nanoscope learn check foundations/05-block
Passing unlocks nanoscope.blocks.Block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
    """Given: causal multi-head attention."""

    def __init__(self, d_model, context_length, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        head_dim = D // self.n_heads

        def split(t):
            return t.view(B, T, self.n_heads, head_dim).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(B, T, D))


class MyBlock(nn.Module):
    def __init__(self, d_model, context_length, n_heads=4):
        super().__init__()
        # TODO: self.ln1, self.ln2   nn.LayerNorm(d_model)
        # TODO: self.attn            MultiHead(d_model, context_length, n_heads)
        # TODO: self.fc              nn.Linear(d_model, 4 * d_model, bias=False)
        # TODO: self.proj            nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        # x = x + attention of the normalised x
        # x = x + the MLP of the normalised x   (fc, gelu with approximate="tanh", proj)
        raise NotImplementedError("residual attention, then residual MLP")
