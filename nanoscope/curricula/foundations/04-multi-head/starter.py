"""Lesson 4: multi-head attention.

Write MultiHead, then run:  nanoscope learn check foundations/04-multi-head
Passing unlocks nanoscope.blocks.Attention.
"""

import math

import torch
import torch.nn as nn


class MultiHead(nn.Module):
    def __init__(self, d_model, context_length, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide by n_heads"
        self.n_heads = n_heads
        # TODO: four linear layers d_model -> d_model, no bias:
        #   self.q, self.k, self.v, self.out

    def forward(self, x):
        B, T, D = x.shape
        head_dim = D // self.n_heads
        # 1. q, k, v from the three layers
        # 2. split heads: (B, T, D) -> (B, n_heads, T, head_dim)
        #      t.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        # 3. scores = q @ k.transpose(-2, -1) / sqrt(head_dim)   (B, n_heads, T, T)
        # 4. mask the future, softmax, weights @ v               (B, n_heads, T, head_dim)
        # 5. merge heads: transpose(1, 2).reshape(B, T, D), then self.out
        raise NotImplementedError("multi-head attention")
