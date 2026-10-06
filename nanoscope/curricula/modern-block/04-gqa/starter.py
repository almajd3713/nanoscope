"""Modern lesson 4: grouped-query attention.   Check: nanoscope learn check modern-block/04-gqa

Your multi-head attention from the Foundations path is below. Change it so there are fewer key
and value heads than query heads.
"""

import math

import torch
import torch.nn as nn


class MyGQA(nn.Module):
    def __init__(self, d_model, context_length, n_heads=4, n_kv_heads=2):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        # TODO: k and v should produce only n_kv_heads * head_dim numbers each
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        head_dim = D // self.n_heads

        def split(t, heads):
            return t.view(B, T, heads, head_dim).transpose(1, 2)

        q = split(self.q(x), self.n_heads)
        k = split(self.k(x), self.n_heads)   # TODO: n_kv_heads
        v = split(self.v(x), self.n_heads)   # TODO: n_kv_heads
        # TODO: repeat k and v so each K/V head is shared by n_heads // n_kv_heads query heads
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(B, T, D))
