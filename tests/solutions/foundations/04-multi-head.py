import math

import torch
import torch.nn as nn


class MultiHead(nn.Module):
    def __init__(self, d_model, context_length, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide by n_heads"
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
        merged = (weights @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(merged)
