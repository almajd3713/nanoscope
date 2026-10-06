import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHead(d_model, context_length, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))
