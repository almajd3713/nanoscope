import math

import torch
import torch.nn as nn


class OneHead(nn.Module):
    def __init__(self, d_model, context_length):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        T = x.size(1)
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return self.out(weights @ v)
