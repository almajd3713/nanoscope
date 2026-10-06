"""Lesson 3: one causal attention head, from scratch.

Write OneHead, then run:  nanoscope learn check foundations/03-attention-head
Do not use F.scaled_dot_product_attention or nn.MultiheadAttention: build it yourself.
"""

import math

import torch
import torch.nn as nn


class OneHead(nn.Module):
    def __init__(self, d_model, context_length):
        super().__init__()
        # TODO: four linear layers, d_model -> d_model, no bias, named exactly:
        #   self.q, self.k, self.v, self.out
        # (the check looks for the weights q.weight, k.weight, v.weight and out.weight)

    def forward(self, x):
        # x: (batch, time, d_model)
        # 1. q, k, v = self.q(x), self.k(x), self.v(x)
        # 2. scores = q @ k.transpose(-2, -1) / math.sqrt(d_model)         (batch, time, time)
        # 3. hide the future: scores[..., t, s] = -inf for s > t
        #      hint: torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), masked_fill
        # 4. weights = softmax over the last dimension
        # 5. return self.out(weights @ v)
        raise NotImplementedError("attention: scores, mask, softmax, mix")
