import torch
import torch.nn as nn


class MyRoPE(nn.Module):
    def __init__(self, d_model, context_length, base=10000.0):
        super().__init__()
        theta = base ** (-torch.arange(0, d_model, 2).float() / d_model)
        angles = torch.outer(torch.arange(context_length).float(), theta)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        T = x.size(-2)
        cos, sin = self.cos[:T], self.sin[:T]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
