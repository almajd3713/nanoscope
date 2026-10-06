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


def init_weights(model, n_layers):
    """GPT-2 initialisation: matrices ~ N(0, 0.02), the projections that write into the
    residual stream scaled down by sqrt(2 * n_layers), biases zero."""
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(p)
        elif p.ndim == 2:
            std = 0.02 / math.sqrt(2 * n_layers) if name.endswith(("out.weight", "proj.weight")) \
                else 0.02
            nn.init.normal_(p, mean=0.0, std=std)


class MyGPT2(nn.Module):
    def __init__(self, vocab_size, context_length=256, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            MyBlock(d_model, context_length, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tie the output matrix to the embedding
        init_weights(self, n_layers)

    def forward(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
