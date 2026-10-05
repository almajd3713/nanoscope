"""The formulas, written the way a textbook writes them."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def naive_causal_attention(q, k, v):
    """Loop over heads and query positions; softmax over keys at or before the query."""
    B, H, T, D = q.shape
    out = torch.zeros_like(q)
    for h in range(H):
        for t in range(T):
            scores = torch.einsum("bd,bsd->bs", q[:, h, t], k[:, h, : t + 1]) / math.sqrt(D)
            out[:, h, t] = torch.einsum("bs,bsd->bd", scores.softmax(-1), v[:, h, : t + 1])
    return out


def naive_rope(x, base=10000.0):
    """Treat (x[i], x[i + D/2]) as a complex number and multiply by e^(i * m * theta_i)."""
    B, H, T, D = x.shape
    half = D // 2
    z = torch.complex(x[..., :half].double(), x[..., half:].double())
    theta = base ** (-torch.arange(0, D, 2).double() / D)
    m = torch.arange(T).double()
    z = z * torch.polar(torch.ones(T, half, dtype=torch.double), torch.outer(m, theta))
    return torch.cat([z.real, z.imag], dim=-1).float()


def rms_norm(x, weight, eps=1e-6):
    """x divided by its root mean square, then scaled per channel."""
    return x / torch.sqrt((x**2).mean(-1, keepdim=True) + eps) * weight


def layer_norm(x, weight, bias, eps=1e-5):
    """Subtract the mean, divide by the standard deviation (biased), scale and shift."""
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def gelu(x):
    """The tanh approximation GPT-2 uses."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def silu(x):
    return x / (1 + torch.exp(-x))


def swiglu(x, w1, w3, w2):
    """silu(x W1) * (x W3), then W2. Weights are laid out (out_features, in_features)."""
    return (silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, non-embedding) parameter counts; shared tensors are counted once."""
    params = {id(p): p for p in model.parameters()}
    emb = {
        id(p) for m in model.modules() if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    total = sum(p.numel() for p in params.values())
    return total, total - sum(params[i].numel() for i in emb if i in params)
