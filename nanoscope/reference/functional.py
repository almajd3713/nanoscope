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


def embed_one_hot(ids, table):
    """A lookup is a one-hot row times the table: (..., vocab) @ (vocab, d)."""
    vocab = table.size(0)
    one_hot = (ids.unsqueeze(-1) == torch.arange(vocab)).to(table.dtype)
    return one_hot @ table


def add_learned_position(x, table):
    """Position t of every sequence gets table[t] added; x is (batch, time, d)."""
    out = x.clone()
    for t in range(x.size(1)):
        out[:, t] = x[:, t] + table[t]
    return out


def tied_head(x, embedding_table):
    """Logits from the same matrix the embedding reads: one dot product per vocabulary row."""
    return x @ embedding_table.T


def causal_mask(length):
    """mask[t][s] is True when query t may look at key s, that is when s <= t."""
    mask = torch.zeros(length, length, dtype=torch.bool)
    for t in range(length):
        for s in range(t + 1):
            mask[t, s] = True
    return mask


def softmax(x):
    """exp(x - max) over its sum, along the last dimension."""
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)


def weighted_sum(weights, values):
    """out[t] = sum over s of weights[t][s] * values[s]; (T, S) and (S, D) give (T, D)."""
    out = torch.zeros(weights.size(0), values.size(1), dtype=values.dtype)
    for t in range(weights.size(0)):
        for s in range(weights.size(1)):
            out[t] += weights[t, s] * values[s]
    return out


def repeat_kv_heads(kv, n_heads):
    """Grouped-query attention: query head h reads key/value head h // (n_heads / n_kv).

    kv is (n_kv, ...); the result lists each key/value head for every query head."""
    n_kv = kv.size(0)
    assert n_heads % n_kv == 0, "n_heads must be a multiple of n_kv_heads"
    group = n_heads // n_kv
    return torch.stack([kv[h // group] for h in range(n_heads)])


def qk_norm(x, weight, eps=1e-6):
    """RMS-normalise each query or key vector over its head dimension."""
    return rms_norm(x, weight, eps)


def log_sum_exp(x):
    m = x.max(-1, keepdim=True).values
    return (m + torch.log(torch.exp(x - m).sum(-1, keepdim=True))).squeeze(-1)


def z_loss(logits, coefficient):
    """coefficient * mean(log Z squared), with Z the softmax normaliser of each position."""
    return coefficient * (log_sum_exp(logits) ** 2).mean()
