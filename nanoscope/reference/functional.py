"""The formulas, written the way a textbook writes them."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def naive_causal_attention(q, k, v, window=None):
    """Loop over heads and query positions; softmax over keys at or before the query (and,
    with a window, no more than `window` positions back, the query itself included)."""
    B, H, T, D = q.shape
    out = torch.zeros_like(q)
    for h in range(H):
        for t in range(T):
            start = 0 if window is None else max(0, t - window + 1)
            keys = k[:, h, start : t + 1]
            scores = torch.einsum("bd,bsd->bs", q[:, h, t], keys) / math.sqrt(D)
            out[:, h, t] = torch.einsum("bs,bsd->bd", scores.softmax(-1), v[:, h, start : t + 1])
    return out


def alibi_slopes(n_heads):
    """Head h (0-based) subtracts slope_h per position of distance. For a power-of-two head
    count the slopes are 2^(-8 (h+1) / n_heads); otherwise take the sequence for the next lower
    power of two and add every other slope of the next higher power's sequence."""
    def power_of_two(n):
        return [2.0 ** (-8.0 * (h + 1) / n) for h in range(n)]

    if n_heads & (n_heads - 1) == 0:
        return power_of_two(n_heads)
    lower = 1 << (n_heads.bit_length() - 1)
    extra = power_of_two(2 * lower)[0::2]
    return power_of_two(lower) + extra[: n_heads - lower]


def naive_alibi_attention(q, k, v, window=None):
    """Causal attention where the score of key s for query t is q.k / sqrt(D) - slope * (t - s),
    each head with its own slope; no position is added to q or k."""
    B, H, T, D = q.shape
    slopes = alibi_slopes(H)
    out = torch.zeros_like(q)
    for h in range(H):
        for t in range(T):
            start = 0 if window is None else max(0, t - window + 1)
            keys = k[:, h, start : t + 1]
            scores = torch.einsum("bd,bsd->bs", q[:, h, t], keys) / math.sqrt(D)
            scores = scores - slopes[h] * (t - torch.arange(start, t + 1))
            out[:, h, t] = torch.einsum("bs,bsd->bd", scores.softmax(-1), v[:, h, start : t + 1])
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


def naive_moe(x, gate_weight, experts, top_k, aux_coef=0.0):
    """Mixture of experts, one token and one expert at a time. `experts` is a list of
    (w1, w3, w2) SwiGLU weights. A token goes to its top_k experts by router probability, the
    chosen probabilities rescaled to sum to 1. Returns (output, load_balance): load_balance is
    aux_coef * E * sum_e share_e * meanprob_e, where share_e is the fraction of all
    (token, slot) assignments that went to expert e."""
    shape = x.shape
    tokens = x.reshape(-1, shape[-1])
    n, n_experts = tokens.size(0), len(experts)
    out = torch.zeros_like(tokens)
    assigned = [0] * n_experts
    prob_sum = torch.zeros(n_experts)
    for i in range(n):
        logits = gate_weight @ tokens[i]
        probs = (torch.exp(logits - logits.max())) / torch.exp(logits - logits.max()).sum()
        prob_sum = prob_sum + probs
        best = sorted(range(n_experts), key=lambda e: (-float(probs[e].detach()), e))[:top_k]
        total = sum(probs[e] for e in best)
        for e in best:
            w1, w3, w2 = experts[e]
            out[i] = out[i] + (probs[e] / total) * swiglu(tokens[i][None], w1, w3, w2)[0]
            assigned[e] += 1
    share = torch.tensor(assigned, dtype=torch.float) / (n * top_k)
    balance = aux_coef * n_experts * (share * prob_sum / n).sum()
    return out.reshape(shape), balance


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


def naive_attention_head(x, wq, wk, wv, wo):
    """One causal attention head with its four projections. x is (B, T, D), weights are
    (out_features, in_features) like nn.Linear's."""
    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
    mixed = naive_causal_attention(q[:, None], k[:, None], v[:, None])[:, 0]
    return mixed @ wo.T


def naive_multi_head_attention(x, wq, wk, wv, wo, n_heads):
    """n_heads causal heads: head h reads columns h*d .. (h+1)*d of q, k and v (d = D / heads)."""
    B, T, D = x.shape
    d = D // n_heads
    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
    heads = []
    for h in range(n_heads):
        cols = slice(h * d, (h + 1) * d)
        heads.append(naive_causal_attention(q[:, None, :, cols], k[:, None, :, cols],
                                            v[:, None, :, cols])[:, 0])
    return torch.cat(heads, dim=-1) @ wo.T


def naive_gqa(x, wq, wk, wv, wo, n_heads, n_kv_heads):
    """Grouped-query attention: n_heads query heads, n_kv_heads key/value heads; query head
    h reads key/value head h // (n_heads / n_kv_heads), so k and v are repeated for the group."""
    B, T, D = x.shape
    d = D // n_heads
    group = n_heads // n_kv_heads
    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T  # k, v: (B, T, n_kv_heads * d)
    heads = []
    for h in range(n_heads):
        kv = h // group
        cols = slice(kv * d, (kv + 1) * d)
        heads.append(naive_causal_attention(q[:, None, :, h * d:(h + 1) * d], k[:, None, :, cols],
                                            v[:, None, :, cols])[:, 0])
    return torch.cat(heads, dim=-1) @ wo.T


def naive_qk_norm_attention(x, wq, wk, wv, wo, q_norm_weight, k_norm_weight, n_heads):
    """Multi-head attention that RMS-normalises every query and key vector (per head, over
    the head dimension, with its own learned scale) before the dot product."""
    B, T, D = x.shape
    d = D // n_heads
    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
    heads = []
    for h in range(n_heads):
        cols = slice(h * d, (h + 1) * d)
        qh = rms_norm(q[..., cols], q_norm_weight)
        kh = rms_norm(k[..., cols], k_norm_weight)
        heads.append(naive_causal_attention(qh[:, None], kh[:, None], v[:, None, :, cols])[:, 0])
    return torch.cat(heads, dim=-1) @ wo.T


def naive_block(x, ln1_weight, ln1_bias, wq, wk, wv, wo, ln2_weight, ln2_bias, w_fc, w_proj,
                n_heads):
    """A pre-LayerNorm transformer layer: x + attention(ln1(x)), then + mlp(ln2(x)) with a
    GELU MLP (no biases in the attention or the MLP)."""
    x = x + naive_multi_head_attention(layer_norm(x, ln1_weight, ln1_bias), wq, wk, wv, wo,
                                       n_heads)
    return x + gelu(layer_norm(x, ln2_weight, ln2_bias) @ w_fc.T) @ w_proj.T
