"""Every model component checked against a naive reference implementation."""

import math

import pytest
import torch
import torch.nn.functional as F

from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.models.gpt2 import CausalSelfAttention
from nanoscope.models.modern import Attention, RMSNorm, SwiGLU, apply_rope, rope_tables
from nanoscope.reference.functional import (
    gelu,
    layer_norm,
    naive_causal_attention,
    naive_rope,
    rms_norm,
    swiglu,
)

torch.manual_seed(0)
ATOL = 1e-5


def test_gpt2_attention_matches_naive_loop():
    attn = CausalSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 9, 32)
    q, k, v = attn.qkv(x).split(32, dim=2)
    q, k, v = (t.view(2, 9, 4, 8).transpose(1, 2) for t in (q, k, v))
    expected = attn.proj(naive_causal_attention(q, k, v).transpose(1, 2).reshape(2, 9, 32))
    torch.testing.assert_close(attn(x), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("qk_norm", [True, False])
def test_modern_attention_matches_naive_loop(n_kv_heads, rope, qk_norm):
    attn = Attention(d_model=32, n_heads=4, n_kv_heads=n_kv_heads, rope=rope, qk_norm=qk_norm)
    x = torch.randn(2, 9, 32)
    q = attn.q(x).view(2, 9, 4, 8).transpose(1, 2)
    k = attn.k(x).view(2, 9, n_kv_heads, 8).transpose(1, 2)
    v = attn.v(x).view(2, 9, n_kv_heads, 8).transpose(1, 2)
    if qk_norm:
        q = q / q.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * attn.q_norm.weight
        k = k / k.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * attn.k_norm.weight
    if rope:
        q, k = naive_rope(q), naive_rope(k)
    # query head h reads key/value head h // (n_heads / n_kv_heads)
    kv_for = [h // (4 // n_kv_heads) for h in range(4)]
    k, v = k[:, kv_for], v[:, kv_for]
    expected = attn.proj(naive_causal_attention(q, k, v).transpose(1, 2).reshape(2, 9, 32))
    torch.testing.assert_close(attn(x), expected, atol=ATOL, rtol=0)


def test_rope_matches_complex_rotation():
    x = torch.randn(2, 3, 11, 16)
    cos, sin = rope_tables(16, 11, x.device)
    torch.testing.assert_close(apply_rope(x, cos, sin), naive_rope(x), atol=ATOL, rtol=0)


def test_rope_scores_depend_only_on_relative_position():
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    cos, sin = rope_tables(16, 20, q.device)

    def score(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    torch.testing.assert_close(score(3, 1), score(15, 13), atol=ATOL, rtol=0)


def test_rmsnorm_matches_formula():
    norm = RMSNorm(16)
    norm.weight.data = torch.randn(16)
    x = torch.randn(4, 16)
    torch.testing.assert_close(norm(x), rms_norm(x, norm.weight), atol=ATOL, rtol=0)


def test_layernorm_gelu_and_swiglu_match_their_formulas():
    x = torch.randn(4, 16)
    ln = torch.nn.LayerNorm(16)
    ln.weight.data, ln.bias.data = torch.randn(16), torch.randn(16)
    torch.testing.assert_close(ln(x), layer_norm(x, ln.weight, ln.bias), atol=ATOL, rtol=0)
    torch.testing.assert_close(F.gelu(x, approximate="tanh"), gelu(x), atol=ATOL, rtol=0)
    mlp = SwiGLU(16)
    torch.testing.assert_close(
        mlp(x), swiglu(x, mlp.w1.weight, mlp.w3.weight, mlp.proj.weight), atol=ATOL, rtol=0)


def test_swiglu_matches_gelu_mlp_parameter_count():
    d = 96
    swiglu = sum(p.numel() for p in SwiGLU(d).parameters())
    assert abs(swiglu - 8 * d * d) / (8 * d * d) < 0.02


@pytest.mark.parametrize("model", [
    GPT2(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4),
    Modern(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4),
    Modern(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4, rope=False,
           swiglu=False, rmsnorm=False, qk_norm=False, z_loss=0.0, tie_weights=False),
])
def test_future_tokens_do_not_change_past_logits(model):
    idx = torch.randint(50, (2, 16))
    changed = idx.clone()
    changed[:, 10:] = torch.randint(50, (2, 6))
    out, out_changed = model(idx), model(changed)
    logits = out if isinstance(out, torch.Tensor) else out[0]
    logits_changed = out_changed if isinstance(out_changed, torch.Tensor) else out_changed[0]
    torch.testing.assert_close(logits[:, :10], logits_changed[:, :10], atol=ATOL, rtol=0)
    assert not torch.allclose(logits[:, 10:], logits_changed[:, 10:])


def test_weight_tying():
    gpt2 = GPT2(vocab_size=50)
    assert gpt2.head.weight is gpt2.tok_emb.weight
    modern, untied = Modern(vocab_size=50), Modern(vocab_size=50, tie_weights=False)
    assert modern.head.weight is modern.tok_emb.weight
    assert untied.head.weight is not untied.tok_emb.weight


def test_modern_has_no_biases():
    model = Modern(vocab_size=50, rmsnorm=False)
    assert not [n for n, _ in model.named_parameters() if n.endswith("bias")]


def test_z_loss_is_mean_squared_log_normalizer():
    model = Modern(vocab_size=50, context_length=8, d_model=32, n_layers=1, n_heads=4, z_loss=1e-3)
    logits, aux = model(torch.randint(50, (2, 8)))
    expected = 1e-3 * torch.logsumexp(logits, dim=-1).pow(2).mean()
    torch.testing.assert_close(aux, expected)
    assert isinstance(Modern(vocab_size=50, z_loss=0.0)(torch.randint(50, (1, 4))), torch.Tensor)


@pytest.mark.parametrize("cls", [Bigram, GPT2, Modern])
def test_untrained_loss_is_near_uniform(cls):
    model = cls(vocab_size=512)
    out = model(torch.randint(512, (4, 32)))
    logits = out if isinstance(out, torch.Tensor) else out[0]
    loss = F.cross_entropy(logits.reshape(-1, 512), torch.randint(512, (4 * 32,)))
    assert abs(loss.item() - math.log(512)) < 0.5
