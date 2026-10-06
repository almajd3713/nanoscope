"""The block core: registry, specs, lazy exports. Each block's own tests are added with it."""

import copy
import pickle

import pytest
import torch
import torch.nn as nn

from nanoscope.blocks import registry
from nanoscope.blocks.spec import BlockModule, BlockSpec, build_option


@registry.block("primitive", "primitive", reference="linear")
class Scale(BlockModule):
    """A toy block for the tests: x times a learned vector."""

    def __init__(self, d_model: int, context_length: int, factor: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((d_model,), factor))

    def forward(self, x):
        return x * self.weight


@registry.block("structure")
class Pair(BlockModule):
    def __init__(self, d_model: int, context_length: int, first, second=None) -> None:
        super().__init__()
        self.first = build_option(first, d_model, context_length)
        self.second = build_option(second, d_model, context_length)


def test_registry_records_blocks_and_refuses_conflicts():
    info = registry.get_info("Scale")
    assert (info.family, info.tier, info.reference) == ("primitive", "primitive", "linear")
    assert info.module == Scale.__module__
    assert info in registry.all_blocks()
    with pytest.raises(KeyError, match="unknown block 'Nope'"):
        registry.get_info("Nope")
    with pytest.raises(ValueError, match="tier"):
        registry._register(registry.BlockInfo("X", "norm", "huge", "m"))
    with pytest.raises(ValueError, match="already registered differently"):
        registry._register(registry.BlockInfo("Scale", "norm", "composite", "m"))


def test_lazy_exports_import_on_first_use():
    import nanoscope.blocks as blocks

    assert blocks.BlockSpec is BlockSpec
    assert "BlockModule" in dir(blocks)
    with pytest.raises(AttributeError, match="no block 'Missing'"):
        getattr(blocks, "Missing")  # noqa: B009


def test_spec_calling_with_options_only_gives_a_spec_without_tensors():
    spec = Scale(factor=2.0)
    assert isinstance(spec, BlockSpec) and spec.cls is Scale
    assert spec.options == {"factor": 2.0}
    assert isinstance(Scale(), BlockSpec)
    module = spec.build(d_model=4, context_length=8)
    assert isinstance(module, Scale) and module.weight.tolist() == [2.0] * 4


def test_spec_checks_its_options_at_once():
    with pytest.raises(TypeError, match="no option factr"):
        Scale(factr=2.0)
    with pytest.raises(TypeError, match="needs: first"):
        Pair()
    with pytest.raises(TypeError, match="by keyword"):
        Scale(2.0)


def test_spec_builds_independent_params():
    spec = Pair(first=Scale(factor=1.0), second=Scale(factor=1.0))
    a, b = spec.build(4, 8), spec.build(4, 8)
    assert a.first.weight is not b.first.weight
    assert a.first.weight is not a.second.weight
    a.first.weight.data += 1
    assert b.first.weight.tolist() == [1.0] * 4
    assert a.second.weight.tolist() == [1.0] * 4


def test_spec_to_dict_nests_specs_for_describe_and_the_graph():
    spec = Pair(first=Scale(factor=2.0), second=Scale())
    assert spec.to_dict() == {"block": "Pair", "args": {
        "first": {"block": "Scale", "args": {"factor": 2.0}},
        "second": {"block": "Scale", "args": {}}}}


def test_built_blocks_survive_deepcopy_and_pickle():
    module = Scale(factor=3.0).build(4, 8)
    for clone in (copy.deepcopy(module), pickle.loads(pickle.dumps(module))):
        assert isinstance(clone, Scale) and clone.weight.tolist() == [3.0] * 4
        assert clone.weight is not module.weight


def test_library_imports_submodules():
    """Nothing in nanoscope/ but blocks/__init__.py imports the blocks package root, so
    gating that root can never change shipped code."""
    import ast
    from pathlib import Path

    import nanoscope

    root = Path(nanoscope.__file__).parent
    for file in root.rglob("*.py"):
        if file == root / "blocks" / "__init__.py":
            continue
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "nanoscope.blocks":
                raise AssertionError(f"{file.relative_to(root)} imports the blocks package root")
            if isinstance(node, ast.ImportFrom) and node.module == "nanoscope" and any(
                    a.name == "blocks" for a in node.names):
                raise AssertionError(f"{file.relative_to(root)} imports the blocks package root")
            if isinstance(node, ast.Import) and any(
                    a.name == "nanoscope.blocks" for a in node.names):
                raise AssertionError(f"{file.relative_to(root)} imports the blocks package root")


def test_shipped_marks_library_models():
    from nanoscope.models import GPT2, Bigram, Modern

    for cls in (Bigram, GPT2, Modern):
        assert cls.__nanoscope_shipped__ is True and registry.is_shipped(cls)

    class Sub(GPT2):
        pass

    assert registry.is_shipped(Sub)  # subclassing a shipped model counts as running it


# ---- primitives -------------------------------------------------------------------------

from nanoscope.reference import functional as R  # noqa: E402

ATOL = 1e-5


def test_linear_matches_matmul_and_counts_flops():
    from nanoscope.blocks.primitives import Linear

    layer = Linear(out_features=6).build(4, 8)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(layer(x), x @ layer.linear.weight.T, atol=ATOL, rtol=0)
    assert layer.linear.bias is None and layer.flops_per_token(8) == 6 * 4 * 6
    assert Linear(bias=True).build(4, 8)(x).shape == (2, 3, 4)


def test_activation_matches_formulas():
    from nanoscope.blocks.primitives import Activation

    x = torch.randn(5, 7)
    build = lambda kind: Activation(kind=kind).build(7, 4)  # noqa: E731
    torch.testing.assert_close(build("gelu")(x), R.gelu(x), atol=ATOL, rtol=0)
    torch.testing.assert_close(build("silu")(x), R.silu(x), atol=ATOL, rtol=0)
    assert torch.equal(build("relu")(x), x.clamp(min=0))
    with pytest.raises(ValueError, match="gelu, silu, relu"):
        Activation(kind="tanh").build(7, 4)


def test_primitives_attention_parts():
    from nanoscope.blocks.primitives import (
        CausalMask,
        MergeHeads,
        ScaledDotScores,
        Softmax,
        SplitHeads,
        WeightedSum,
    )

    B, H, T, D = 2, 3, 5, 4
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    scores = ScaledDotScores().build(H * D, T)(q, k)
    assert scores.shape == (B, H, T, T)
    torch.testing.assert_close(scores[0, 1, 2, 3], (q[0, 1, 2] * k[0, 1, 3]).sum() / 2)
    masked = CausalMask().build(H * D, T)(scores)
    assert (masked.isinf() == ~R.causal_mask(T)).all()
    weights = Softmax().build(H * D, T)(masked)
    torch.testing.assert_close(weights, R.softmax(masked), atol=ATOL, rtol=0)
    out = WeightedSum().build(H * D, T)(weights, v)
    torch.testing.assert_close(out, R.naive_causal_attention(q, k, v), atol=ATOL, rtol=0)
    x = torch.randn(B, T, H * D)
    heads = SplitHeads(n_heads=H).build(H * D, T)(x)
    assert heads.shape == (B, H, T, D)
    assert torch.equal(MergeHeads().build(H * D, T)(heads), x)
    # the two attention matmuls together are the 12 * d_model * context of the PaLM formula
    both = (ScaledDotScores().build(H * D, T).flops_per_token(T)
            + WeightedSum().build(H * D, T).flops_per_token(T))
    assert both == 12 * H * D * T


def test_causal_mask_handles_shorter_sequences():
    from nanoscope.blocks.primitives import CausalMask

    mask = CausalMask().build(8, 16)
    out = mask(torch.zeros(1, 1, 4, 4))
    assert (out.isinf() == ~R.causal_mask(4)).all()


# ---- composite and residual ---------------------------------------------------------------


def test_composite_named_slots_build_and_count_flops():
    from nanoscope.blocks.composite import Composite
    from nanoscope.blocks.primitives import Activation, Linear

    class Feed(Composite):
        SLOTS = ("up", "act", "down")

        def forward(self, x):
            return self.down(self.act(self.up(x)))

    spec = Feed(up=Linear(out_features=16), act=Activation(kind="relu"),
                down=Linear(in_features=16))
    assert spec.to_dict()["args"]["act"] == {"block": "Activation", "args": {"kind": "relu"}}
    feed = spec.build(4, 8)
    assert [n for n, _ in feed.named_children()] == ["up", "act", "down"]
    x = torch.randn(2, 3, 4)
    expected = feed.down.linear(feed.up.linear(x).relu())
    torch.testing.assert_close(feed(x), expected, atol=1e-6, rtol=0)
    assert feed.flops_per_token(8) == 6 * 4 * 16 + 6 * 16 * 4
    other = spec.build(4, 8)
    assert other.up.linear.weight is not feed.up.linear.weight
    with pytest.raises(TypeError, match="needs: up, act, down"):
        Feed()
    with pytest.raises(TypeError, match="no option extra"):
        Feed(up=Linear(), act=Linear(), down=Linear(), extra=Linear())


def test_residual_adds_its_input():
    from nanoscope.blocks.composite import Residual
    from nanoscope.blocks.primitives import Linear

    block = Residual(inner=Linear()).build(4, 8)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(block(x), x + x @ block.inner.linear.weight.T, atol=1e-6, rtol=0)
    assert block.flops_per_token(8) == 6 * 4 * 4
    with pytest.raises(TypeError, match="needs: inner"):
        Residual()


# ---- embedding, head, norms -----------------------------------------------------------------


def test_embedding_matches_one_hot_matmul():
    from nanoscope.blocks.embedding import LearnedPosition, TokenEmbedding

    emb = TokenEmbedding(vocab_size=11).build(6, 5)
    ids = torch.randint(11, (3, 5))
    torch.testing.assert_close(emb(ids), R.embed_one_hot(ids, emb.weight), atol=ATOL, rtol=0)
    assert isinstance(emb, nn.Embedding) and list(emb.state_dict()) == ["weight"]
    pos = LearnedPosition().build(6, 5)
    x = torch.randn(3, 4, 6)  # shorter than the context: only the first 4 positions are used
    torch.testing.assert_close(pos(x), R.add_learned_position(x, pos.weight), atol=ATOL, rtol=0)
    assert emb.flops_per_token(5) == 0 and pos.flops_per_token(5) == 0


def test_head_tied_or_untied():
    from nanoscope.blocks.embedding import TokenEmbedding
    from nanoscope.blocks.head import Head

    emb, head = TokenEmbedding(vocab_size=11).build(6, 5), Head(vocab_size=11).build(6, 5)
    x = torch.randn(2, 5, 6)
    assert head.weight is not emb.weight and head.bias is None
    head.weight = emb.weight  # what Decoder does for tie_weights=True
    torch.testing.assert_close(head(x), R.tied_head(x, emb.weight), atol=ATOL, rtol=0)
    assert head.flops_per_token(5) == 6 * 11 * 6


def test_norm_layers_match_their_formulas():
    from nanoscope.blocks.norm import LayerNorm, RMSNorm

    x = torch.randn(4, 16)
    ln = LayerNorm().build(16, 8)
    ln.weight.data, ln.bias.data = torch.randn(16), torch.randn(16)
    torch.testing.assert_close(ln(x), R.layer_norm(x, ln.weight, ln.bias), atol=ATOL, rtol=0)
    assert ln.flops_per_token(8) == 6 * 32
    nobias = LayerNorm(bias=False).build(16, 8)
    assert nobias.bias is None and nobias.flops_per_token(8) == 6 * 16
    rms = RMSNorm().build(16, 8)
    rms.weight.data = torch.randn(16)
    torch.testing.assert_close(rms(x), R.rms_norm(x, rms.weight), atol=ATOL, rtol=0)
    assert rms.flops_per_token(8) == 6 * 16


def test_rope_matches_complex_rotation_and_is_relative():
    from nanoscope.blocks.positional import RoPE

    rope = RoPE().build(8, 16)
    q, k = torch.randn(2, 3, 16, 8), torch.randn(2, 3, 16, 8)
    rq, rk = rope(q, k)
    torch.testing.assert_close(rq, R.naive_rope(q), atol=ATOL, rtol=0)
    torch.testing.assert_close(rk, R.naive_rope(k), atol=ATOL, rtol=0)
    # the score between positions m and n depends only on n - m: put the same vectors at
    # shifted positions and the dot product does not change
    v, w = torch.randn(8), torch.randn(8)

    def score(m, n):
        x = torch.zeros(1, 1, 16, 8)
        y = torch.zeros(1, 1, 16, 8)
        x[..., m, :], y[..., n, :] = v, w
        a, b = rope(x, y)
        return (a[..., m, :] * b[..., n, :]).sum()

    torch.testing.assert_close(score(2, 5), score(7, 10), atol=1e-5, rtol=0)
    assert not torch.allclose(score(2, 5), score(2, 6), atol=1e-5)
    assert rope(q[..., :5, :], k[..., :5, :])[0].shape == (2, 3, 5, 8)
    assert not list(rope.state_dict())  # tables are not saved
    with pytest.raises(ValueError, match="even"):
        RoPE().build(7, 16)


def test_nope_changes_nothing():
    from nanoscope.blocks.positional import NoPE

    nope = NoPE().build(8, 16)
    q, k = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    assert nope(q, k)[0] is q and nope(q, k)[1] is k
    assert nope.flops_per_token(16) == 0


@pytest.mark.parametrize("n_kv", [1, 2, 4])
@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("window", [None, 3])
def test_attention_matches_loop_reference(n_kv, use_rope, qk_norm, window):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.positional import RoPE

    torch.manual_seed(0)
    d, heads, T = 16, 4, 7
    attn = Attention(n_heads=heads, n_kv_heads=n_kv, qk_norm=qk_norm, window=window,
                     pos=RoPE() if use_rope else None).build(d, 8)
    if qk_norm:
        attn.q_norm.weight.data, attn.k_norm.weight.data = torch.randn(4), torch.randn(4)
    x = torch.randn(2, T, d)
    hd = d // heads

    def split(y, n):
        return y.view(2, T, n, hd).transpose(1, 2)
    q, k, v = split(attn.q(x), heads), split(attn.k(x), n_kv), split(attn.v(x), n_kv)
    if qk_norm:
        q, k = R.qk_norm(q, attn.q_norm.weight), R.qk_norm(k, attn.k_norm.weight)
    if use_rope:
        q, k = R.naive_rope(q), R.naive_rope(k)
    k = torch.stack([R.repeat_kv_heads(k[b], heads) for b in range(2)])
    v = torch.stack([R.repeat_kv_heads(v[b], heads) for b in range(2)])
    y = R.naive_causal_attention(q, k, v, window=window)
    expected = attn.proj(y.transpose(1, 2).reshape(2, T, d))
    torch.testing.assert_close(attn(x), expected, atol=ATOL, rtol=0)


def test_attention_options_flops_and_errors():
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.positional import RoPE

    plain = Attention(n_heads=4).build(16, 8)
    assert plain.n_kv_heads == 4 and list(plain.state_dict()) == [
        "q.weight", "k.weight", "v.weight", "proj.weight"]
    assert plain.flops_per_token(8) == 6 * 4 * 256 + 12 * 16 * 8
    windowed = Attention(n_heads=4, window=2).build(16, 8)
    assert windowed.flops_per_token(8) == 6 * 4 * 256 + 12 * 16 * 2
    gqa = Attention(n_heads=4, n_kv_heads=1, pos=RoPE(base=500.0), qk_norm=True)
    assert gqa.to_dict() == {"block": "Attention", "args": {
        "n_heads": 4, "n_kv_heads": 1, "pos": {"block": "RoPE", "args": {"base": 500.0}},
        "qk_norm": True}}
    built = gqa.build(16, 8)
    assert built.k.out_features == 4 and built.q_norm.weight.shape == (4,)
    with pytest.raises(ValueError, match="n_kv_heads"):
        Attention(n_heads=4, n_kv_heads=3).build(16, 8)
    with pytest.raises(ValueError, match="window"):
        Attention(n_heads=4, window=0).build(16, 8)
    x = torch.randn(1, 3, 16)  # shorter than the context
    assert Attention(n_heads=4, window=2).build(16, 8)(x).shape == x.shape
