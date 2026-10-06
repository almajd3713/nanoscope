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


def test_mlp_matches_formulas_and_parameter_parity():
    from nanoscope.blocks.mlp import GELUMLP, SwiGLU

    x = torch.randn(2, 5, 16)
    swi = SwiGLU().build(16, 8)
    torch.testing.assert_close(
        swi(x), R.swiglu(x, swi.w1.weight, swi.w3.weight, swi.proj.weight), atol=ATOL, rtol=0)
    gelu = GELUMLP(bias=True).build(16, 8)
    expected = R.gelu(x @ gelu.fc.weight.T + gelu.fc.bias) @ gelu.proj.weight.T + gelu.proj.bias
    torch.testing.assert_close(gelu(x), expected, atol=ATOL, rtol=0)
    assert list(GELUMLP().build(16, 8).state_dict()) == ["fc.weight", "proj.weight"]
    assert list(swi.state_dict()) == ["w1.weight", "w3.weight", "proj.weight"]
    # d=96: 4d GELU and the default SwiGLU hidden hold the same number of weights
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(SwiGLU().build(96, 8)) == n(GELUMLP().build(96, 8))
    assert SwiGLU().build(16, 8).flops_per_token(8) == 6 * 3 * 16 * 48
    assert GELUMLP().build(16, 8).flops_per_token(8) == 6 * 2 * 16 * 64
    assert SwiGLU(hidden=10).build(16, 8).w1.out_features == 10


@pytest.mark.parametrize("order", ["pre", "post"])
def test_block_reference_composition(order):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.mlp import SwiGLU
    from nanoscope.blocks.norm import RMSNorm
    from nanoscope.blocks.structure import Block

    torch.manual_seed(0)
    spec = Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=SwiGLU(), order=order)
    blk = spec.build(16, 8)
    assert blk.norm1 is not blk.norm2 and blk.norm1.weight is not blk.norm2.weight
    blk.norm1.weight.data, blk.norm2.weight.data = torch.randn(16), torch.randn(16)
    x = torch.randn(2, 6, 16)
    if order == "pre":
        h = x + blk.attn(R.rms_norm(x, blk.norm1.weight))
        expected = h + blk.mlp(R.rms_norm(h, blk.norm2.weight))
    else:
        h = R.rms_norm(x + blk.attn(x), blk.norm1.weight)
        expected = R.rms_norm(h + blk.mlp(h), blk.norm2.weight)
    torch.testing.assert_close(blk(x), expected, atol=ATOL, rtol=0)
    assert blk.flops_per_token(8) == sum(
        m.flops_per_token(8) for m in (blk.norm1, blk.attn, blk.norm2, blk.mlp))
    assert list(blk.state_dict())[:2] == ["norm1.weight", "attn.q.weight"]
    assert spec.to_dict()["args"]["order"] == order
    with pytest.raises(ValueError, match="order"):
        Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=SwiGLU(), order="mid").build(16, 8)


def _decoder(**kw):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.mlp import SwiGLU
    from nanoscope.blocks.norm import RMSNorm
    from nanoscope.blocks.positional import RoPE
    from nanoscope.blocks.structure import Block, Decoder

    layer = Block(norm=RMSNorm(), attn=Attention(n_heads=2, n_kv_heads=1, pos=RoPE()),
                  mlp=SwiGLU())
    args = dict(vocab_size=50, context_length=16, d_model=16, n_layers=2, block=layer,
                final_norm=RMSNorm())
    return Decoder(**{**args, **kw})


def test_decoder_is_causal_with_near_uniform_untrained_loss():
    import math

    torch.manual_seed(0)
    model = _decoder()
    idx = torch.randint(0, 50, (4, 12))
    changed = idx.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 50
    a, b = model(idx), model(changed)
    torch.testing.assert_close(a[:, :8], b[:, :8], atol=ATOL, rtol=0)
    assert not torch.allclose(a[:, 8:], b[:, 8:])
    loss = nn.functional.cross_entropy(a.reshape(-1, 50), torch.randint(0, 50, (48,)))
    assert abs(loss.item() - math.log(50)) < 0.1


def test_decoder_options_init_and_aux_loss():
    model = _decoder(z_loss=1e-4)
    logits, aux = model(torch.randint(0, 50, (2, 5)))
    assert logits.shape == (2, 5, 50) and aux.ndim == 0 and aux > 0
    assert model.head.weight is model.tok_emb.weight and model.pos_emb is None
    assert model.blocks[0] is not model.blocks[1]
    assert model.blocks[0].attn.q.weight is not model.blocks[1].attn.q.weight
    assert abs(model.blocks[0].attn.q.weight.std().item() - 0.02) < 0.006
    assert abs(model.blocks[0].mlp.proj.weight.std().item() - 0.01) < 0.004  # 0.02/sqrt(4)
    untied = _decoder(tie_weights=False)
    assert untied.head.weight is not untied.tok_emb.weight
    assert untied(torch.zeros(1, 3).long()).ndim == 3
    from nanoscope.blocks.embedding import LearnedPosition
    learned = _decoder(pos_emb=LearnedPosition())
    assert learned.pos_emb.weight.shape == (16, 16)
    with pytest.raises(AssertionError, match="context_length"):
        model(torch.zeros(1, 17).long())


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("learned_pos", [False, True])
def test_decoder_flops_equal_the_palm_formula(tie, learned_pos):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.embedding import LearnedPosition
    from nanoscope.blocks.mlp import SwiGLU
    from nanoscope.blocks.norm import RMSNorm
    from nanoscope.blocks.structure import Block, Decoder
    from nanoscope.reference.modern_ref import ModernRef

    layer = Block(norm=RMSNorm(), attn=Attention(n_heads=4, n_kv_heads=2, qk_norm=True),
                  mlp=SwiGLU())
    model = Decoder(vocab_size=100, context_length=32, d_model=32, n_layers=3, block=layer,
                    final_norm=RMSNorm(), tie_weights=tie,
                    pos_emb=LearnedPosition() if learned_pos else None)
    ref = ModernRef(vocab_size=100, context_length=32, d_model=32, n_layers=3, n_heads=4,
                    n_kv_heads=2, rope=not learned_pos, tie_weights=tie)
    for ctx in (8, 32):
        assert model.flops_per_token(ctx) == ref.flops_per_token(ctx)


def test_decoder_layer_pattern_repeats():
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.mlp import SwiGLU
    from nanoscope.blocks.norm import RMSNorm
    from nanoscope.blocks.structure import Block, Decoder

    def layer(window):
        return Block(norm=RMSNorm(), attn=Attention(n_heads=2, window=window), mlp=SwiGLU())
    sliding, global_ = layer(4), layer(None)
    model = Decoder(vocab_size=50, context_length=16, d_model=16, n_layers=5,
                    pattern=[sliding, sliding, global_])
    assert [b.attn.window for b in model.blocks] == [4, 4, None, 4, 4]
    assert model.blocks[0].attn.q.weight is not model.blocks[1].attn.q.weight
    assert model(torch.zeros(1, 6).long()).shape == (1, 6, 50)
    # sliding layers do less attention work than global ones
    assert model.blocks[0].flops_per_token(16) < model.blocks[2].flops_per_token(16)
    with pytest.raises(TypeError, match="exactly one"):
        Decoder(vocab_size=50, context_length=16, d_model=16, n_layers=2)
    with pytest.raises(TypeError, match="exactly one"):
        Decoder(vocab_size=50, context_length=16, d_model=16, n_layers=2, block=sliding,
                pattern=[global_])
    with pytest.raises(TypeError, match="exactly one"):
        Decoder(vocab_size=50, context_length=16, d_model=16, n_layers=2, pattern=[])


@pytest.mark.parametrize("window", [None, 3])
def test_attention_weights_are_the_softmax_forward_applies(window):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.positional import RoPE

    attn = Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True,
                     window=window).build(16, 8)
    x = torch.randn(2, 6, 16)
    weights = attn.attention_weights(x)
    assert weights.shape == (2, 4, 6, 6)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 6), atol=ATOL, rtol=0)
    assert torch.all(weights.triu(1) == 0)  # causal
    if window:
        assert torch.all(weights.tril(-window) == 0)  # nothing further back than the window
    _, _, v = attn._qkv(x)
    expected = attn.proj((weights @ v).transpose(1, 2).reshape(2, 6, 16))
    torch.testing.assert_close(attn(x), expected, atol=ATOL, rtol=0)


def test_register_block_makes_a_user_module_composable():
    from nanoscope.blocks import register_block
    from nanoscope.blocks.registry import reference_for
    from nanoscope.blocks.structure import Block, Decoder

    def naive_gate(x, weight):
        return x * torch.sigmoid(weight)

    @register_block(reference=naive_gate, family="mlp")
    class Gate(nn.Module):
        """x times sigmoid of a learned vector."""

        def __init__(self, d_model, context_length, scale=1.0):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(d_model))
            self.scale = scale

        def forward(self, x):
            return x * torch.sigmoid(self.weight) * self.scale

    info = registry.get_info("Gate")
    assert (info.family, info.tier, info.user) == ("mlp", "composite", True)
    assert info.reference.endswith("naive_gate") and reference_for("Gate") is naive_gate
    assert registry.get_info("RMSNorm").user is False
    spec = Gate(scale=2.0)  # options only: a spec, like every block
    assert isinstance(spec, BlockSpec) and spec.to_dict() == {
        "block": "Gate", "args": {"scale": 2.0}}
    gate = spec.build(8, 4)
    assert isinstance(gate, nn.Module) and gate.scale == 2.0
    torch.testing.assert_close(gate(torch.ones(2, 8)), torch.full((2, 8), 1.0))  # 2 * 0.5
    assert copy.deepcopy(gate).scale == 2.0
    # it composes: a Decoder built from it trains-shaped output
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.norm import RMSNorm
    model = Decoder(vocab_size=20, context_length=8, d_model=16, n_layers=2,
                    block=Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=Gate()))
    assert model(torch.zeros(1, 4).long()).shape == (1, 4, 20)

    @register_block  # bare form, no reference
    class Plain(BlockModule):
        def __init__(self, d_model, context_length):
            super().__init__()
    assert registry.get_info("Plain").reference is None and reference_for("Plain") is None

    with pytest.raises(TypeError, match="must take d_model and context_length"):
        @register_block
        class Bad(nn.Module):
            def __init__(self, width):
                super().__init__()


def test_discover_without_import(tmp_path):
    from nanoscope.blocks.discover import discover

    (tmp_path / "boom.py").write_text(
        "raise RuntimeError('this file must never be imported')\n"
        "from nanoscope.blocks import Decoder, Composite, register_block\n"
        "import nanoscope.blocks as nb\n\n"
        "def naive(x):\n    return x\n\n"
        "@register_block(reference=naive, family='mlp')\n"
        "class Gate(nn.Module):\n"
        "    '''A gate.'''\n"
        "    def __init__(self, d_model, context_length, hidden: int = 64, act=None):\n"
        "        raise RuntimeError('never run')\n\n"
        "@nb.register_block\n"
        "class Plain(nn.Module):\n"
        "    def __init__(self, d_model, context_length, *, bias: bool):\n        pass\n\n"
        "class MyLM(Decoder):\n"
        "    def __init__(self, vocab_size):\n"
        "        super().__init__(vocab_size, 8, d_model=16, n_layers=1, block=None)\n\n"
        "class Slots(Composite):\n    SLOTS = ('a',)\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "typo.py").write_text("def (:\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "skipped.py").write_text("@register_block\nclass Nope: pass\n")
    found = discover(tmp_path)
    blocks = {b["name"]: b for b in found["blocks"]}
    assert set(blocks) == {"Gate", "Plain"}
    gate = blocks["Gate"]
    assert (gate["family"], gate["reference"], gate["doc"]) == ("mlp", "naive", "A gate.")
    assert [o["name"] for o in gate["options"]] == ["hidden", "act"]
    assert gate["options"][0] == {"name": "hidden", "annotation": "int", "required": False,
                                  "default": 64}
    assert blocks["Plain"]["family"] == "custom" and blocks["Plain"]["reference"] is None
    assert blocks["Plain"]["options"][0]["required"] is True
    assert [(m["name"], m["kind"]) for m in found["models"]] == [
        ("MyLM", "decoder"), ("Slots", "composite")]
    assert [e["file"].endswith("typo.py") for e in found["errors"]] == [True]
    assert not (tmp_path / "__pycache__").exists()  # nothing was imported
