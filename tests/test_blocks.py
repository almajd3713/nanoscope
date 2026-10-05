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
