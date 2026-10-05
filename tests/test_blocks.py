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
