"""The block library: short modules, one concept each, that models are composed from.

    from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, RoPE, SwiGLU

Names are exported lazily (PEP 562), so `import nanoscope.blocks` costs nothing and a name
can be locked by a lesson at the moment it is imported. Library code imports the submodule
it needs (`nanoscope.blocks.attention`), never this package root.
"""

from __future__ import annotations

import importlib
from typing import Any

# name -> module that defines it; blocks add themselves as they are written
_EXPORTS: dict[str, str] = {
    "BlockInfo": "nanoscope.blocks.registry",
    "register_block": "nanoscope.blocks.registry",
    "BlockModule": "nanoscope.blocks.spec",
    "BlockSpec": "nanoscope.blocks.spec",
    "Composite": "nanoscope.blocks.composite",
    "Residual": "nanoscope.blocks.composite",
    "TokenEmbedding": "nanoscope.blocks.embedding",
    "LearnedPosition": "nanoscope.blocks.embedding",
    "Head": "nanoscope.blocks.head",
    "LayerNorm": "nanoscope.blocks.norm",
    "RMSNorm": "nanoscope.blocks.norm",
    "RoPE": "nanoscope.blocks.positional",
    "NoPE": "nanoscope.blocks.positional",
    "ALiBi": "nanoscope.blocks.positional",
    "Attention": "nanoscope.blocks.attention",
    "Block": "nanoscope.blocks.structure",
    "AttentionTemplate": "nanoscope.blocks.templates.attention",
    "BlockTemplate": "nanoscope.blocks.templates.block",
    "Decoder": "nanoscope.blocks.structure",
    "GELUMLP": "nanoscope.blocks.mlp",
    "SwiGLU": "nanoscope.blocks.mlp",
    **{name: "nanoscope.blocks.primitives" for name in (
        "Linear", "Activation", "CausalMask", "ScaledDotScores", "Softmax", "WeightedSum",
        "SplitHeads", "MergeHeads")},
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nanoscope.blocks' has no block {name!r}")
    _check_unlocked(name)
    return getattr(importlib.import_module(module), name)


def _check_unlocked(name: str) -> None:
    """Under the `guided` policy a block the learner has not unlocked can't be imported from
    here. Library code imports the submodules, so this never touches a shipped model."""
    from nanoscope.learn import unlocks

    if not unlocks.exists():  # the common case: no gating at all
        return
    from nanoscope.learn.gating import LockedBlockError, check

    locked = check([f"block:{name}"])
    if locked:
        raise LockedBlockError(locked[0])


def load_all() -> None:
    """Import every block module (registering the blocks) without going through the
    gated names. Used by the catalog and the graph, which are not composing a model."""
    for module in sorted(set(_EXPORTS.values())):
        importlib.import_module(module)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)
