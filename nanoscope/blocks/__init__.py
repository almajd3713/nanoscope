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
    "BlockModule": "nanoscope.blocks.spec",
    "BlockSpec": "nanoscope.blocks.spec",
    "Composite": "nanoscope.blocks.composite",
    "Residual": "nanoscope.blocks.composite",
    **{name: "nanoscope.blocks.primitives" for name in (
        "Linear", "Activation", "CausalMask", "ScaledDotScores", "Softmax", "WeightedSum",
        "SplitHeads", "MergeHeads")},
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nanoscope.blocks' has no block {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)
