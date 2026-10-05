"""What blocks exist: one `BlockInfo` per block class, filled in as the block modules load.

The palette, `describe`, the graph and the lesson gating all read this table. Library code
imports `nanoscope.blocks.registry` (a submodule), never the package root, so gating the
root's exports can't affect shipped models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

TIERS = ("primitive", "composite")

T = TypeVar("T")


@dataclass(frozen=True)
class BlockInfo:
    name: str
    family: str  # embedding, positional, norm, attention, mlp, structure, primitive, ...
    tier: str  # "primitive" (never locked) or "composite"
    module: str  # where the class lives, e.g. nanoscope.blocks.attention
    reference: str | None = None  # the naive function in nanoscope.reference it is checked against
    features: tuple[str, ...] = field(default_factory=tuple)  # options a lesson can lock, e.g. GQA


_BLOCKS: dict[str, BlockInfo] = {}


def _register(info: BlockInfo) -> BlockInfo:
    if info.tier not in TIERS:
        raise ValueError(f"block {info.name!r}: tier must be one of {', '.join(TIERS)}")
    if info.name in _BLOCKS and _BLOCKS[info.name] != info:
        raise ValueError(f"block {info.name!r} is already registered differently")
    _BLOCKS[info.name] = info
    return info


def block(family: str, tier: str = "composite", *, reference: str | None = None,
          features: tuple[str, ...] = ()) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a block class (by its name) in this module's table."""
    def register(cls: type[T]) -> type[T]:
        _register(BlockInfo(cls.__name__, family, tier, cls.__module__, reference, features))
        return cls
    return register


def get_info(name: str) -> BlockInfo:
    if name not in _BLOCKS:
        raise KeyError(f"unknown block {name!r}; known: {', '.join(sorted(_BLOCKS)) or 'none'}")
    return _BLOCKS[name]


def all_blocks() -> list[BlockInfo]:
    return sorted(_BLOCKS.values(), key=lambda b: (b.family, b.name))


def shipped(cls: type[T]) -> type[T]:
    """Mark a library model (Bigram, GPT2, Modern). Gating never blocks a shipped model, and
    the build-time check only looks at models that are not shipped. A subclass inherits the
    mark: subclassing a shipped model counts as running it (a documented loophole)."""
    cls.__nanoscope_shipped__ = True  # type: ignore[attr-defined]
    return cls


def is_shipped(cls: type[Any]) -> bool:
    return bool(getattr(cls, "__nanoscope_shipped__", False))
