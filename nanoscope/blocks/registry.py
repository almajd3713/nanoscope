"""What blocks exist: one `BlockInfo` per block class, filled in as the block modules load.

The palette, `describe`, the graph and the lesson gating all read this table. Library code
imports `nanoscope.blocks.registry` (a submodule), never the package root, so gating the
root's exports can't affect shipped models.
"""

from __future__ import annotations

import inspect
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
    user: bool = False  # registered by a user with `register_block`, not shipped with nanoscope


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


_REFERENCES: dict[str, Callable[..., Any]] = {}  # user block name -> its reference function


def register_block(cls: type[T] | None = None, *, reference: Callable[..., Any] | None = None,
                   family: str = "custom") -> Any:
    """Put your own `nn.Module` in the palette, so models can compose it like a shipped block.

        @register_block(reference=naive_gate, family="mlp")
        class Gate(nn.Module):
            def __init__(self, d_model, context_length, hidden=None): ...

    The constructor takes `d_model` and `context_length` first, like every block; the other
    arguments are the block's options, which a model file writes as `Gate(hidden=64)`.
    `reference` is a plain function that computes the same thing slowly and obviously: the
    check job compares the block against it on random inputs, and a block that passes is
    shown as certified. Use as `@register_block` or `@register_block(reference=fn, ...)`.
    """
    def register(cls: type[T]) -> type[T]:
        from nanoscope.blocks.spec import BlockModule

        init_params = inspect.signature(cls.__init__).parameters  # type: ignore[misc]
        missing = [n for n in ("d_model", "context_length") if n not in init_params]
        if missing:
            raise TypeError(
                f"register_block({cls.__name__}): __init__ must take d_model and context_length "
                f"(missing {', '.join(missing)}) so a model can build it at any size")
        block_cls: Any = cls
        if not issubclass(cls, BlockModule):
            # the same class, now also a BlockModule: calling it with options only gives a spec
            block_cls = type(cls.__name__, (BlockModule, cls), {
                "__module__": cls.__module__, "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__})
        name = reference.__qualname__ if reference is not None else None
        _register(BlockInfo(cls.__name__, family, "composite", cls.__module__, name, (), True))
        if reference is not None:
            _REFERENCES[cls.__name__] = reference
        else:
            _REFERENCES.pop(cls.__name__, None)
        return block_cls
    return register if cls is None else register(cls)


def reference_for(name: str) -> Callable[..., Any] | None:
    """The reference function a user block was registered with, if any."""
    return _REFERENCES.get(name)


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
