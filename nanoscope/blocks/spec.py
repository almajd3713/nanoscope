"""Block specs: a block described by its options, before it has a size.

    Attention(n_heads=4, n_kv_heads=2)      # no d_model: returns a BlockSpec, no tensors
    spec.build(d_model=128, context_length=256)   # a real module with fresh parameters

`Decoder` and `Block` build their specs once per layer, so layers never share parameters.
`spec.to_dict()` is what `describe` and the graph show.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

CONTEXT = ("d_model", "context_length")  # given by whoever builds the block, never by the user


def _options(cls: type) -> dict[str, inspect.Parameter]:
    params = inspect.signature(cls.__init__).parameters
    return {k: p for k, p in params.items() if k != "self" and k not in CONTEXT}


@dataclass(frozen=True)
class BlockSpec:
    """A block class and its own options. Cheap, immutable, and safe to reuse."""

    cls: type
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        accepted = _options(self.cls)
        unknown = set(self.options) - set(accepted)
        if unknown:
            raise TypeError(f"{self.cls.__name__} has no option {', '.join(sorted(unknown))}; "
                            f"its options are: {', '.join(accepted) or 'none'}")
        missing = [k for k, p in accepted.items()
                   if p.default is inspect.Parameter.empty and k not in self.options]
        if missing:
            raise TypeError(f"{self.cls.__name__} needs: {', '.join(missing)}")

    def build(self, d_model: int, context_length: int) -> nn.Module:
        """A new module; nested specs among the options are built by the block itself."""
        return self.cls(d_model=d_model, context_length=context_length, **self.options)

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, BlockSpec):
                return value.to_dict()
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value
        return {"block": self.cls.__name__, "args": {k: plain(v) for k, v in self.options.items()}}


def build_option(value: Any, d_model: int, context_length: int) -> Any:
    """Build an option that is a spec (or None / an already-built module) the block was given."""
    if isinstance(value, BlockSpec):
        return value.build(d_model, context_length)
    return value


def _empty_instance(cls: type) -> Any:
    return object.__new__(cls)


class BlockModule(nn.Module):
    """Base of every block. Called with its options only, it returns a `BlockSpec`; called
    with `d_model` and `context_length` as well, it is an ordinary module."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if "d_model" in kwargs or "context_length" in kwargs:
            return super().__new__(cls)
        if args:
            raise TypeError(f"{cls.__name__}: give options by keyword, e.g. "
                            f"{cls.__name__}(option=value)")
        return BlockSpec(cls, kwargs)

    def __reduce_ex__(self, protocol: Any) -> Any:
        # copy.deepcopy and pickle call cls.__new__(cls) with no options, which would give a
        # spec. Rebuild an empty instance instead and restore its state.
        return (_empty_instance, (type(self),), self.__dict__.copy())
