"""How big is a model, and how much compute does a token cost?

Non-embedding parameters follow Kaplan et al. (2020): everything except embedding tables.
A tied output matrix is the embedding table, so it is excluded; an untied one counts.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, non-embedding) parameter counts; shared tensors are counted once."""
    params = {id(p): p for p in model.parameters()}
    emb = {
        id(p) for m in model.modules() if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    total = sum(p.numel() for p in params.values())
    return total, total - sum(params[i].numel() for i in emb if i in params)


def flops_per_token(model: nn.Module, context_length: int) -> int:
    """Training FLOPs per token: the model's own estimate if it has one, else 6N."""
    own = getattr(model, "flops_per_token", None)
    if callable(own):
        return int(own(context_length))  # pyright: ignore[reportArgumentType]
    return 6 * count_params(model)[1]


def build_on_meta(model_cls: type[nn.Module], **kwargs: Any) -> nn.Module:
    """Construct a model without allocating its weights, for counting."""
    with torch.device("meta"):
        return model_cls(**kwargs)


def match_params(
    model_cls: type[nn.Module],
    target: int,
    knob: str,
    candidates: range | list[int],
    **kwargs: Any,
) -> dict[str, Any]:
    """Pick the value of `knob` whose non-embedding parameter count is closest to `target`.

    Example: give Modern the parameter count of GPT2 by adjusting its MLP width:
        kw = match_params(Modern, gpt2_params, "ffn_hidden", range(64, 1024, 8),
                          vocab_size=4096, d_model=128)
    """
    if knob not in inspect.signature(model_cls).parameters:
        raise TypeError(f"{model_cls.__name__} has no parameter {knob!r}")

    def distance(value: int) -> int:
        return abs(count_params(build_on_meta(model_cls, **kwargs, **{knob: value}))[1] - target)

    best = min(candidates, key=distance)
    return {**kwargs, knob: best}
