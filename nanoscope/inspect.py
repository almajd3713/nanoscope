"""What a model is, before it trains: shapes, parameters, FLOPs and memory per module.

    describe(Modern, "tinystories-5min", n_layers=6)

traces one forward pass on the `meta` device, so nothing is allocated and nothing is
downloaded, however large the model. The result is a JSON-ready dict (`describe.v1`) with one
row per module, in the order `named_modules` lists them.

It builds the model class, which runs the user's code: the API calls it from a worker, never
in its own process (plan 8.1).
"""

from __future__ import annotations

import inspect as _inspect
from typing import Any

import torch
import torch.nn as nn

from nanoscope import __version__
from nanoscope.blocks.registry import all_blocks
from nanoscope.presets import Preset, get_preset
from nanoscope.sizing import count_params, flops_per_token

VOCAB_SIZES = {"bytes": 257, "gpt2": 50257}  # tokenizers with a fixed vocabulary
BYTES_PER_PARAM = 4  # fp32 master weights, gradients and AdamW's two moments
ACTIVATION_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


def _vocab_size(preset: Preset) -> int:
    if preset.tokenizer == "bpe":
        if preset.vocab_size is None:
            raise ValueError("tokenizer='bpe' needs a vocab_size")
        return preset.vocab_size
    return VOCAB_SIZES[preset.tokenizer]


def _shapes(value: Any) -> list[list[int]]:
    """The shapes of every tensor in a module's input or output."""
    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, (tuple, list)):
        return [s for v in value for s in _shapes(v)]
    return []


def _numel(shapes: list[list[int]]) -> int:
    total = 0
    for shape in shapes:
        n = 1
        for d in shape:
            n *= d
        total += n
    return total


def describe(model_cls: type[nn.Module], preset: str | Preset = "tinystories-5min",
             **kwargs: Any) -> dict[str, Any]:
    """Trace `model_cls` at `preset`'s context length and batch size. Keywords that match the
    model's constructor go to it; the rest override preset fields, as in `run`."""
    from nanoscope.run import _FROM_DATA, _split_kwargs

    if isinstance(preset, str):
        preset = get_preset(preset)
    model_kwargs, overrides = _split_kwargs(model_cls, preset, kwargs)
    preset = preset.override(**overrides)
    from_data = {"vocab_size": _vocab_size(preset), "context_length": preset.context_length}
    params = _inspect.signature(model_cls).parameters
    model_kwargs = {k: v for k, v in from_data.items() if k in params} | model_kwargs
    full_kwargs = {n: p.default for n, p in params.items()
                   if p.default is not _inspect.Parameter.empty and n not in _FROM_DATA
                   } | model_kwargs

    with torch.device("meta"):
        model = model_cls(**model_kwargs)
    rows = _trace(model, preset)

    total, non_embedding = count_params(model)
    own_flops = _flops(model, preset.context_length)
    activations = sum(r["_out_numel"] for r in rows if r["_leaf"])
    for row in rows:
        row.pop("_out_numel")
        row.pop("_leaf")
    weights = total * BYTES_PER_PARAM
    memory = {
        "weights": weights,
        "gradients": weights,
        "adamw_state": 2 * weights,
        "activations": activations * ACTIVATION_BYTES.get(preset.precision, 4),
    }
    memory["total"] = sum(memory.values())
    return {
        "schema": 1, "nanoscope": __version__, "model": model_cls.__name__,
        "kwargs": full_kwargs, "preset": preset.name,
        "context_length": preset.context_length, "batch_size": preset.batch_size,
        "params": {"total": total, "non_embedding": non_embedding},
        "flops_per_token": own_flops[0], "flops_source": own_flops[1],
        "memory": memory, "modules": rows,
    }


def _flops(module: nn.Module, context_length: int) -> tuple[int, str]:
    if callable(getattr(module, "flops_per_token", None)):
        return flops_per_token(module, context_length), "analytic"
    return 6 * count_params(module)[1], "6N"


def _trace(model: nn.Module, preset: Preset) -> list[dict[str, Any]]:
    """One forward pass on meta tensors with a hook on every module: a row per module, with
    the shapes of its first call (None for a module the pass never called, like a ModuleList)."""
    names = {id(m): n for n, m in model.named_modules()}
    rows: dict[str, dict[str, Any]] = {}

    def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        name = names[id(module)]
        if name in rows:  # a module called twice keeps its first shapes
            return
        shapes_out = _shapes(output)
        rows[name] = {"in": _shapes(args), "out": shapes_out, "_out_numel": _numel(shapes_out)}

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        idx = torch.zeros(preset.batch_size, preset.context_length, dtype=torch.long,
                          device="meta")
        with torch.no_grad():
            model(idx)
    finally:
        for h in handles:
            h.remove()

    blocks = {info.name: info for info in all_blocks()}
    out = []
    for name, module in model.named_modules():
        seen = rows.get(name)
        total, non_embedding = count_params(module)
        flops, source = _flops(module, preset.context_length)
        info = blocks.get(type(module).__name__)
        out.append({
            "path": name, "type": type(module).__name__,
            "block": info.name if info else None, "family": info.family if info else None,
            "tier": info.tier if info else None,
            "depth": name.count(".") + 1 if name else 0,
            "input_shapes": seen["in"] if seen else None,
            "output_shapes": seen["out"] if seen else None,
            "params": total, "params_non_embedding": non_embedding,
            "flops_per_token": flops, "flops_source": source,
            "_out_numel": seen["_out_numel"] if seen else 0,
            "_leaf": seen is not None and not list(module.children()),
        })
    return out
