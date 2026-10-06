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


class ShapeError(ValueError):
    """The meta trace failed inside `module`; `line` is where in `file` it is written (the
    graph's span for that block when the file is representable, else the traceback's)."""

    def __init__(self, module: str, message: str, file: str | None, line: int | None) -> None:
        where = f" ({file}:{line})" if file and line else ""
        super().__init__(f"{module or '<model>'}: {message}{where}")
        self.module, self.reason, self.file, self.line = module, message, file, line


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
    rows = _trace(model, preset, model_cls)

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


def _trace(model: nn.Module, preset: Preset, model_cls: type[nn.Module]) -> list[dict[str, Any]]:
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

    active: list[str] = []  # modules entered and not yet left: the last is where an error is

    def enter(module: nn.Module, args: tuple[Any, ...]) -> None:
        active.append(names[id(module)])

    def leave(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        active.pop()
        hook(module, args, output)

    handles = [m.register_forward_pre_hook(enter) for m in model.modules()]
    handles += [m.register_forward_hook(leave) for m in model.modules()]
    try:
        idx = torch.zeros(preset.batch_size, preset.context_length, dtype=torch.long,
                          device="meta")
        with torch.no_grad():
            model(idx)
    except Exception as exc:
        module = active[-1] if active else ""
        file, line = _locate(model_cls, module, exc)
        raise ShapeError(module, f"{type(exc).__name__}: {exc}", file, line) from exc
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


# -- where in the source a module is written ---------------------------------------------------

BLOCK_ARGS = {"norm1": "norm", "norm2": "norm", "norm": "final_norm"}  # module attr -> graph arg


def _locate(model_cls: type[nn.Module], module: str,
            exc: BaseException) -> tuple[str | None, int | None]:
    """(file, line) for the module that failed: its node in the file's graph if the class is
    representable, else the innermost frame of the traceback that is in the model's file."""
    try:
        file = _inspect.getsourcefile(model_cls)
    except (TypeError, OSError):
        file = None
    if file is None:
        return None, None
    from nanoscope.blocks.graph import parse

    try:
        cls = next((c for c in parse(file)["classes"] if c["name"] == model_cls.__name__), None)
    except (OSError, ValueError):
        cls = None
    if cls is not None and cls["representable"] and cls["kind"] == "decoder":
        span = _span_of(cls, module)
        if span is not None:
            return file, span["line"]
    tb, line = exc.__traceback__, None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == file:
            line = tb.tb_lineno
        tb = tb.tb_next
    return file, line


def _span_of(cls: dict[str, Any], module: str) -> dict[str, int] | None:
    args = cls["args"]
    parts = module.split(".") if module else []
    if not parts:
        return cls["call_span"] if "call_span" in cls else cls["span"]
    node: dict[str, Any] | None
    if parts[0] == "blocks" and len(parts) >= 2 and parts[1].isdigit():
        if "block" in args:
            node = args["block"]
        elif "pattern" in args and args["pattern"]["kind"] == "list" and args["pattern"]["items"]:
            items = args["pattern"]["items"]
            node = items[int(parts[1]) % len(items)]
        else:
            return None
        rest = parts[2:]
    elif parts[0] in ("norm", "pos_emb"):
        node = args.get("final_norm" if parts[0] == "norm" else "pos_emb")
        rest = parts[1:]
    else:  # tok_emb and head have no node of their own
        return cls.get("call_span") or cls["span"]
    span = node["span"] if node else None
    for part in rest:
        if node is None or node["kind"] != "block":
            break
        child = node["args"].get(BLOCK_ARGS.get(part, part))
        if child is None:
            break
        node, span = child, child["span"]
    return span or cls["span"]
