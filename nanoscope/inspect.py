"""What a model is, before it trains: shapes, parameters, FLOPs and memory per module.

    describe(Modern, "tinystories-5min", n_layers=6)

traces one forward pass on the `meta` device, so nothing is allocated and nothing is
downloaded, however large the model. The result is a JSON-ready dict (`describe.v1`) with one
row per module, in the order `named_modules` lists them.

And what a trained model does with a prompt, at any checkpoint it kept:

    inspect_checkpoint("my-runs/modern/seed-0", "Once upon a time", step=500)

returns the attention weights of every head and a logit lens (the next-token guess read off
the residual stream after each block) as an `inspect.v1` dict.

Both build the model class, which runs the user's code: the API calls them from a worker,
never in its own process (plan 8.1).
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


# -- text -----------------------------------------------------------------------------------

def _human(n: float, unit: str = "") -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= scale:
            return f"{n / scale:.3g}{suffix}{unit}"
    return f"{n:g}{unit}"


def _shape(shapes: list[list[int]] | None) -> str:
    if shapes is None:
        return "-"
    return " ".join("x".join(str(d) for d in s) for s in shapes)


def format_describe(report: dict[str, Any]) -> str:
    """The report as a table: one indented row per module, then the totals."""
    mem = report["memory"]
    lines = [
        f"{report['model']}  preset {report['preset']}  "
        f"batch {report['batch_size']} x context {report['context_length']}",
        f"params {_human(report['params']['total'])} "
        f"(non-embedding {_human(report['params']['non_embedding'])})  "
        f"FLOPs/token {_human(report['flops_per_token'])} ({report['flops_source']})",
        f"memory ~{_human(mem['total'], 'B')}: weights {_human(mem['weights'], 'B')}, "
        f"gradients {_human(mem['gradients'], 'B')}, AdamW {_human(mem['adamw_state'], 'B')}, "
        f"activations {_human(mem['activations'], 'B')}",
        "",
    ]
    rows = [(("  " * r["depth"]) + (r["path"].rsplit(".", 1)[-1] or "(model)"), r["type"],
             _shape(r["input_shapes"]), _shape(r["output_shapes"]), _human(r["params"]),
             _human(r["flops_per_token"]) + ("" if r["flops_source"] == "analytic" else "*"))
            for r in report["modules"]]
    header = ("module", "type", "in", "out", "params", "FLOPs/token")
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(6)]
    for row in [header, *rows]:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    lines += ["", "* FLOPs/token estimated as 6 x non-embedding parameters"]
    return "\n".join(lines)


DEFAULT_PROMPT = "Once upon a time"
MAX_PROMPT_TOKENS = 64  # attention maps are T x T per head, so the result grows with T squared
LENS_DIGITS = 4


def inspect_checkpoint(ref: str, prompt: str = DEFAULT_PROMPT, step: int | None = None,
                       top_k: int = 5, device: str = "cpu") -> dict[str, Any]:
    """Attention weights and a logit lens for `prompt` at the checkpoint of `step` (the latest
    if None). Archived steps (`run(..., checkpoint_steps=[...])`) are never pruned, so they are
    the ones to look at over training."""
    from nanoscope.blocks.attention import Attention
    from nanoscope.store import load_run, saved_steps
    from nanoscope.train_loop import _split_output

    loaded = load_run(ref, device=device, step=step)
    model, tokenizer = loaded.model, loaded.tokenizer
    ids = tokenizer.encode(prompt) if prompt else [tokenizer.eos_token_id]
    limit = min(MAX_PROMPT_TOKENS, loaded.preset.context_length)
    if len(ids) > limit:
        raise ValueError(f"the prompt is {len(ids)} tokens; inspect takes at most {limit} "
                         f"(attention maps grow with the square of the length)")

    attention: list[tuple[str, Attention, torch.Tensor]] = []
    residual: list[tuple[str, torch.Tensor]] = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            def grab(m: nn.Module, args: tuple[Any, ...], name: str = name) -> None:
                attention.append((name, m, args[0].detach()))  # type: ignore[arg-type]
            handles.append(module.register_forward_pre_hook(grab))
    layers = getattr(model, "blocks", None)
    lens_ready = (isinstance(layers, nn.ModuleList) and isinstance(getattr(model, "norm", None),
                  nn.Module) and isinstance(getattr(model, "head", None), nn.Module))
    if lens_ready:
        assert isinstance(layers, nn.ModuleList)
        def first_input(_m: nn.Module, args: tuple[Any, ...]) -> None:
            residual.append(("embeddings", args[0].detach()))
        handles.append(layers[0].register_forward_pre_hook(first_input))
        for i, layer in enumerate(layers):
            def keep(_m: nn.Module, _a: Any, out: Any, name: str = f"blocks.{i}") -> None:
                residual.append((name, (out[0] if isinstance(out, tuple) else out).detach()))
            handles.append(layer.register_forward_hook(keep))

    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    try:
        with torch.no_grad():
            logits, _ = _split_output(model(x))
            maps = [(name, module.attention_weights(inp)[0]) for name, module, inp in attention]
            lens_logits = ([(name, model.head(model.norm(h))[0]) for name, h in residual]  # type: ignore[operator]
                           if lens_ready else [("output", logits[0])])
    finally:
        for h in handles:
            h.remove()

    def text(i: int) -> str:
        return "<eos>" if i == tokenizer.eos_token_id else tokenizer.decode([i])

    def lens_row(name: str, layer_logits: torch.Tensor) -> dict[str, Any]:
        probs = layer_logits.float().softmax(-1)
        top = probs.topk(min(top_k, probs.size(-1)), dim=-1)
        positions = []
        for t in range(len(ids)):
            nxt = None
            if t + 1 < len(ids):
                target = ids[t + 1]
                nxt = {"id": target, "text": text(target),
                       "p": round(float(probs[t, target]), LENS_DIGITS),
                       "rank": int((probs[t] > probs[t, target]).sum()) + 1}
            positions.append({
                "top": [{"id": int(i), "text": text(int(i)), "p": round(float(p), LENS_DIGITS)}
                        for p, i in zip(top.values[t], top.indices[t], strict=True)],
                "next": nxt})
        return {"name": name, "positions": positions}

    notes = []
    if not maps:
        notes.append(f"{type(model).__name__} has no Attention blocks, so there are no maps")
    if not lens_ready:
        notes.append(f"{type(model).__name__} has no blocks/norm/head, so the lens shows only "
                     "the model's output")
    return {
        "schema": 1, "nanoscope": __version__, "ref": str(ref), "model": type(model).__name__,
        "step": loaded.step, "steps": saved_steps(loaded.run_dir), "prompt": prompt,
        "tokens": [{"id": i, "text": text(i)} for i in ids],
        "attention": [{"module": name, "layer": n, "heads": int(w.size(0)),
                       "weights": [[[round(float(v), LENS_DIGITS) for v in row[: r + 1]]
                                    for r, row in enumerate(head)] for head in w]}
                      for n, (name, w) in enumerate(maps)],
        "lens": {"top_k": top_k, "layers": [lens_row(n, lg) for n, lg in lens_logits]},
        "notes": notes,
    }


def _quoted(text: str) -> str:
    import json

    return json.dumps(text, ensure_ascii=False)


def format_inspect(report: dict[str, Any]) -> str:
    """The logit lens as a table (the top guess after each layer, per position) and, for the
    last token, the position each head attends to most."""
    tokens = report["tokens"]
    steps = ", ".join(map(str, report["steps"]))
    lines = [f"{report['ref']} at step {report['step']} (checkpoints: {steps}); "
             f"prompt of {len(tokens)} tokens", ""]
    layers = report["lens"]["layers"]
    header = ["pos", "token", *(layer["name"] for layer in layers), "actual next (p, rank)"]
    rows = []
    for t, tok in enumerate(tokens):
        cells = [str(t), _quoted(tok["text"])]
        for layer in layers:
            best = layer["positions"][t]["top"][0]
            cells.append(f"{_quoted(best['text'])} {best['p']:.3f}")
        nxt = layers[-1]["positions"][t]["next"]
        cells.append("" if nxt is None
                     else f"{_quoted(nxt['text'])} {nxt['p']:.3f}, #{nxt['rank']}")
        rows.append(cells)
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    lines.append("logit lens: the top next-token guess after each layer (p)")
    for r in [header, *rows]:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
    if report["attention"]:
        last = len(tokens) - 1
        lines += ["", f"attention from the last token {_quoted(tokens[last]['text'])}: "
                      "the position each head weights most"]
        for entry in report["attention"]:
            cells = []
            for h, head in enumerate(entry["weights"]):
                row = head[last]
                pos = max(range(len(row)), key=row.__getitem__)
                cells.append(f"h{h} {pos} {_quoted(tokens[pos]['text'])} {row[pos]:.2f}")
            lines.append(f"{entry['module']}  " + "  ".join(cells))
    lines += [f"note: {n}" for n in report["notes"]]
    return "\n".join(lines)
