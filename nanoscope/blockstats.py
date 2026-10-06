"""Per-block training statistics, written at eval steps only.

    run(Modern, "tinystories-5min", block_stats=True)   # appends to <run>/blockstats.jsonl

At each eval step one forward and backward pass runs on the first validation batch, and every
block of the model (`model.blocks`, or the whole model if it has none) gets:

    activation_rms     root mean square of the block's output
    grad_norm          norm of the loss gradient over the block's parameters
    update_to_weight   how far the weights moved per step since the previous eval, relative
                       to their size (None at the first eval)
    attention_entropy  mean entropy in nats of the attention weights, from the explicit
                       softmax (None for a block without attention)

It plugs in as an `on_eval` hook, so the trainer knows nothing about it. The gradient comes
from `torch.autograd.grad`, which leaves the training step's own `.grad` untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope import __version__
from nanoscope.blocks.attention import Attention
from nanoscope.dataset import Data
from nanoscope.presets import Preset
from nanoscope.schemas.upgrade import upgrade
from nanoscope.train_loop import _split_output

FILE = "blockstats.jsonl"


def _blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    layers = getattr(model, "blocks", None)
    if isinstance(layers, nn.ModuleList):
        return [(f"blocks.{i}", m) for i, m in enumerate(layers)]
    return [("model", model)]


def _norm(tensors: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(t.float().pow(2).sum()) for t in tensors))


class BlockStats:
    """An `on_eval(step, val_loss)` hook. Build one with the model being trained."""

    def __init__(self, model: nn.Module, data: Data, preset: Preset, run_dir: Path,
                 device: torch.device) -> None:
        self.model, self.preset, self.device = model, preset, device
        self.path = Path(run_dir) / FILE
        val = torch.from_numpy(data.val.astype(np.int64))
        ctx, bs = preset.context_length, preset.batch_size
        rows = max(1, min(bs, (len(val) - 1) // ctx))
        self.inputs = val[: rows * ctx].view(rows, ctx)
        self.targets = val[1 : rows * ctx + 1].view(rows, ctx)
        self._previous: dict[str, list[torch.Tensor]] = {}
        self._previous_step = 0

    def __call__(self, step: int, val_loss: float) -> None:
        line = {"schema": 1, "nanoscope": __version__, "step": step, "val_loss": val_loss,
                "blocks": self.measure(step)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def measure(self, step: int) -> list[dict[str, Any]]:
        blocks = _blocks(self.model)
        outputs: dict[str, torch.Tensor] = {}
        attn_inputs: dict[str, list[tuple[Attention, torch.Tensor]]] = {n: [] for n, _ in blocks}
        handles = []
        for name, block in blocks:
            def keep(_m: nn.Module, _a: Any, out: Any, name: str = name) -> None:
                tensor = out[0] if isinstance(out, tuple) else out
                outputs[name] = tensor.detach()
            handles.append(block.register_forward_hook(keep))
            for module in block.modules():
                if isinstance(module, Attention):
                    def grab(m: nn.Module, args: tuple[Any, ...], name: str = name) -> None:
                        attn_inputs[name].append((m, args[0].detach()))  # type: ignore[arg-type]
                    handles.append(module.register_forward_pre_hook(grab))

        was_training = self.model.training
        self.model.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        try:
            logits, aux = _split_output(self.model(self.inputs.to(self.device)))
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   self.targets.to(self.device).reshape(-1)) + aux
            grads = dict(zip((id(p) for p in params),
                             torch.autograd.grad(loss, params, allow_unused=True), strict=True))
            with torch.no_grad():
                stats = [self._block(name, block, outputs.get(name), attn_inputs[name], grads,
                                     step) for name, block in blocks]
        finally:
            for h in handles:
                h.remove()
            self.model.train(was_training)
        self._previous_step = step
        return stats

    def _block(self, name: str, block: nn.Module, output: torch.Tensor | None,
               attn: list[tuple[Attention, torch.Tensor]], grads: dict[int, Any],
               step: int) -> dict[str, Any]:
        params = list(block.parameters())
        weights = [p.detach().clone() for p in params]
        ratio = None
        if name in self._previous and step > self._previous_step:
            moved = _norm([w - old for w, old in zip(weights, self._previous[name], strict=True)])
            ratio = moved / max(_norm(weights), 1e-12) / (step - self._previous_step)
        self._previous[name] = weights
        entropies = []
        for module, x in attn:
            w = module.attention_weights(x)
            entropies.append(float(-torch.special.xlogy(w, w).sum(-1).mean()))
        return {
            "name": name,
            "activation_rms": (None if output is None
                               else float(output.float().pow(2).mean().sqrt())),
            "grad_norm": _norm([grads[id(p)] for p in params if grads.get(id(p)) is not None]),
            "update_to_weight": ratio,
            "attention_entropy": sum(entropies) / len(entropies) if entropies else None,
        }


def read_blockstats(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every line of a run's blockstats.jsonl, oldest first (empty if it has none)."""
    path = Path(run_dir) / FILE
    if not path.exists():
        return []
    return [upgrade(json.loads(line), "blockstats", path)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
