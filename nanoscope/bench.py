"""`nanoscope bench`: how fast does this model train here, and what is the bottleneck?

Small models often leave the GPU idle, waiting for Python to launch the next kernel. The
verdict compares the time the GPU spends running kernels with the wall time of a step.
Close other GPU programs first: anything else running skews the numbers.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from nanoscope import __version__, paths
from nanoscope.dataset import load_data
from nanoscope.presets import Preset, get_preset
from nanoscope.run import _resolve_device as resolve_device
from nanoscope.sizing import flops_per_token
from nanoscope.train_loop import _peak_tflops, train

CPU_BOUND_BELOW = 0.7  # GPU busy fraction


@dataclass
class BenchResult:
    model: str
    device: str
    step_ms: float
    tokens_per_sec: float
    tflops: float
    mfu: float | None
    gpu_busy: float | None  # fraction of a step the GPU spends running kernels
    preset: str = ""
    steps: int = 0
    compile: str = "off"

    def to_dict(self) -> dict[str, Any]:
        """The bench.v1 row that `save=True` appends to hardware/bench.jsonl."""
        return {
            "schema": 1, "nanoscope": __version__,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model, "preset": self.preset, "device": self.device,
            "torch": torch.__version__, "steps": self.steps, "compile": self.compile,
            "step_ms": self.step_ms, "tokens_per_sec": self.tokens_per_sec,
            "tflops": self.tflops, "mfu": self.mfu, "gpu_busy": self.gpu_busy,
            "verdict": self.verdict,
        }

    @property
    def verdict(self) -> str:
        if self.gpu_busy is None:
            return "CPU run: nothing to compare against."
        if self.gpu_busy < CPU_BOUND_BELOW:
            return (f"CPU-bound: the GPU is busy {self.gpu_busy:.0%} of each step and waits "
                    "on Python the rest.")
        return f"GPU-bound: the GPU is busy {self.gpu_busy:.0%} of each step."

    def __str__(self) -> str:
        lines = [
            f"{self.model} on {self.device}",
            f"  step time      {self.step_ms:.1f} ms",
            f"  throughput     {self.tokens_per_sec:,.0f} tokens/s  ({self.tflops:.2f} TFLOPs)",
        ]
        if self.mfu is not None:
            lines.append(f"  FLOPs util.    {self.mfu:.1%} of peak")
        lines.append(f"  {self.verdict}")
        return "\n".join(lines)


def _bench_preset(preset: Preset, steps: int) -> Preset:
    return preset.override(
        max_steps=steps, warmup_steps=1, eval_interval=10**9, sample_interval=10**9,
        checkpoint_interval=10**9, sample_length=1,
    )


def _step_seconds(model, data, preset, device, warmup: int, compile) -> list[float]:
    with tempfile.TemporaryDirectory() as tmp:
        result = train(model, data, preset, Path(tmp), device, resume=False, compile=compile)
    return [row["elapsed"] for row in result.metrics[warmup:]]


def bench(model_cls, preset: str | Preset = "tinystories-5min", steps: int = 60,
          warmup: int = 10, device: str | None = None, profile: bool = True,
          compile: bool | str = False, save: bool = False, **model_kwargs) -> BenchResult:
    import inspect

    preset = get_preset(preset) if isinstance(preset, str) else preset
    dev = resolve_device(device)
    data = load_data(preset)
    preset = _bench_preset(preset, warmup + steps)

    def build():
        from_data = {"vocab_size": data.tokenizer.vocab_size,
                     "context_length": preset.context_length}
        params = inspect.signature(model_cls).parameters
        return model_cls(**({k: v for k, v in from_data.items() if k in params} | model_kwargs))

    model = build()
    seconds = _step_seconds(model, data, preset, dev, warmup, compile)
    step = statistics.median(seconds)
    tokens = preset.batch_size * preset.context_length
    flops = flops_per_token(model, preset.context_length) * tokens
    peak = _peak_tflops(dev)
    tflops = flops / step / 1e12

    busy = None
    if dev.type == "cuda" and profile:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torch_profile

        model = build()
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            _step_seconds(model, data, preset, dev, warmup, compile)
            torch.cuda.synchronize(dev)
        kernel_s = sum(e.self_device_time_total for e in prof.key_averages()) / 1e6
        busy = min(kernel_s / (warmup + steps) / step, 1.0)

    result = BenchResult(
        model=model_cls.__name__, device=str(dev), step_ms=step * 1e3,
        tokens_per_sec=tokens / step, tflops=tflops,
        mfu=tflops / peak if peak else None, gpu_busy=busy,
        preset=preset.name, steps=steps, compile=str(compile).lower() if compile else "off",
    )
    if save:
        path = paths.hardware_dir() / "bench.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
    return result
