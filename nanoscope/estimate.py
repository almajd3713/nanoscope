"""How long will this take on this machine?

    estimate_seconds(Modern, "tinystories-5min", "cuda:0", declared_minutes=3)

Measured speed from `nanoscope bench --save` (hardware/bench.jsonl) wins: tokens to train on
divided by the measured tokens per second, for the same model on the same kind of device.
Without a measurement the lesson's declared estimate is used, and without that, nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nanoscope import paths
from nanoscope.presets import Preset, get_preset


@dataclass(frozen=True)
class Estimate:
    seconds: float | None
    source: str  # "bench" | "declared" | "unknown"
    detail: str

    def __str__(self) -> str:
        if self.seconds is None:
            return self.detail
        minutes = self.seconds / 60
        shown = f"{self.seconds:.0f} s" if minutes < 1 else f"{minutes:.3g} min"
        return f"about {shown} ({self.detail})"


def _kind(device: str) -> str:
    return str(device).split(":")[0]


def _bench_rows() -> list[dict[str, Any]]:
    path = paths.hardware_dir() / "bench.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("tokens_per_sec"):
            rows.append(row)
    return rows


def estimate_seconds(model: type | str, preset: str | Preset, device: str = "cpu",
                     declared_minutes: float | None = None) -> Estimate:
    """Seconds to train `model` on `preset` on a `device` like this one."""
    name = model if isinstance(model, str) else model.__name__
    if isinstance(preset, str):
        preset = get_preset(preset)
    matches = [r for r in _bench_rows()
               if str(r.get("model", "")).lower() == name.lower()
               and _kind(r.get("device", "")) == _kind(device)]
    if matches:
        # the newest measurement, preferring one made on this preset
        same = [r for r in matches if r.get("preset") == preset.name]
        row = (same or matches)[-1]
        tokens = preset.max_steps * preset.batch_size * preset.context_length
        return Estimate(tokens / row["tokens_per_sec"], "bench",
                        f"measured {row['tokens_per_sec']:,.0f} tokens/s on {row['device']}, "
                        f"{row.get('at', 'earlier')}")
    if declared_minutes is not None:
        return Estimate(declared_minutes * 60, "declared",
                        "the lesson's estimate; run `nanoscope bench` for one measured here")
    return Estimate(None, "unknown", f"no measurement for {name} on {_kind(device)}; run "
                    f"`nanoscope bench {name.lower()} --save`")
