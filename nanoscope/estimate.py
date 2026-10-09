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


def estimate_study(runs: list[tuple[str, Preset]], devices: list[str] | None = None,
                   workers_per_device: int = 1) -> dict[str, Any]:
    """How long a study takes: one `estimate_seconds` per run (model name, preset with the
    steps that run will take), shared out over the devices the way workers take jobs (the next
    run goes to the device that is free first).

    `total_seconds` is the compute of all measured runs added up; `wall_seconds` is when the
    last device finishes. Runs with no measurement or declared estimate are counted in
    `unknown_runs` and left out of both, so `complete` says whether the numbers are whole."""
    devices = devices or ["cpu"]
    lanes = [d for d in devices for _ in range(workers_per_device)]
    free = [0.0] * len(lanes)
    total, unknown, sources = 0.0, 0, set()
    for name, preset in runs:
        seconds = [estimate_seconds(name, preset, d) for d in lanes]
        known = [i for i, e in enumerate(seconds) if e.seconds is not None]
        if not known:
            unknown += 1
            continue
        pick = min(known, key=lambda i: free[i] + seconds[i].seconds)  # type: ignore[operator]
        free[pick] += seconds[pick].seconds  # type: ignore[operator]
        total += seconds[pick].seconds  # type: ignore[operator]
        sources.add(seconds[pick].source)
    out: dict[str, Any] = {
        "runs": len(runs), "unknown_runs": unknown, "complete": unknown == 0,
        "devices": devices, "total_seconds": total if len(runs) > unknown else None,
        "wall_seconds": max(free) if len(runs) > unknown else None,
        "sources": sorted(sources)}
    if len(runs) == unknown:
        out["text"] = ("no speed measurements for these models on these devices; run "
                       "`nanoscope bench <model> --save` to get an estimate")
    else:
        wall = out["wall_seconds"]
        shown = f"{wall:.0f} s" if wall < 60 else f"{wall / 60:.3g} min"
        out["text"] = (f"about {shown} on {', '.join(dict.fromkeys(devices))} "
                       f"({total / 60:.3g} min of compute over {len(runs) - unknown} runs)")
        if unknown:
            out["text"] += f"; {unknown} run(s) have no measurement and are left out"
    return out


def spec_runs(spec: Any) -> list[tuple[str, Preset]]:
    """The runs of a study spec as (model name, preset), from the spec alone (nothing imported).
    A tokens budget sets each run's steps; a FLOPs budget needs the models' sizes, so those runs
    keep the preset's steps and the caller should treat the estimate as rough."""
    import math

    if spec.custom_preset is not None:
        preset = Preset(**spec.custom_preset)
    else:
        preset = get_preset(spec.preset).override(**{
            k: tuple(v) if k == "betas" else v for k, v in spec.overrides.items()})
    if spec.budget and "tokens" in spec.budget:
        steps = math.ceil(spec.budget["tokens"] / (preset.batch_size * preset.context_length))
        preset = preset.override(max_steps=steps)
    runs = []
    for seed in spec.seeds:  # noqa: B007
        for variant in spec.variants:
            name = variant.model.rpartition(":")[2]
            runs.append((name, preset))
    return runs
