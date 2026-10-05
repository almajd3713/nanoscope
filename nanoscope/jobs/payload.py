"""Run payloads: a preset as plain data, and back. Jobs carry refs and data, never code."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from nanoscope.presets import Preset, get_preset


def preset_fields(preset: Preset) -> dict[str, Any]:
    """The `preset`, `overrides` and `custom_preset` keys of a run payload for this preset."""
    try:
        registered = get_preset(preset.name)
    except KeyError:
        return {"preset": preset.name, "custom_preset": asdict(preset)}
    changed = {f.name: getattr(preset, f.name) for f in fields(preset)
               if getattr(preset, f.name) != getattr(registered, f.name)}
    return {"preset": preset.name, "overrides": changed}


def build_preset(payload: dict[str, Any]) -> Preset:
    if payload.get("custom_preset") is not None:
        return Preset.from_dict(payload["custom_preset"])
    overrides = dict(payload.get("overrides") or {})
    if "betas" in overrides:
        overrides["betas"] = tuple(overrides["betas"])
    return get_preset(payload["preset"]).override(**overrides)
