"""Where a run lives: its ref, worked out from the request alone.

`run()` names its folder from the model, the keywords that differ from the defaults, and the
preset. A service that has only *read* the model (never imported it) needs the same name to tell
a client "that run is already done", so the rule lives here, free of torch and of user code.

    runs/<preset folder>/<model name>[-<hash of what differs>]/seed-<n>
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import Any

from nanoscope.presets import Preset, get_preset, list_presets

REQUIRED = object()  # a constructor parameter with no default


def hash_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def preset_dir(preset: Preset) -> str:
    """A registered preset's name; a custom or modified Preset object gets name-<hash>."""
    if preset.name in list_presets() and get_preset(preset.name) == preset:
        return preset.name
    return f"{preset.name}-{hash_of(asdict(preset))}"


def run_name(model_name: str, defaults: Mapping[str, Any], model_kwargs: Mapping[str, Any],
             given: Preset, preset: Preset) -> str:
    """`bigram` for defaults; `bigram-1a2b3c4d` once anything differs, so runs never collide.
    `defaults` maps each constructor parameter to its default (or `REQUIRED`)."""
    changed_model = {k: v for k, v in model_kwargs.items()
                     if defaults[k] is REQUIRED or v != defaults[k]}
    changed_preset = {f.name: getattr(preset, f.name) for f in fields(preset)
                      if getattr(preset, f.name) != getattr(given, f.name)}
    name = model_name.lower()
    if changed_model or changed_preset:
        name += "-" + hash_of({"model": changed_model, "preset": changed_preset})
    return name


def run_ref(model_name: str, defaults: Mapping[str, Any], model_kwargs: Mapping[str, Any],
            given: Preset, preset: Preset, seed: int) -> str:
    return f"{preset_dir(given)}/{run_name(model_name, defaults, model_kwargs, given, preset)}" \
           f"/seed-{seed}"
