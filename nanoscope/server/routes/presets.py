from __future__ import annotations

from fastapi import APIRouter

from nanoscope.presets import get_preset, list_presets
from nanoscope.server.models import PresetSpecDoc
from nanoscope.specs import PresetSpec

router = APIRouter(prefix="/api", tags=["presets"])


@router.get("/presets")
def presets() -> list[PresetSpecDoc]:
    """Every preset with its fields (what a run form shows, and what can be overridden)."""
    return [PresetSpecDoc(**PresetSpec.from_preset(get_preset(n)).to_dict())
            for n in list_presets()]


@router.get("/presets/{name}")
def preset(name: str) -> PresetSpecDoc:
    return PresetSpecDoc(**PresetSpec.from_preset(get_preset(name)).to_dict())
