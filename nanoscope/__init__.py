"""Nanoscope — see what your language model learns."""

__version__ = "0.3.0rc1"

from nanoscope.compare import compare
from nanoscope.presets import Preset, get_preset, list_presets
from nanoscope.run import RunGroup, RunResult, run
from nanoscope.store import load_run
from nanoscope.study import FLOPs, Study, Tokens

__all__ = [
    "FLOPs", "Preset", "RunGroup", "RunResult", "Study", "Tokens",
    "compare", "get_preset", "list_presets", "load_run", "run",
]
