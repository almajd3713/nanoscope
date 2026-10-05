"""Where nanoscope keeps things on disk.

With $NANOSCOPE_HOME set, everything lives under it (containers set it to /nanoscope).
Without it the defaults are today's: ./runs, ./experiments and ~/.nanoscope/data.
Every function reads the environment when called, never at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

_USER_HOME = Path.home() / ".nanoscope"


def home() -> Path | None:
    """$NANOSCOPE_HOME, or None when it isn't set."""
    value = os.environ.get("NANOSCOPE_HOME")
    return Path(value) if value else None


def runs_dir() -> Path:
    base = home()
    return base / "runs" if base else Path("runs")


def reports_dir() -> Path:
    base = home()
    return base / "experiments" if base else Path("experiments")


def data_dir() -> Path:
    """Token data cache. $NANOSCOPE_DATA_DIR wins over $NANOSCOPE_HOME."""
    explicit = os.environ.get("NANOSCOPE_DATA_DIR")
    if explicit:
        return Path(explicit)
    base = home()
    return base / "data" if base else _USER_HOME / "data"


def learn_dir() -> Path:
    """Lesson progress and unlocks."""
    base = home()
    return base / "learn" if base else _USER_HOME / "learn"


def hardware_dir() -> Path:
    """Bench results and other facts about this machine."""
    base = home()
    return base / "hardware" if base else _USER_HOME / "hardware"


def workspace_dir() -> Path:
    """The user's own model and study files. $NANOSCOPE_WORKSPACE wins over $NANOSCOPE_HOME."""
    explicit = os.environ.get("NANOSCOPE_WORKSPACE")
    if explicit:
        return Path(explicit)
    base = home()
    return base / "workspace" if base else Path("workspace")


def baselines_dir() -> Path:
    """Results shipped with the package; read-only."""
    return Path(__file__).parent / "baselines"
