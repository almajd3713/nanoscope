"""Where nanoscope keeps things on disk.

With $NANOSCOPE_HOME set, everything lives under it (containers set it to /nanoscope).
Without it the defaults are today's: ./runs, ./experiments and ~/.nanoscope/data.
Every function reads the environment when called, never at import time.
"""

from __future__ import annotations

import os
import re
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


def learn_dir(owner: str = "local") -> Path:
    """Lesson progress and unlocks. The local single user keeps them in `learn/`; a hosted
    deployment passes each user's name and gets `users/<name>/learn/`."""
    base = home() or _USER_HOME
    if owner == "local":
        return base / "learn"
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", owner):
        raise ValueError(f"owner {owner!r} must be letters, digits, '_', '-' or '.'")
    return base / "users" / owner / "learn"


def hardware_dir() -> Path:
    """Bench results and other facts about this machine."""
    base = home()
    return base / "hardware" if base else _USER_HOME / "hardware"


def server_dir() -> Path:
    """The API server's own files (its access token)."""
    base = home()
    return base / "server" if base else _USER_HOME / "server"


def workspace_dir() -> Path:
    """The user's own model and study files. $NANOSCOPE_WORKSPACE wins over $NANOSCOPE_HOME."""
    explicit = os.environ.get("NANOSCOPE_WORKSPACE")
    if explicit:
        return Path(explicit)
    base = home()
    return base / "workspace" if base else Path("workspace")


def queue_db() -> Path:
    """The job queue (SQLite)."""
    base = home()
    return base / "queue.db" if base else _USER_HOME / "queue.db"


def workers_dir() -> Path:
    """One small JSON file per live worker (device, slots, current jobs, heartbeat)."""
    base = home()
    return base / "workers" if base else _USER_HOME / "workers"


def job_logs_dir() -> Path:
    """The output of each job's process, as <id>.log."""
    base = home()
    return base / "jobs" if base else _USER_HOME / "jobs"


def baselines_dir() -> Path:
    """Results shipped with the package; read-only."""
    return Path(__file__).parent / "baselines"
