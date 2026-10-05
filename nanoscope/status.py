"""A run's status.json: its lifecycle state, readable from outside the process.

Written atomically (tmp file, then replace), so a reader never sees half a file. States:
queued, preparing, running, done, stopped (Ctrl-C), cancelled (a STOP file), failed.
`nanoscope status` and the API read this instead of guessing from file times.
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HEARTBEAT_SECONDS = 5
STATES = ("queued", "preparing", "running", "done", "stopped", "cancelled", "failed")
FINISHED = ("done", "stopped", "cancelled", "failed")


def error_info(exc: BaseException) -> dict[str, Any]:
    """The exception as status.json records it: type, message, last 20 traceback lines."""
    lines = "".join(traceback.format_exception(exc)).splitlines()
    return {"type": type(exc).__name__, "message": str(exc), "traceback": lines[-20:]}


class StatusFile:
    """Writes `status.json` in a run folder; also an `on_step` hook that keeps the heartbeat."""

    def __init__(self, run_dir: Path, max_steps: int | None, device: str | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(run_dir) / "status.json"
        self._previous = self.path.read_bytes() if self.path.exists() else None
        self._clock = clock
        self._last_beat = float("-inf")
        self.doc: dict[str, Any] = {
            "schema": 1,
            "state": "queued",
            "error": None,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "device": device,
            "job_id": os.environ.get("NANOSCOPE_JOB_ID"),
            "step": 0,
            "max_steps": max_steps,
            "started_at": self._now(),
            "updated_at": self._now(),
            "heartbeat_at": None,
        }

    def _now(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).isoformat(timespec="seconds")

    def write(self, state: str, step: int | None = None,
              error: dict[str, Any] | None = None) -> None:
        if state not in STATES:
            raise ValueError(f"unknown run state {state!r}")
        self.doc["state"] = state
        if step is not None:
            self.doc["step"] = step
        if error is not None:
            self.doc["error"] = error
        self.doc["updated_at"] = self.doc["heartbeat_at"] = self._now()
        self._last_beat = self._clock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.doc, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def __call__(self, step: int, row: dict[str, Any]) -> None:
        """on_step hook: `running` from the first step, then a heartbeat every few seconds."""
        first = self.doc["state"] != "running"
        if first or self._clock() - self._last_beat >= HEARTBEAT_SECONDS:
            self.write("running", step=step)
        else:
            self.doc["step"] = step

    def restore(self) -> None:
        """Put back what status.json held before this writer touched it (a refused run)."""
        if self._previous is None:
            self.path.unlink(missing_ok=True)
        else:
            self.path.write_bytes(self._previous)

    def fail(self, exc: BaseException) -> None:
        """Ctrl-C before training is `stopped`; anything else is `failed` with the error."""
        if isinstance(exc, KeyboardInterrupt):
            self.write("stopped", error=None)
        else:
            self.write("failed", error=error_info(exc))
