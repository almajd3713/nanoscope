"""Data preparation you can watch: `data_dir/<preset>/prepare.json`.

Downloading tokens, training a tokenizer and tokenizing a split can each take minutes. While
they run, `prepare.json` says which stage is under way and how far along it is, so another
terminal, `nanoscope status --data` and the API can show it. Nothing is written when the data
is already prepared.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths
from nanoscope.fsutil import write_json_atomic
from nanoscope.schemas.upgrade import read_json


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PrepareFile:
    def __init__(self, preset_name: str) -> None:
        self.path = paths.data_dir() / preset_name / "prepare.json"
        self.doc: dict[str, Any] = {
            "schema": 1, "nanoscope": __version__, "preset": preset_name, "stage": "download",
            "label": None, "done": None, "total": None, "started_at": _stamp(),
            "updated_at": _stamp(), "error": None,
        }
        self._last = float("-inf")
        self.touched = False  # did this process do any preparing?

    def stage(self, stage: str, label: str | None = None, done: int | None = None,
              total: int | None = None, *, force: bool = True) -> None:
        """Move to a stage, or report progress within it (throttled unless force=True)."""
        self.touched = True
        self.doc.update(stage=stage, label=label, done=done, total=total)
        if force or time.monotonic() - self._last >= 1.0:
            self._write()

    def finish(self) -> None:
        if self.touched:
            self.doc.update(stage="done", error=None)
            self._write()

    def fail(self, exc: BaseException) -> None:
        if self.touched:
            self.doc.update(stage="failed", error={"type": type(exc).__name__,
                                                   "message": str(exc)})
            self._write()

    def _write(self) -> None:
        self.doc["updated_at"] = _stamp()
        write_json_atomic(self.path, self.doc)
        self._last = time.monotonic()


def read_all(root: Path | None = None) -> list[dict[str, Any]]:
    """Every preset's prepare.json under the data folder."""
    root = root or paths.data_dir()
    return [read_json(p, "prepare") for p in sorted(root.glob("*/prepare.json"))]


def format_prepare(docs: list[dict[str, Any]], root: Path | None = None) -> str:
    if not docs:
        return f"no data preparation recorded under {root or paths.data_dir()}"
    lines = []
    for d in docs:
        progress = f" {d['done']:,}/{d['total']:,}" if d.get("total") else (
            f" {d['done']:,}" if d.get("done") else "")
        what = f" ({d['label']})" if d.get("label") else ""
        error = f" {d['error']['type']}: {d['error']['message']}" if d.get("error") else ""
        lines.append(f"{d['preset']}  {d['stage']}{what}{progress}  updated {d['updated_at']}"
                     f"{error}")
    return "\n".join(lines)
