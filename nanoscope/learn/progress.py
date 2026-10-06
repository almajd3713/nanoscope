"""Where a learner is: one entry per lesson in `learn/progress.json`.

    mark("foundations/01-bigram", "started")
    state("foundations/01-bigram")        # "started"

Written atomically, so `nanoscope learn list`, the API and the GUI can read it while a check
runs. States: not-started, started, checking, passed, failed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths
from nanoscope.fsutil import write_json_atomic
from nanoscope.schemas.upgrade import upgrade

STATES = ("not-started", "started", "checking", "passed", "failed")
FILE = "progress.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def progress_path(owner: str = "local") -> Path:
    return paths.learn_dir(owner) / FILE


def read(owner: str = "local") -> dict[str, Any]:
    """The whole progress document (an empty one if nothing has been recorded yet)."""
    path = progress_path(owner)
    if not path.exists():
        return {"schema": 1, "nanoscope": __version__, "lessons": {}}
    return upgrade(json.loads(path.read_text(encoding="utf-8")), "progress", path)


def entry(lesson_id: str, owner: str = "local") -> dict[str, Any]:
    """A lesson's record: state, attempts, last_check and timestamps."""
    found = read(owner)["lessons"].get(lesson_id)
    return dict(found) if found else {
        "state": "not-started", "attempts": 0, "last_check": None, "started_at": None,
        "first_checked_at": None, "updated_at": None, "passed_at": None}


def state(lesson_id: str, owner: str = "local") -> str:
    return entry(lesson_id, owner)["state"]


def mark(lesson_id: str, new_state: str, *, check_id: str | None = None,
         owner: str = "local") -> dict[str, Any]:
    """Move a lesson to `new_state` and save. Entering `checking` counts an attempt;
    `passed` stamps the time; a lesson that has passed stays passed if a later check fails
    (what was earned is not taken back)."""
    if new_state not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}, got {new_state!r}")
    doc = read(owner)
    current = entry(lesson_id, owner)
    stamp = now()
    if new_state == "checking":
        current["attempts"] += 1
        current["first_checked_at"] = current.get("first_checked_at") or stamp
    if new_state in ("started", "checking") and current["started_at"] is None:
        current["started_at"] = stamp
    if new_state == "passed":
        current["passed_at"] = current["passed_at"] or stamp
    if check_id is not None:
        current["last_check"] = check_id
    if not (current["state"] == "passed" and new_state in ("failed", "started", "checking")):
        current["state"] = new_state
    elif new_state == "checking":
        current["state"] = "passed"  # re-checking a passed lesson keeps it passed
    current["updated_at"] = stamp
    doc["lessons"][lesson_id] = current
    doc["nanoscope"] = __version__
    write_json_atomic(progress_path(owner), doc)
    return current


def record_prediction(lesson_id: str, file: Path, sha256: str, owner: str = "local") -> str:
    """Remember that `file` was committed to as a prediction now, and what it contained."""
    doc = read(owner)
    stamp = now()
    doc.setdefault("predictions", {})[lesson_id] = {"at": stamp, "file": str(file),
                                                     "sha256": sha256}
    doc["nanoscope"] = __version__
    write_json_atomic(progress_path(owner), doc)
    return stamp


def prediction(lesson_id: str, owner: str = "local") -> dict[str, Any] | None:
    found = read(owner).get("predictions", {}).get(lesson_id)
    return dict(found) if found else None
