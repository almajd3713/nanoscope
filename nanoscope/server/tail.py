"""Live events for a run, built from the files the trainer already writes.

    state       status.json changed (preparing, running, done, ...)
    step        a new row in metrics.jsonl (at most 4 a second, newest only)
    eval        a row with a validation loss        (never coalesced)
    sample      a row with generated text           (never coalesced)
    checkpoint  a new file in checkpoints/          (never coalesced)
    blockstats  a new line in blockstats.jsonl      (never coalesced)
    reset       metrics.jsonl shrank or was replaced: refetch (a resumed run rewrites it)

Because everything comes from files, a run started from the CLI streams exactly like one started
through the API, and restarting the server loses nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from nanoscope import store
from nanoscope.server import sse

MIN_STEP_INTERVAL = 0.25  # seconds: at most 4 step events a second
Event = tuple[str, dict[str, Any]]


class _Lines:
    """The complete lines appended to a file since the last read, and when it was replaced."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.inode: int | None = None

    def seek_end(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            return
        self.inode, self.offset = stat.st_ino, stat.st_size

    def read(self) -> tuple[list[str], bool]:
        """(new complete lines, reset): reset is True if the file shrank or changed identity."""
        try:
            stat = self.path.stat()
        except OSError:
            return [], False
        reset = False
        if self.inode is not None and (stat.st_ino != self.inode or stat.st_size < self.offset):
            reset, self.offset = True, 0
        self.inode = stat.st_ino
        if stat.st_size <= self.offset:
            return [], reset
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        end = data.rfind(b"\n") + 1  # a line still being written is left for next time
        self.offset += end
        return [ln for ln in data[:end].decode("utf-8", errors="replace").splitlines()
                if ln.strip()], reset


def _json(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class Tail:
    """Follow one run. Call `poll()` to get what happened since the last call."""

    def __init__(self, run_dir: Path, ref: str | None = None, *, since_step: int | None = None,
                 min_interval: float = MIN_STEP_INTERVAL,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.run_dir = run_dir
        self.ref = ref or store.ref_of(run_dir)
        self.min_interval, self.clock = min_interval, clock
        self.metrics = _Lines(run_dir / "metrics.jsonl")
        self.blockstats = _Lines(run_dir / "blockstats.jsonl")
        self._since = since_step
        self._status: str | None = None
        self._checkpoints: set[str] | None = None
        self._pending: dict[str, Any] | None = None  # the newest step row, not yet sent
        self._last_step_sent = float("-inf")
        self._started = False

    def _tag(self, kind: str, data: dict[str, Any]) -> Event:
        return kind, {"ref": self.ref, **data}

    def poll(self) -> list[Event]:
        events: list[Event] = []
        if not self._started:
            self._started = True
            if self._since is None:  # a live view: from now on, plus the current state
                self.metrics.seek_end()
                self.blockstats.seek_end()
                self._checkpoints = set(self._checkpoint_names())
        events += self._state()
        events += self._rows()
        events += self._blockstats()
        events += self._new_checkpoints()
        events += self._flush(force=False)
        return events

    def finish(self) -> list[Event]:
        """The last pending step row, if any (call when the stream ends)."""
        return self._flush(force=True)

    def _state(self) -> list[Event]:
        path = self.run_dir / "status.json"
        try:
            text = path.read_text(encoding="utf-8")
            doc = json.loads(text)
        except (OSError, ValueError):
            return []
        marker = f"{doc.get('state')}|{doc.get('step')}|{doc.get('error')}"
        if marker == self._status:
            return []
        self._status = marker
        return [self._tag("state", {k: v for k, v in doc.items()
                                    if k in ("state", "step", "max_steps", "error", "updated_at",
                                             "device", "job_id")})]

    def _rows(self) -> list[Event]:
        lines, reset = self.metrics.read()
        events: list[Event] = []
        if reset:
            self._pending = None
            self._since = None
            events.append(self._tag("reset", {"reason": "metrics.jsonl was rewritten"}))
        for line in lines:
            row = _json(line)
            if row is None or "step" not in row:
                continue
            if self._since is not None and row["step"] <= self._since:
                continue
            if "val_loss" in row or "sample" in row:
                events += self._flush(force=True)  # keep the order: steps before their eval
                if "val_loss" in row:
                    events.append(self._tag("eval", {
                        "step": row["step"], "val_loss": row["val_loss"],
                        "val_bpb": row.get("val_bpb")}))
                if "sample" in row:
                    events.append(self._tag("sample", {"step": row["step"],
                                                       "text": row["sample"]}))
            self._pending = row  # a step event keeps only the newest row
        return events

    def _flush(self, force: bool) -> list[Event]:
        if self._pending is None:
            return []
        now = self.clock()
        if not force and now - self._last_step_sent < self.min_interval:
            return []
        row = {k: v for k, v in self._pending.items() if k != "sample"}
        self._pending, self._last_step_sent = None, now
        return [self._tag("step", row)]

    def _blockstats(self) -> list[Event]:
        lines, reset = self.blockstats.read()
        return [self._tag("blockstats", row) for ln in lines if (row := _json(ln))] if (
            lines or reset) else []

    def _checkpoint_names(self) -> list[str]:
        folder = self.run_dir / "checkpoints"
        return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.pt")) if (
            folder.exists()) else []

    def _new_checkpoints(self) -> list[Event]:
        names = self._checkpoint_names()
        known = self._checkpoints if self._checkpoints is not None else set()
        new = [n for n in names if n not in known]
        self._checkpoints = set(names)
        out = []
        for name in new:
            match = re.search(r"step_(\d+)", name)
            out.append(self._tag("checkpoint", {
                "name": name, "step": int(match.group(1)) if match else None,
                "archived": name.startswith("archive/")}))
        return out


def frames(events: list[Event]) -> list[str]:
    return [sse.frame(kind, data, str(data["step"]) if "step" in data and kind in (
        "step", "eval") else None) for kind, data in events]


async def stream(tails: Callable[[], list[Tail]], *, interval: float = 0.25,
                 stop: asyncio.Event | None = None,
                 keepalive_every: float = 15.0) -> AsyncIterator[str]:
    """SSE frames for a set of tails (re-evaluated each pass, so runs that appear are picked
    up), with a keepalive comment while nothing happens."""
    known: dict[str, Tail] = {}
    quiet_since = time.monotonic()
    while stop is None or not stop.is_set():
        for tail in tails():
            known.setdefault(tail.ref, tail)
        produced = False
        for tail in list(known.values()):
            for frame in frames(tail.poll()):
                produced = True
                yield frame
        if produced:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > keepalive_every:
            quiet_since = time.monotonic()
            yield sse.KEEPALIVE
        await asyncio.sleep(interval)
    for tail in known.values():
        for frame in frames(tail.finish()):
            yield frame


def run_dir_of(ref: str) -> Path:
    return Path(os.fspath(store.resolve(ref)))
