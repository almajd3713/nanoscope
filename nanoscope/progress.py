"""Seeing what is happening: a progress bar for one run, and a snapshot of many.

The bar works in terminals and notebooks. The snapshot reads only run folders, so it
works from another terminal, for studies running on several devices, and after a crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoscope.schemas.upgrade import read_json

RUNNING_IF_UPDATED_WITHIN = 120  # seconds, for folders without a status.json
STALE_HEARTBEAT = 60  # seconds without a heartbeat before a "running" run looks dead
STATE_ORDER = ("done", "running", "preparing", "stopped", "cancelled", "failed", "queued")


class ProgressBar:
    """An on_step hook: one bar per run, plus a printed line at every evaluation."""

    def __init__(self, total: int, desc: str) -> None:
        self.total, self.desc = total, desc
        self.bar = None

    def __call__(self, step: int, row: dict[str, Any]) -> None:
        from tqdm.auto import tqdm

        if self.bar is None:  # created on the first step, so a resumed run starts mid-way
            self.bar = tqdm(total=self.total, initial=step - 1, desc=self.desc, unit="step",
                            dynamic_ncols=True)
        self.bar.update(1)
        self.bar.set_postfix(loss=f"{row['loss']:.3f}", refresh=False)
        if "val_loss" in row:
            self.bar.write(f"  step {step:>6}  val loss {row['val_loss']:.4f}  "
                           f"({row['val_bpb']:.3f} bits per byte)")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


@dataclass
class RunState:
    run_dir: Path
    step: int
    max_steps: int
    val_bpb: float | None
    updated: float  # seconds since the last metrics row
    status: str | None = None  # the state in status.json, when the run has one
    error: dict[str, Any] | None = None
    heartbeat_age: float | None = None  # seconds since status.json was last refreshed

    @property
    def source(self) -> str:
        """Where `state` comes from: status.json, or (older runs) file times."""
        return "status" if self.status else "files"

    @property
    def stale(self) -> bool:
        """Says running, but nothing has refreshed status.json lately: the process is gone."""
        return (self.state == "running" and self.heartbeat_age is not None
                and self.heartbeat_age > STALE_HEARTBEAT)

    @property
    def state(self) -> str:
        if self.status:
            return self.status
        if self.step >= self.max_steps:
            return "done"
        if self.step == 0:
            return "queued"
        return "running" if self.updated < RUNNING_IF_UPDATED_WITHIN else "stopped"


def _last_line(path: Path, block: int = 4096) -> str | None:
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - block))
        lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else None


def _with_status(state: RunState, now: float) -> RunState:
    """Add what status.json says, when the run has one."""
    path = state.run_dir / "status.json"
    if path.exists():
        doc = read_json(path, "status")
        beat = doc.get("heartbeat_at") or doc["updated_at"]
        state.status, state.error = doc["state"], doc.get("error")
        state.heartbeat_age = max(0.0, now - datetime.fromisoformat(beat).timestamp())
    return state


def snapshot(root: str | Path) -> list[RunState]:
    """The state of every run under root, from its config.json and metrics.jsonl."""
    states = []
    now = time.time()
    for config_path in sorted(Path(root).glob("**/config.json")):
        run_dir = config_path.parent
        config = read_json(config_path, "config")
        metrics = run_dir / "metrics.jsonl"
        step, updated, bpb = 0, float("inf"), None
        if metrics.exists() and metrics.stat().st_size:
            updated = now - metrics.stat().st_mtime
            line = _last_line(metrics)
            step = json.loads(line)["step"] if line else 0
            bpb = _latest_bpb(metrics)
        states.append(_with_status(
            RunState(run_dir, step, config["preset"]["max_steps"], bpb, updated), now))
    # A run that is still preparing (or failed doing so) has a status.json but no config yet.
    listed = {s.run_dir.resolve() for s in states}
    for status_path in sorted(Path(root).glob("**/status.json")):
        if status_path.parent.resolve() not in listed:
            doc = read_json(status_path, "status")
            states.append(_with_status(
                RunState(status_path.parent, doc.get("step", 0), doc.get("max_steps") or 0, None,
                         float("inf")), now))
    # Studies list their planned runs; show the ones that haven't started as queued.
    seen = {s.run_dir.resolve() for s in states}
    for plan_path in sorted(Path(root).glob("**/plan.json")):
        plan = read_json(plan_path, "plan")
        for job in plan["runs"]:
            run_dir = plan_path.parent / job["dir"]
            if run_dir.resolve() not in seen:
                states.append(RunState(run_dir, 0, job["max_steps"], None, float("inf")))
    return states


def _latest_bpb(metrics: Path) -> float | None:
    for line in reversed(metrics.read_text(encoding="utf-8").splitlines()):
        if '"val_bpb"' in line:
            return json.loads(line)["val_bpb"]
    return None


def _ago(seconds: float) -> str:
    if seconds == float("inf"):
        return "-"
    if seconds < 120:
        return f"{seconds:.0f}s ago"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.0f}h ago"


def _state_cell(r: RunState) -> str:
    if r.stale and r.heartbeat_age is not None:
        quiet = _ago(r.heartbeat_age).removesuffix(" ago")
        return f"{r.state} (no heartbeat for {quiet})"
    return r.state


def format_snapshot(states: list[RunState], root: str | Path) -> str:
    if not states:
        return f"no runs under {root}"
    counts = {s: sum(1 for r in states if r.state == s) for s in STATE_ORDER}
    summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    rows = [["run", "state", "step", "val bpb", "updated"]]
    for r in states:
        rows.append([
            str(r.run_dir.relative_to(root)) if r.run_dir.is_relative_to(root) else str(r.run_dir),
            _state_cell(r),
            f"{r.step}/{r.max_steps} ({r.step / r.max_steps:.0%})" if r.max_steps else str(r.step),
            f"{r.val_bpb:.3f}" if r.val_bpb is not None else "-",
            _ago(r.updated),
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
             for row in rows]
    errors = [f"{r.run_dir.relative_to(root) if r.run_dir.is_relative_to(root) else r.run_dir} "
              f"failed: {r.error['type']}: {r.error['message']}"
              for r in states if r.state == "failed" and r.error]
    out = [f"{len(states)} runs under {root}: {summary}", "", *lines]
    return "\n".join(out + ([""] + errors if errors else []))


def one_line(states: list[RunState], total: int | None = None) -> str:
    """Compact progress for a study: '9/24 done · running gpt2/seed-1 1092/1954'."""
    done = sum(1 for r in states if r.state == "done")
    total = total if total is not None else len(states)
    running = [f"{r.run_dir.parent.name}/{r.run_dir.name} {r.step}/{r.max_steps}"
               for r in states if r.state == "running"]
    return f"{done}/{total} done" + (f" · running {', '.join(running)}" if running else "")
