"""Seeing what is happening: a progress bar for one run, and a snapshot of many.

The bar works in terminals and notebooks. The snapshot reads only run folders, so it
works from another terminal, for studies running on several devices, and after a crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNNING_IF_UPDATED_WITHIN = 120  # seconds


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

    @property
    def state(self) -> str:
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


def snapshot(root: str | Path) -> list[RunState]:
    """The state of every run under root, from its config.json and metrics.jsonl."""
    states = []
    now = time.time()
    for config_path in sorted(Path(root).glob("**/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = run_dir / "metrics.jsonl"
        step, updated, bpb = 0, float("inf"), None
        if metrics.exists() and metrics.stat().st_size:
            updated = now - metrics.stat().st_mtime
            line = _last_line(metrics)
            step = json.loads(line)["step"] if line else 0
            bpb = _latest_bpb(metrics)
        states.append(RunState(run_dir, step, config["preset"]["max_steps"], bpb, updated))
    # Studies list their planned runs; show the ones that haven't started as queued.
    seen = {s.run_dir.resolve() for s in states}
    for plan_path in sorted(Path(root).glob("**/plan.json")):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
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


def format_snapshot(states: list[RunState], root: str | Path) -> str:
    if not states:
        return f"no runs under {root}"
    counts = {s: sum(1 for r in states if r.state == s)
              for s in ("done", "running", "stopped", "queued")}
    summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    rows = [["run", "state", "step", "val bpb", "updated"]]
    for r in states:
        rows.append([
            str(r.run_dir.relative_to(root)) if r.run_dir.is_relative_to(root) else str(r.run_dir),
            r.state,
            f"{r.step}/{r.max_steps} ({r.step / r.max_steps:.0%})",
            f"{r.val_bpb:.3f}" if r.val_bpb is not None else "-",
            _ago(r.updated),
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
             for row in rows]
    return "\n".join([f"{len(states)} runs under {root}: {summary}", "", *lines])


def one_line(states: list[RunState], total: int | None = None) -> str:
    """Compact progress for a study: '9/24 done · running gpt2/seed-1 1092/1954'."""
    done = sum(1 for r in states if r.state == "done")
    total = total if total is not None else len(states)
    running = [f"{r.run_dir.parent.name}/{r.run_dir.name} {r.step}/{r.max_steps}"
               for r in states if r.state == "running"]
    return f"{done}/{total} done" + (f" · running {', '.join(running)}" if running else "")
