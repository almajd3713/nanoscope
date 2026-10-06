from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope import store
from nanoscope.progress import RunState
from nanoscope.schemas.upgrade import read_json
from nanoscope.server.models import ConfigDoc, StatusDoc

router = APIRouter(prefix="/api", tags=["runs"])

class RunEntry(BaseModel):
    """One line of the run table."""

    ref: str
    state: str
    step: int
    max_steps: int
    val_bpb: float | None
    updated: float  # seconds since the last metrics row
    stale: bool = False  # says running, but nothing has refreshed status.json lately
    error: dict[str, Any] | None = None


class RunDetail(BaseModel):
    ref: str
    config: ConfigDoc | None
    status: StatusDoc | None
    summary: dict[str, Any]


def entry(run: RunState) -> RunEntry:
    return RunEntry(ref=store.ref_of(run.run_dir), state=run.state, step=run.step,
                    max_steps=run.max_steps, val_bpb=run.val_bpb, updated=run.updated,
                    stale=run.stale, error=run.error)


def read_metrics(run_dir: Path, since_step: int = 0) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a line being written right now
            if row.get("step", 0) > since_step:
                rows.append(row)
    return rows


def summary_of(run_dir: Path) -> dict[str, Any]:
    rows = read_metrics(run_dir)
    evals = [r for r in rows if "val_loss" in r]
    last = rows[-1] if rows else {}
    return {"final_step": last.get("step", 0), "final_train_loss": last.get("loss"),
            "final_val_loss": evals[-1]["val_loss"] if evals else None,
            "final_val_bpb": evals[-1].get("val_bpb") if evals else None,
            "n_evals": len(evals)}


@router.get("/runs")
def list_runs(prefix: str = "", state: str | None = None) -> list[RunEntry]:
    """Every run under a ref prefix (all when empty), optionally only those in one state."""
    return [entry(r) for r in store.list_runs(prefix, state)]


class MetricsPage(BaseModel):
    rows: list[dict[str, Any]]
    last_step: int  # ask again with since_step=last_step for what came after


class Sample(BaseModel):
    step: int
    text: str


class CheckpointEntry(BaseModel):
    name: str
    step: int | None
    bytes: int
    archived: bool  # kept for good by checkpoint_steps, never pruned
    latest: bool


@router.get("/runs/{ref:path}/metrics")
def run_metrics(ref: str, since_step: int = 0) -> MetricsPage:
    """The rows of metrics.jsonl after `since_step` (all of them by default)."""
    rows = read_metrics(store.resolve(ref), since_step)
    return MetricsPage(rows=rows, last_step=rows[-1]["step"] if rows else since_step)


@router.get("/runs/{ref:path}/samples")
def run_samples(ref: str) -> list[Sample]:
    """The generated samples, one per `sample_interval`, oldest first."""
    path = store.resolve(ref) / "samples.txt"
    if not path.exists():
        return []
    out = []
    parts = re.split(r"^--- step (\d+) ---\n", path.read_text(encoding="utf-8", errors="replace"),
                     flags=re.M)
    for step, text in zip(parts[1::2], parts[2::2], strict=True):
        out.append(Sample(step=int(step), text=text.rstrip("\n")))
    return out


@router.get("/runs/{ref:path}/blockstats")
def run_blockstats(ref: str) -> list[dict[str, Any]]:
    """Per-block statistics recorded at each eval step (runs started with block_stats=True)."""
    path = store.resolve(ref) / "blockstats.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


@router.get("/runs/{ref:path}/checkpoints")
def run_checkpoints(ref: str) -> list[CheckpointEntry]:
    """The checkpoint files of a run (the weights are not served)."""
    folder = store.resolve(ref) / "checkpoints"
    if not folder.exists():
        return []
    latest = None
    latest_file = store.resolve(ref) / "latest.json"
    if latest_file.exists():
        with contextlib.suppress(ValueError):
            latest = json.loads(latest_file.read_text(encoding="utf-8")).get("checkpoint")
    out = []
    for path in sorted(folder.rglob("*.pt")):
        match = re.search(r"step_(\d+)", path.name)
        out.append(CheckpointEntry(
            name=path.relative_to(folder).as_posix(), step=int(match.group(1)) if match else None,
            bytes=path.stat().st_size, archived="archive" in path.parts,
            latest=path.name == latest))
    return out


@router.get("/runs/{ref:path}", response_model=RunDetail)
def get_run(ref: str) -> RunDetail:
    """A run's config, status and a summary of its metrics."""
    run_dir = store.resolve(ref)
    config = read_json(run_dir / "config.json", "config") if (
        run_dir / "config.json").exists() else None
    status = read_json(run_dir / "status.json", "status") if (
        run_dir / "status.json").exists() else None
    if config is None and status is None:
        raise FileNotFoundError(f"{ref} holds no run (no config.json or status.json)")
    return RunDetail(
        ref=ref, config=ConfigDoc.model_validate(config) if config else None,
        status=StatusDoc.model_validate(status) if status else None,
        summary=summary_of(run_dir))
