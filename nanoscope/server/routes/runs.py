from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nanoscope import paths, queue, store
from nanoscope.progress import RunState
from nanoscope.schemas.upgrade import read_json
from nanoscope.server.errors import problem
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import ConfigDoc, JobDoc, StatusDoc
from nanoscope.server.routes.models import resolve_ref
from nanoscope.server.routes.validate import model_spec

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


class RunRequest(BaseModel):
    """Train one seed of a model on a preset. Every field but `model` has the library's
    default, so `{"model": "bigram"}` is the level-0 request."""

    model: str
    preset: str = "tinystories-5min"
    seed: int = 0
    kwargs: dict[str, Any] = {}  # model parameters, and preset fields to override
    compile: bool | str = False
    push_to_hub: str | None = None
    wandb: bool | str = False


class RunSubmission(BaseModel):
    ref: str
    state: str  # "done" (nothing to do), or the job's state
    run: RunDetail | None = None  # when the run already exists and is done
    job: JobDoc | None = None  # otherwise: the job that trains it


@router.post("/runs", status_code=202)
def submit_run(body: RunRequest, request: Request) -> Any:
    """Queue a run on the interactive lane. A run that is already done returns it (200); one
    already queued or running returns its job (202). The model is read, never imported here:
    the worker runs it."""
    from dataclasses import fields as dataclass_fields

    from nanoscope.presets import Preset, get_preset
    from nanoscope.runref import REQUIRED, run_ref
    from nanoscope.specs import validate_run_spec

    spec, locked = model_spec(body.model)
    problems = validate_run_spec(spec, body.preset, body.kwargs, None, locked)
    if problems:
        first = problems[0]
        lock = next((p for p in problems if p.code == "locked"), None)
        extra: dict[str, Any] = {"problems": [p.to_dict() for p in problems]}
        if lock is not None:
            lessons = list(dict.fromkeys((p.hint or "").rpartition(" ")[2]
                                         for p in problems if p.code == "locked"))
            extra.update(lesson=lessons[0], lessons=lessons)
        return problem(422, first.message, request, code="locked" if lock else "about:blank",
                       title="Locked until you build it" if lock else None, **extra)
    tunable = {p.name for p in spec.tunable()}
    preset_names = {f.name for f in dataclass_fields(Preset)}
    model_kwargs = {k: v for k, v in body.kwargs.items() if k in tunable}
    overrides = {k: v for k, v in body.kwargs.items() if k not in tunable and k in preset_names}
    given = get_preset(body.preset)
    preset = given.override(**overrides)
    defaults = {p.name: REQUIRED if p.required else p.default for p in spec.params
                if not p.from_data}
    ref = run_ref(spec.name, defaults, model_kwargs, given, preset, body.seed)
    payload: dict[str, Any] = {
        "model": resolve_ref(body.model), "preset": body.preset, "seed": body.seed,
        "kwargs": model_kwargs, "compile": body.compile, "wandb": body.wandb}
    if overrides:
        payload["overrides"] = overrides
    if body.push_to_hub:
        payload["push_to_hub"] = body.push_to_hub
    job_id = queue.enqueue("run", payload, lane="interactive", ref=ref)
    if job_id is None:  # the library says this run is already done
        return JSONResponse(
            RunSubmission(ref=ref, state="done", run=get_run(ref)).model_dump(mode="json"),
            status_code=200)
    job = job_doc(queue.get(job_id))
    return RunSubmission(ref=ref, state=job.state, job=job)


class StopResult(BaseModel):
    ref: str
    stopping: list[str]  # runs that were asked to stop (they save a checkpoint and can resume)
    cancelled_jobs: list[int]


@router.post("/runs/{ref:path}/stop")
def stop_run(ref: str) -> StopResult:
    """Ask a run to stop at its next step (it saves a checkpoint, and `resume` continues it);
    a job that has not started is cancelled. Works for runs started outside the API too."""
    run_dir = store.resolve(ref, must_exist=False)
    cancelled = queue.cancel_prefix(ref) if paths.queue_db().exists() else []
    stopping = store.request_stop(ref) if run_dir.exists() else []
    if not cancelled and not run_dir.exists():
        raise FileNotFoundError(f"no run or queued job at ref {ref!r}")
    return StopResult(ref=store.ref_of(run_dir), stopping=stopping, cancelled_jobs=cancelled)


@router.post("/runs/{ref:path}/resume", status_code=202)
def resume_run(ref: str, request: Request) -> Any:
    """Continue a stopped or interrupted run from its last checkpoint, with exactly the
    configuration it started with. A finished run has nothing to resume (409)."""
    from nanoscope.specs import FROM_DATA

    run_dir = store.resolve(ref)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{ref} has no config.json: it never started, submit it instead")
    config = read_json(config_path, "config")
    status = run_dir / "status.json"
    state = read_json(status, "status")["state"] if status.exists() else None
    if state == "done":
        return problem(409, f"{ref} is finished: there is nothing to resume", request)
    model = config["model"]
    if not model.get("rebuildable", True) or "ref" not in model:
        return problem(409, f"{ref} was trained from a class a worker cannot import again; "
                       "its source is saved next to the run", request)
    payload: dict[str, Any] = {
        "model": model["ref"], "preset": config["preset"]["name"],
        "custom_preset": config["preset"], "seed": config["seed"],
        "kwargs": {k: v for k, v in model.get("kwargs", {}).items() if k not in FROM_DATA}}
    job_id = queue.enqueue("run", payload, lane="interactive", ref=store.ref_of(run_dir))
    assert job_id is not None
    job = job_doc(queue.get(job_id))
    return RunSubmission(ref=store.ref_of(run_dir), state=job.state, job=job)


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
