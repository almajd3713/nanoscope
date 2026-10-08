from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nanoscope import paths, queue
from nanoscope.learn import gating, loader, progress, unlocks
from nanoscope.learn.cli import locked_by, start_lesson, workspace_lesson_dir
from nanoscope.learn.loader import LessonSpec
from nanoscope.server import sse
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import JobDoc, ProgressDoc

router = APIRouter(prefix="/api", tags=["learn"])


class ComputeDoc(BaseModel):
    preset: str | None
    budget: str | None
    estimate_minutes: float


class LessonSummary(BaseModel):
    id: str
    slug: str
    title: str
    summary: str
    level: int
    state: str  # not-started, started, checking, passed, failed
    locked_by: list[str]  # prerequisite lessons not passed yet
    unlocks: list[str]
    compute: dict[str, ComputeDoc]


class PathSummary(BaseModel):
    id: str
    title: str
    level: int
    summary: str
    prerequisites: list[str]
    compute: dict[str, ComputeDoc]
    lessons: list[LessonSummary]


class LessonDetail(LessonSummary):
    prerequisites: list[str]
    forbid: list[str]
    checks: list[dict[str, Any]]
    experiment: dict[str, Any]
    depth: dict[str, dict[str, Any]]
    text: dict[str, str]  # intro, surface, deep, reading (Markdown)
    files: list[str]  # the starter files the lesson ships
    workspace: str  # where `start` puts them, relative to the workspace


def _compute(compute: dict[str, Any]) -> dict[str, ComputeDoc]:
    return {k: ComputeDoc(preset=c.preset, budget=c.budget, estimate_minutes=c.estimate_minutes)
            for k, c in compute.items()}


def _summary(lesson: LessonSpec) -> LessonSummary:
    return LessonSummary(
        id=lesson.id, slug=lesson.slug, title=lesson.title, summary=lesson.summary,
        level=lesson.level, state=progress.state(lesson.id), locked_by=locked_by(lesson),
        unlocks=lesson.unlocks, compute=_compute(lesson.compute))


@router.get("/curricula")
def curricula() -> list[PathSummary]:
    """Every learning path with its lessons, state and compute estimates."""
    out = []
    for path_id in loader.list_path_ids():
        path = loader.load_path(path_id)  # a broken lesson is a 422 with every problem listed
        out.append(PathSummary(
            id=path.id, title=path.title, level=path.level, summary=path.summary,
            prerequisites=path.prerequisites, compute=_compute(path.compute),
            lessons=[_summary(lesson) for lesson in path.lessons]))
    return out


@router.get("/curricula/{path}/{lesson}")
def lesson_detail(path: str, lesson: str) -> LessonDetail:
    """One lesson: its text (Surface, Deep, Reading), checks, unlocks and starter files."""
    spec = loader.load_lesson(f"{path}/{lesson}")
    files = [n for n in ("starter.py", "notebook.py") if (spec.dir / n).exists()]
    return LessonDetail(
        **_summary(spec).model_dump(), prerequisites=spec.prerequisites, forbid=spec.forbid,
        checks=[{"id": c.id, "kind": c.kind, **c.args} for c in spec.checks],
        experiment=spec.experiment, depth=spec.depth,
        text={"intro": spec.text.intro, "surface": spec.text.surface, "deep": spec.text.deep,
              "reading": spec.text.reading},
        files=files,
        workspace=workspace_lesson_dir(spec).relative_to(paths.workspace_dir()).as_posix())


class StartRequest(BaseModel):
    gating: str | None = None  # "guided" or "open": only used the very first time


class StartResult(BaseModel):
    lesson: str
    workspace: str
    copied: list[str]
    kept: list[str]  # files you had already edited: never overwritten
    state: str
    policy: str
    first_start: bool


@router.post("/curricula/{path}/{lesson}/start")
def start(path: str, lesson: str, body: StartRequest | None = None) -> StartResult:
    """Copy the lesson's starter files into your workspace (never over your edits) and mark it
    started. The very first start turns gating on (`guided`) unless you ask for `open`."""
    spec = loader.load_lesson(f"{path}/{lesson}")
    first = not unlocks.exists()
    if first:
        choice = (body.gating if body and body.gating else "guided")
        unlocks.set_policy(choice)  # a ValueError for anything else: a 422
    folder, copied, kept = start_lesson(spec)
    root = paths.workspace_dir()
    return StartResult(
        lesson=spec.id, workspace=folder.relative_to(root).as_posix(),
        copied=[p.relative_to(root).as_posix() for p in copied],
        kept=[p.relative_to(root).as_posix() for p in kept], state=progress.state(spec.id),
        policy=unlocks.policy(), first_start=first)


class CheckRequest(BaseModel):
    variant: str = "cpu"


@router.post("/curricula/{path}/{lesson}/check", status_code=202)
def check(path: str, lesson: str, body: CheckRequest | None = None) -> JobDoc:
    """Run the lesson's checks on your file as a job (they train models and run your code, so
    a worker does it). The result lands in `learn/checks/` and in `/learn/progress`."""
    spec = loader.load_lesson(f"{path}/{lesson}")
    variant = body.variant if body else "cpu"
    if variant not in spec.compute:
        raise ValueError(f"{spec.id} has no {variant} variant; it has: {', '.join(spec.compute)}")
    job_id = queue.enqueue("check", {"lesson": spec.id, "variant": variant}, lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))


class PredictRequest(BaseModel):
    verdict: str | None = None  # better, worse or within noise
    low: float | None = None  # the interval you expect for the difference
    high: float | None = None
    note: str | None = None


class PredictResult(BaseModel):
    lesson: str
    at: str
    file: str  # the prediction.toml in your workspace, relative to it


@router.post("/curricula/{path}/{lesson}/predict")
def predict(path: str, lesson: str, body: PredictRequest | None = None) -> PredictResult:
    """Commit to a prediction before the experiment runs (what `nanoscope learn predict` does).
    Once the lesson has been checked it is too late: a 422 says so."""
    from nanoscope.learn.checks import commit_prediction

    spec = loader.load_lesson(f"{path}/{lesson}")
    given = body.model_dump(exclude_none=True) if body else {}
    file, stamp = commit_prediction(spec, given)
    return PredictResult(lesson=spec.id, at=stamp,
                         file=file.relative_to(paths.workspace_dir()).as_posix())


@router.get("/learn/progress")
def learn_progress() -> ProgressDoc:
    """Where you are in each lesson (progress.json)."""
    return ProgressDoc.model_validate(progress.read())


class UnlocksView(BaseModel):
    """The unlocks document plus every lockable id with its current state."""

    schema_version: int
    policy: str
    first_run: bool  # no unlocks.json yet: nobody has chosen guided or open
    unlocks: dict[str, dict[str, Any]]
    lockable: dict[str, dict[str, Any]]  # id -> {lesson, state: earned|skipped|open|locked}


def _view() -> UnlocksView:
    doc = unlocks.read()
    table = gating.lock_table()
    lockable: dict[str, dict[str, Any]] = {}
    for unlock_id, lesson in sorted(table.items()):
        entry = doc["unlocks"].get(unlock_id)
        state = entry["how"] if entry else ("open" if doc["policy"] == "open" else "locked")
        lockable[unlock_id] = {"lesson": lesson, "state": state,
                               "reason": entry.get("reason") if entry else None}
    return UnlocksView(schema_version=doc["schema"], policy=doc["policy"],
                       first_run=not unlocks.exists(), unlocks=doc["unlocks"], lockable=lockable)


@router.get("/learn/unlocks")
def learn_unlocks() -> UnlocksView:
    """The gating policy, what you earned or skipped, and what is still locked."""
    return _view()


class UnlockRequest(BaseModel):
    id: str | None = None  # "block:Attention", "feature:gqa" (or a bare name)
    all: bool = False
    reason: str | None = None  # why you may skip that lesson (required with an id)


@router.post("/learn/unlock")
def learn_unlock(body: UnlockRequest) -> UnlocksView:
    """Unlock everything (`all`), or skip one lesson's lock with a reason. Both are recorded, so
    your progress shows what you earned and what you skipped."""
    table = gating.lock_table()
    if body.all:
        unlocks.open_everything(sorted(table))
        return _view()
    if not body.id:
        raise ValueError("name what to unlock (id), or send all: true")
    if not body.reason or not body.reason.strip():
        raise ValueError('skipping a lesson needs a reason, e.g. "I know this"')
    unlock_id = next((c for c in (body.id, f"block:{body.id}", f"feature:{body.id}")
                      if c in table), None)
    if unlock_id is None:
        raise ValueError(f"{body.id!r} is not locked by any lesson "
                         f"(lockable: {', '.join(sorted(table)) or 'none'})")
    if not unlocks.exists():
        unlocks.set_policy("guided")
    unlocks.grant(unlock_id, "skipped", table[unlock_id], reason=body.reason.strip())
    return _view()


class PolicyRequest(BaseModel):
    policy: str  # "guided" or "open"


@router.post("/learn/policy")
def learn_policy(body: PolicyRequest) -> UnlocksView:
    """`open` unlocks everything; `guided` turns gating back on, keeping what you earned or
    skipped."""
    if body.policy == "open":
        unlocks.open_everything(sorted(gating.lock_table()))
    elif body.policy == "guided":
        unlocks.reset_to_guided()
    else:
        raise ValueError(f"policy must be one of {', '.join(unlocks.POLICIES)}, "
                         f"got {body.policy!r}")
    return _view()


async def learn_changes(stop: asyncio.Event | None = None,
                        interval_ms: int = 15000) -> AsyncIterator[str]:
    """SSE frames when unlocks.json (`unlocks`) or progress.json (`progress`) change."""
    from watchfiles import awatch

    folder = paths.learn_dir()
    folder.mkdir(parents=True, exist_ok=True)
    names = {unlocks.FILE: "unlocks", progress.FILE: "progress"}
    async for changes in awatch(folder, stop_event=stop, yield_on_timeout=True,
                                rust_timeout=interval_ms, debounce=50):
        if not changes:
            yield sse.KEEPALIVE
            continue
        for kind in sorted({names[Path(n).name] for _, n in changes if Path(n).name in names}):
            payload = _view().model_dump() if kind == "unlocks" else progress.read()
            yield sse.frame(kind, payload)


@router.get("/learn/events")
async def learn_events() -> StreamingResponse:
    """Server-sent events: `unlocks` or `progress` with the new document whenever either file
    changes (a lesson check finished, `nanoscope learn unlock` ran in a terminal)."""
    return sse.response(learn_changes())
