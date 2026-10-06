from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope import paths, queue
from nanoscope.learn import loader, progress, unlocks
from nanoscope.learn.cli import locked_by, start_lesson, workspace_lesson_dir
from nanoscope.learn.loader import LessonSpec
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


@router.get("/learn/progress")
def learn_progress() -> ProgressDoc:
    """Where you are in each lesson (progress.json)."""
    return ProgressDoc.model_validate(progress.read())
