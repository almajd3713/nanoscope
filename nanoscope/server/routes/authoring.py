"""Lesson authoring (Extend): the lesson folders in the workspace, what the loader makes of
one, and its author-check as a job (it imports the folder's starter and solution, so a worker
runs it, never this process)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nanoscope import queue
from nanoscope.learn import loader
from nanoscope.learn.authorstate import FILES, folder_id, last_check, short_where
from nanoscope.learn.loader import CurriculumError
from nanoscope.server import workspace
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import JobDoc

router = APIRouter(prefix="/api/authoring", tags=["authoring"])

ROOT = "curricula"


class LessonFolder(BaseModel):
    folder: str  # relative to the workspace
    id: str  # path/slug
    files: list[str]  # the lesson files it has


class AuthoredProblem(BaseModel):
    where: str
    message: str
    hint: str | None = None


class AuthoredLesson(BaseModel):
    """What a learner would see: from lesson.toml and lesson.md."""

    title: str
    summary: str
    level: int
    prerequisites: list[str]
    unlocks: list[str]
    forbid: list[str]
    experiment: dict[str, Any]
    checks: list[dict[str, Any]]
    compute: dict[str, dict[str, Any]]
    text: dict[str, str]  # intro, surface, deep, reading (Markdown)


class AuthoringDoc(BaseModel):
    folder: str
    id: str
    files: dict[str, bool]  # each lesson file: present or not
    loads: bool
    problems: list[AuthoredProblem]  # the loader's own, every one
    lesson: AuthoredLesson | None = None
    result: dict[str, Any] | None = None  # the last author-check.v1, plus `fresh`


class CheckRequest(BaseModel):
    variant: str = "cpu"


def _folder(relative: str) -> Any:
    target = workspace.safe_path(relative, must_exist=False)
    if not target.is_dir():
        raise HTTPException(404, f"no lesson folder {relative!r} in the workspace")
    return target


@router.get("")
def lessons() -> list[LessonFolder]:
    """Lesson folders under `curricula/<path>/<lesson>/` in the workspace."""
    base = workspace.root() / ROOT
    if not base.is_dir():
        return []
    found = []
    for path in sorted(p for p in base.iterdir() if p.is_dir()):
        for folder in sorted(p for p in path.iterdir() if p.is_dir()):
            names = [n for n in FILES if (folder / n).is_file()]
            if names:
                found.append(LessonFolder(folder=workspace.relative(folder), id=folder_id(folder),
                                          files=names))
    return found


@router.get("/{folder:path}")
def authoring(folder: str) -> AuthoringDoc:
    """One lesson folder: whether it loads (every problem if not), the lesson as a learner
    would read it, and the last author-check."""
    directory = _folder(folder)
    doc = AuthoringDoc(
        folder=workspace.relative(directory), id=folder_id(directory), loads=False, problems=[],
        files={n: (directory / n).is_file() for n in FILES}, result=last_check(directory))
    try:
        spec = loader.load_lesson(directory)
    except CurriculumError as exc:
        prefix = f"{doc.id}/"
        doc.problems = [AuthoredProblem(where=short_where(p.field, prefix), message=p.message,
                                        hint=p.hint) for p in exc.problems]
        return doc
    doc.loads = True
    doc.lesson = AuthoredLesson(
        title=spec.title, summary=spec.summary, level=spec.level,
        prerequisites=spec.prerequisites, unlocks=spec.unlocks, forbid=spec.forbid,
        experiment=spec.experiment,
        checks=[{"id": c.id, "kind": c.kind, **c.args} for c in spec.checks],
        compute={k: {"preset": c.preset, "budget": c.budget,
                     "estimate_minutes": c.estimate_minutes} for k, c in spec.compute.items()},
        text={"intro": spec.text.intro, "surface": spec.text.surface, "deep": spec.text.deep,
              "reading": spec.text.reading})
    return doc


@router.post("/{folder:path}/check", status_code=202)
def check(folder: str, body: CheckRequest | None = None) -> JobDoc:
    """Run `nanoscope learn author-check` on the folder as a job: the checks on starter.py
    (should fail) and on solution.py (should pass)."""
    directory = _folder(folder)
    variant = body.variant if body else "cpu"
    if variant not in ("cpu", "gpu"):
        raise ValueError(f"variant must be cpu or gpu, got {variant!r}")
    job_id = queue.enqueue("author-check", {"folder": str(directory), "variant": variant},
                           lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))
