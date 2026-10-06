"""Read paths and lessons from TOML and Markdown, and say everything that is wrong at once.

    path = load_path("foundations")
    lesson = load_lesson("foundations/01-bigram")

A file with mistakes raises `CurriculumError`, whose `problems` lists every one of them
(not just the first), each with the file and the field, so an author fixes a lesson in one go.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoscope.specs import Problem

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

LEVELS = (0, 1, 2, 3)
CHECK_KINDS = {  # kind -> (required args, optional args)
    "defines": ({"class"}, {"args"}),
    "equivalent": ({"reference", "call", "inputs"}, {"class", "args", "tolerance", "trials"}),
    "forbid": (set(), set()),
    "trains": ({"metric", "threshold"}, {"seeds"}),
    "verdict": ({"a", "b", "expect"}, {"seeds", "metric"}),
    "predicted": ({"quantity"}, set()),
    "reproduces": ({"baseline"}, {"metric"}),
}
EXPERIMENT_KINDS = ("run", "study", "check")
MAX_CPU_MINUTES = 15
UNLOCK_ID = re.compile(r"^(block|feature):[A-Za-z_][A-Za-z0-9_]*$")
SECTIONS = ("surface", "deep", "reading")


def curricula_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "curricula"


class CurriculumError(ValueError):
    """A path or lesson that can't be used; `problems` holds every reason."""

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        super().__init__("; ".join(f"{p.field}: {p.message}" if p.field else p.message
                                   for p in problems))


@dataclass(frozen=True)
class Compute:
    """One way to run a lesson: a preset or a budget, and how long it takes."""

    preset: str | None
    budget: str | None
    estimate_minutes: float


@dataclass(frozen=True)
class Check:
    id: str
    kind: str
    args: dict[str, Any]


@dataclass(frozen=True)
class LessonText:
    intro: str
    surface: str
    deep: str
    reading: str


@dataclass(frozen=True)
class LessonSpec:
    id: str  # "foundations/01-bigram"
    path: str
    slug: str  # "01-bigram"
    dir: Path
    title: str
    level: int
    summary: str
    prerequisites: list[str]
    experiment: dict[str, Any]
    checks: list[Check]
    depth: dict[str, dict[str, Any]]
    unlocks: list[str]
    forbid: list[str]
    compute: dict[str, Compute]
    text: LessonText
    has_starter: bool
    has_notebook: bool

    @property
    def estimate_minutes(self) -> float | None:
        cpu = self.compute.get("cpu")
        return cpu.estimate_minutes if cpu else None


@dataclass(frozen=True)
class PathSpec:
    id: str
    dir: Path
    title: str
    level: int
    summary: str
    prerequisites: list[str]
    compute: dict[str, Compute]
    lessons: list[LessonSpec] = field(default_factory=list)


class _Problems:
    def __init__(self, file: str) -> None:
        self.file = file
        self.items: list[Problem] = []

    def add(self, where: str, message: str, hint: str | None = None) -> None:
        self.items.append(Problem("curriculum", f"{self.file}: {where}" if where else self.file,
                                  message, hint))


def _read_toml(file: Path, problems: _Problems) -> dict[str, Any] | None:
    try:
        return tomllib.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.add("", "file is missing")
    except tomllib.TOMLDecodeError as exc:
        problems.add("", f"not valid TOML: {exc}")
    return None


def _str(doc: dict[str, Any], key: str, problems: _Problems, where: str = "",
         required: bool = True) -> str:
    value = doc.get(key)
    path = f"{where}{key}"
    if value is None:
        if required:
            problems.add(path, "is required")
        return ""
    if not isinstance(value, str) or not value.strip():
        problems.add(path, "must be a non-empty string")
        return ""
    return value


def _str_list(doc: dict[str, Any], key: str, problems: _Problems) -> list[str]:
    value = doc.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        problems.add(key, "must be a list of strings")
        return []
    return list(value)


def _unknown(doc: dict[str, Any], known: set[str], problems: _Problems, where: str = "") -> None:
    for key in sorted(set(doc) - known):
        problems.add(f"{where}{key}", "is not a known field",
                     f"known fields: {', '.join(sorted(known))}")


def _level(doc: dict[str, Any], problems: _Problems) -> int:
    level = doc.get("level")
    if isinstance(level, bool) or level not in LEVELS:
        problems.add("level", f"must be one of {', '.join(map(str, LEVELS))}")
        return 0
    return int(level)


def _compute(doc: dict[str, Any], problems: _Problems, required: bool) -> dict[str, Compute]:
    table = doc.get("compute")
    if table is None:
        if required:
            problems.add("compute", "needs a [compute.cpu] table")
        return {}
    if not isinstance(table, dict):
        problems.add("compute", "must be a table with cpu and optionally gpu")
        return {}
    _unknown(table, {"cpu", "gpu"}, problems, "compute.")
    out: dict[str, Compute] = {}
    for name in ("cpu", "gpu"):
        variant = table.get(name)
        if variant is None:
            continue
        where = f"compute.{name}"
        if not isinstance(variant, dict):
            problems.add(where, "must be a table")
            continue
        _unknown(variant, {"preset", "budget", "estimate_minutes"}, problems, f"{where}.")
        preset, budget = variant.get("preset"), variant.get("budget")
        if (preset is None) == (budget is None):
            problems.add(where, "needs exactly one of preset or budget")
        minutes = variant.get("estimate_minutes")
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or minutes <= 0:
            problems.add(f"{where}.estimate_minutes", "must be a number of minutes above 0")
            continue
        out[name] = Compute(preset if isinstance(preset, str) else None,
                            budget if isinstance(budget, str) else None, float(minutes))
    if "gpu" in out and "cpu" not in out:
        problems.add("compute.gpu", "a gpu variant needs a cpu variant too (CI and "
                     "laptops run the cpu one)")
    cpu = out.get("cpu")
    if cpu and cpu.estimate_minutes > MAX_CPU_MINUTES and "gpu" not in out:
        problems.add("compute.cpu.estimate_minutes",
                     f"{cpu.estimate_minutes:g} minutes is over {MAX_CPU_MINUTES}: shrink the "
                     "cpu variant or add a [compute.gpu] variant")
    return out


def _checks(doc: dict[str, Any], problems: _Problems) -> list[Check]:
    raw = doc.get("checks", [])
    if not isinstance(raw, list):
        problems.add("checks", "must be an array of [[checks]] tables")
        return []
    out, seen = [], set()
    for i, item in enumerate(raw):
        where = f"checks[{i}]"
        if not isinstance(item, dict):
            problems.add(where, "must be a table")
            continue
        kind, cid = item.get("kind"), item.get("id")
        if kind not in CHECK_KINDS:
            problems.add(f"{where}.kind", f"must be one of {', '.join(CHECK_KINDS)}")
            continue
        if not isinstance(cid, str) or not cid.strip():
            problems.add(f"{where}.id", "is required")
            continue
        if cid in seen:
            problems.add(f"{where}.id", f"{cid!r} is used twice")
        seen.add(cid)
        required, optional = CHECK_KINDS[kind]
        args = {k: v for k, v in item.items() if k not in ("id", "kind")}
        for key in sorted(required - set(args)):
            problems.add(f"{where}.{key}", f"is required for a {kind} check")
        for key in sorted(set(args) - required - optional):
            problems.add(f"{where}.{key}", f"is not an argument of a {kind} check")
        out.append(Check(cid, kind, args))
    return out


def _experiment(doc: dict[str, Any], problems: _Problems) -> dict[str, Any]:
    exp = doc.get("experiment")
    if exp is None:
        return {}
    if not isinstance(exp, dict) or exp.get("kind") not in EXPERIMENT_KINDS:
        problems.add("experiment.kind", f"must be one of {', '.join(EXPERIMENT_KINDS)}")
        return {}
    return dict(exp)


def _depth(doc: dict[str, Any], problems: _Problems) -> dict[str, dict[str, Any]]:
    depth = doc.get("depth", {})
    if not isinstance(depth, dict):
        problems.add("depth", "must be a table")
        return {}
    out = {}
    for tier, value in depth.items():
        if tier not in ("surface", "deep", "research") or not isinstance(value, dict):
            problems.add(f"depth.{tier}", "tiers are surface, deep and research, each a table "
                         "of experiment arguments (seeds, budget, ...)")
            continue
        out[tier] = dict(value)
    return out


LESSON_FIELDS = {"title", "level", "summary", "prerequisites", "experiment", "checks", "depth",
                 "unlocks", "forbid", "compute"}
PATH_FIELDS = {"title", "level", "summary", "prerequisites", "compute"}


def parse_lesson_md(text: str, problems: _Problems | None = None) -> LessonText:
    """Split lesson.md at its level-2 headings. `## Surface` is required."""
    parts: dict[str, list[str]] = {"intro": []}
    current = "intro"
    for line in text.splitlines():
        match = re.match(r"^##\s+(.*?)\s*$", line)
        if match and not line.startswith("###"):
            current = match.group(1).strip().lower()
            parts.setdefault(current, [])
        else:
            parts[current].append(line)
    body = {k: "\n".join(v).strip() for k, v in parts.items()}
    if not body.get("surface") and problems is not None:
        problems.add("", "needs a '## Surface' section with something in it")
    return LessonText(body.get("intro", ""), body.get("surface", ""), body.get("deep", ""),
                      body.get("reading", ""))


def _resolve(ref: str | Path, root: Path | None, depth: int) -> Path:
    candidate = Path(ref)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    parts = Path(str(ref).replace("\\", "/")).parts
    if len(parts) != depth:
        raise CurriculumError([Problem(
            "curriculum", None, f"{ref!r} should be written like "
            f"{'foundations/01-bigram' if depth == 2 else 'foundations'}")])
    return (root or curricula_dir()).joinpath(*parts)


def load_lesson(ref: str | Path, root: Path | None = None) -> LessonSpec:
    """A lesson by id (`foundations/01-bigram`) or by folder."""
    directory = _resolve(ref, root, 2)
    return _load_lesson_dir(directory, f"{directory.parent.name}/{directory.name}")


def _load_lesson_dir(directory: Path, lesson_id: str) -> LessonSpec:
    problems = _Problems(f"{lesson_id}/lesson.toml")
    doc = _read_toml(directory / "lesson.toml", problems) or {}
    _unknown(doc, LESSON_FIELDS, problems)
    title = _str(doc, "title", problems)
    level = _level(doc, problems)
    unlocks = _str_list(doc, "unlocks", problems)
    for i, item in enumerate(unlocks):
        if not UNLOCK_ID.match(item):
            problems.add(f"unlocks[{i}]", f"{item!r} must look like block:Attention or "
                         "feature:gqa")
    summary = _str(doc, "summary", problems, required=False)
    prerequisites = _str_list(doc, "prerequisites", problems)
    experiment = _experiment(doc, problems)
    checks = _checks(doc, problems)
    depth = _depth(doc, problems)
    forbid = _str_list(doc, "forbid", problems)
    compute = _compute(doc, problems, required=True)
    md_problems = _Problems(f"{lesson_id}/lesson.md")
    md = directory / "lesson.md"
    if md.exists():
        text = parse_lesson_md(md.read_text(encoding="utf-8"), md_problems)
    else:
        md_problems.add("", "file is missing")
        text = LessonText("", "", "", "")
    errors = problems.items + md_problems.items
    if errors:
        raise CurriculumError(errors)
    return LessonSpec(
        id=lesson_id, path=lesson_id.split("/")[0], slug=directory.name, dir=directory,
        title=title, level=level, summary=summary, prerequisites=prerequisites,
        experiment=experiment, checks=checks, depth=depth, unlocks=unlocks, forbid=forbid,
        compute=compute, text=text, has_starter=(directory / "starter.py").exists(),
        has_notebook=(directory / "notebook.py").exists())


def load_path(ref: str | Path, root: Path | None = None) -> PathSpec:
    """A path by name (`foundations`) or by folder, with all its lessons. Problems in the
    path file and in every lesson are reported together."""
    directory = _resolve(ref, root, 1)
    path_id = directory.name
    problems = _Problems(f"{path_id}/path.toml")
    doc = _read_toml(directory / "path.toml", problems) or {}
    _unknown(doc, PATH_FIELDS, problems)
    title = _str(doc, "title", problems)
    level = _level(doc, problems)
    summary = _str(doc, "summary", problems, required=False)
    prerequisites = _str_list(doc, "prerequisites", problems)
    compute = _compute(doc, problems, required=False)
    errors = list(problems.items)
    lessons: list[LessonSpec] = []
    folders = sorted(p for p in directory.iterdir() if p.is_dir() and (p / "lesson.toml").exists()
                     or p.is_dir() and re.match(r"^\d\d-", p.name)) if directory.exists() else []
    for folder in folders:
        try:
            lessons.append(_load_lesson_dir(folder, f"{path_id}/{folder.name}"))
        except CurriculumError as exc:
            errors += exc.problems
    known = {lesson.id for lesson in lessons}
    for lesson in lessons:
        for pre in lesson.prerequisites:
            if "/" in pre and pre.split("/")[0] == path_id and pre not in known:
                errors.append(Problem("curriculum", f"{lesson.id}/lesson.toml: prerequisites",
                                      f"{pre!r} is not a lesson of this path"))
    if errors:
        raise CurriculumError(errors)
    return PathSpec(path_id, directory, title, level, summary, prerequisites, compute, lessons)


def list_path_ids(root: Path | None = None) -> list[str]:
    base = root or curricula_dir()
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "path.toml").exists())
