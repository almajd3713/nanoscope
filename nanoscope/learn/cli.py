"""`nanoscope learn ...`: lessons from the command line, the same ones the GUI shows."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from nanoscope import paths
from nanoscope.learn import loader, progress
from nanoscope.learn.loader import CurriculumError, LessonSpec, PathSpec


def add_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    learn = sub.add_parser("learn", help="Lessons: list, start, check")
    commands = learn.add_subparsers(dest="learn_command")

    ls = commands.add_parser("list", help="Paths and lessons with their state and compute")
    ls.add_argument("--path", default=None, help="only this path, e.g. foundations")

    check = commands.add_parser("check", help="Run a lesson's checks on your file")
    check.add_argument("lesson", help="e.g. foundations/01-bigram")
    check.add_argument("--variant", choices=["cpu", "gpu"], default="cpu",
                       help="which compute variant to run (default cpu)")

    start = commands.add_parser("start", help="Copy a lesson's starter files to your workspace")
    start.add_argument("lesson", help="e.g. foundations/01-bigram")


def _minutes(minutes: float) -> str:
    return f"{minutes:g} min" if minutes < 90 else f"{minutes / 60:.3g} h"


def compute_text(compute: dict) -> str:
    """`cpu 2 min / gpu 4 h`, or `-` when nothing is declared."""
    parts = [f"{name} {_minutes(c.estimate_minutes)}" for name, c in compute.items()]
    return " / ".join(parts) or "-"


def locked_by(lesson: LessonSpec, owner: str = "local") -> list[str]:
    """Prerequisite lessons not passed yet."""
    return [p for p in lesson.prerequisites
            if "/" in p and progress.state(p, owner) != "passed"]


def lesson_state(lesson: LessonSpec, owner: str = "local") -> str:
    state = progress.state(lesson.id, owner)
    needs = locked_by(lesson, owner)
    if state == "not-started" and needs:
        return f"locked (needs {', '.join(needs)})"
    return state


def format_paths(specs: list[PathSpec], owner: str = "local") -> str:
    if not specs:
        return "no learning paths are installed"
    lines = []
    for spec in specs:
        head = f"{spec.id}  {spec.title} (level {spec.level})"
        if spec.compute:
            head += f"  {compute_text(spec.compute)}"
        lines.append(head)
        rows = [(lesson.slug, lesson_state(lesson, owner), compute_text(lesson.compute),
                 lesson.title) for lesson in spec.lessons]
        widths = [max(len(r[i]) for r in rows) for i in range(3)] if rows else []
        for slug, state, compute, title in rows:
            lines.append(f"  {slug.ljust(widths[0])}  {state.ljust(widths[1])}  "
                         f"{compute.ljust(widths[2])}  {title}".rstrip())
        lines.append("")
    return "\n".join(lines).rstrip()


def load_paths(only: str | None = None) -> list[PathSpec]:
    ids = [only] if only else loader.list_path_ids()
    return [loader.load_path(i) for i in ids]


def workspace_lesson_dir(lesson: LessonSpec) -> Path:
    return paths.workspace_dir() / "lessons" / lesson.path / lesson.slug


def start_lesson(lesson: LessonSpec, owner: str = "local") -> tuple[Path, list[Path], list[Path]]:
    """Copy starter.py (and notebook.py) into the workspace; never overwrite an existing file.
    Returns (folder, copied, kept)."""
    folder = workspace_lesson_dir(lesson)
    folder.mkdir(parents=True, exist_ok=True)
    copied, kept = [], []
    for name in ("starter.py", "notebook.py"):
        source = lesson.dir / name
        if not source.exists():
            continue
        target = folder / name
        if target.exists():
            kept.append(target)
        else:
            shutil.copyfile(source, target)
            copied.append(target)
    if progress.state(lesson.id, owner) == "not-started":
        progress.mark(lesson.id, "started", owner=owner)
    return folder, copied, kept


def run(args: argparse.Namespace) -> int:
    try:
        return _run(args)
    except CurriculumError as exc:
        print("this lesson content has problems:")
        for problem in exc.problems:
            print(f"  {problem.field}: {problem.message}")
        return 1


def _run(args: argparse.Namespace) -> int:
    command = args.learn_command
    if command is None:
        print("usage: nanoscope learn list | start <lesson> | check <lesson>")
        return 0
    if command == "list":
        print(format_paths(load_paths(args.path)))
        return 0
    if command == "start":
        lesson = loader.load_lesson(args.lesson)
        folder, copied, kept = start_lesson(lesson)
        print(f"started {lesson.id}: {lesson.title}")
        for path in copied:
            print(f"  copied {path}")
        for path in kept:
            print(f"  kept your {path} (not overwritten)")
        if not lesson.has_starter and not lesson.has_notebook:
            print(f"  no starter files; work in {folder}")
        print(f"  compute: {compute_text(lesson.compute)}")
        needs = locked_by(lesson)
        if needs:
            print(f"  note: this lesson builds on {', '.join(needs)}, not passed yet")
        print(f"  next: edit the file, then run: nanoscope learn check {lesson.id}")
        return 0
    if command == "check":
        from nanoscope.learn.checks import run_lesson_checks

        lesson = loader.load_lesson(args.lesson)
        if args.variant not in lesson.compute:
            print(f"{lesson.id} has no {args.variant} variant; it has: "
                  f"{', '.join(lesson.compute)}")
            return 2
        doc = run_lesson_checks(lesson, variant=args.variant)
        return 0 if doc["passed"] else 1
    return 2
