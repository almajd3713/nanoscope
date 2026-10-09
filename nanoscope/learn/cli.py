"""`nanoscope learn ...`: lessons from the command line, the same ones the GUI shows."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from nanoscope import paths
from nanoscope.learn import gating, loader, progress, unlocks
from nanoscope.learn.loader import CurriculumError, LessonSpec, PathSpec


def add_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    learn = sub.add_parser("learn", help="Lessons: list, start, check")
    commands = learn.add_subparsers(dest="learn_command")

    ls = commands.add_parser("list", help="Paths and lessons with their state and compute")
    ls.add_argument("--path", default=None, help="only this path, e.g. foundations")

    check = commands.add_parser("check", help="Run a lesson's checks on your file")
    check.add_argument("lesson", help="e.g. foundations/01-bigram")
    check.add_argument("--queue", action="store_true",
                       help="run it as a worker job (interactive lane) instead of here")
    check.add_argument("--variant", choices=["cpu", "gpu"], default="cpu",
                       help="which compute variant to run (default cpu)")

    author = commands.add_parser(
        "author-check", help="Check a lesson folder you wrote: it loads, the starter fails, "
        "the solution passes")
    author.add_argument("folder", help="e.g. curricula/my-course/01-layernorm")
    author.add_argument("--variant", choices=["cpu", "gpu"], default="cpu")
    author.add_argument("--json", action="store_true", help="print the author-check.v1 document")
    author.add_argument("--queue", action="store_true",
                        help="run it as a worker job (interactive lane) instead of here")

    unlock = commands.add_parser("unlock", help="Open locked blocks (everything, or one)")
    unlock.add_argument("id", nargs="?", help="e.g. block:Attention or feature:gqa")
    unlock.add_argument("--all", action="store_true", help="unlock everything (policy open)")
    unlock.add_argument("--reason", help="why you can skip this lesson (needed with an id)")

    lock = commands.add_parser("lock", help="Turn gating back on")
    lock.add_argument("--reset", action="store_true",
                      help="back to guided, keeping what you earned or skipped")

    commands.add_parser("status", help="Your lessons and what is unlocked")

    predict = commands.add_parser(
        "predict", help="Commit to a prediction before you run the experiment")
    predict.add_argument("lesson")
    predict.add_argument("--verdict", choices=["better", "worse", "within noise"])
    predict.add_argument("--low", type=float, help="low end of the difference you expect")
    predict.add_argument("--high", type=float, help="high end of the difference you expect")
    predict.add_argument("--note", help="why you expect it")

    start = commands.add_parser("start", help="Copy a lesson's starter files to your workspace")
    start.add_argument("lesson", help="e.g. foundations/01-bigram")
    start.add_argument("--open", action="store_true",
                       help="don't gate blocks (policy open) when starting for the first time")


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


def _resolve_id(text: str) -> str:
    table = gating.lock_table()
    if text in table:
        return text
    for prefix in ("block:", "feature:"):
        if prefix + text in table:
            return prefix + text
    known = ", ".join(sorted(table)) or "none"
    raise ValueError(f"{text!r} is not locked by any lesson (lockable: {known})")


def _unlock(args: argparse.Namespace) -> int:
    if args.all:
        unlocks.open_everything(sorted(gating.lock_table()))
        print("everything is unlocked (policy open); what you earned stays recorded as earned")
        print("  gating back on, keeping what you earned: nanoscope learn lock --reset")
        return 0
    if not args.id:
        print("name what to unlock, e.g. nanoscope learn unlock block:Attention --reason "
              "\"I know this\", or use --all")
        return 2
    if not args.reason or not args.reason.strip():
        print("skipping a lesson needs a reason: add --reason \"I know this\"")
        return 2
    try:
        unlock_id = _resolve_id(args.id)
    except ValueError as exc:
        print(exc)
        return 2
    if not unlocks.exists():
        unlocks.set_policy("guided")
    unlocks.grant(unlock_id, "skipped", gating.lock_table()[unlock_id], reason=args.reason.strip())
    print(f"{unlock_id} unlocked (skipped; recorded with your reason). The lesson "
          f"{gating.lock_table()[unlock_id]} is still there if you want to build it.")
    return 0


def _lock(args: argparse.Namespace) -> int:
    if not args.reset:
        print("usage: nanoscope learn lock --reset")
        return 2
    unlocks.reset_to_guided()
    print("gating is back on (policy guided); earned and skipped unlocks are kept")
    return 0


def format_status(owner: str = "local") -> str:
    """Policy, lessons and the unlock table."""
    doc = unlocks.read(owner)
    lines = [f"policy: {doc['policy']}"
             + ("" if unlocks.exists(owner) else " (no unlocks.json yet: nothing is locked)"),
             "", format_paths(load_paths(), owner), "", "unlocks:"]
    table = gating.lock_table()
    if not table:
        lines.append("  nothing is lockable")
    rows = []
    for unlock_id, lesson in sorted(table.items()):
        entry = doc["unlocks"].get(unlock_id)
        if entry:
            how = entry["how"] + (f" ({entry['reason']})" if entry.get("reason") else "")
            rows.append((unlock_id, how, entry["lesson"] or lesson, entry["at"][:10]))
        elif doc["policy"] == "open":
            rows.append((unlock_id, "open", lesson, ""))
        else:
            rows.append((unlock_id, "locked", lesson, ""))
    widths = [max(len(r[i]) for r in rows) for i in range(3)] if rows else []
    lines += [f"  {r[0].ljust(widths[0])}  {r[1].ljust(widths[1])}  {r[2].ljust(widths[2])}  "
              f"{r[3]}".rstrip() for r in rows]
    return "\n".join(lines)


def _predict(args: argparse.Namespace) -> int:
    from nanoscope.learn.checks import PredictionMissing, PredictionTooLate, commit_prediction

    lesson = loader.load_lesson(args.lesson)
    given = {k: v for k, v in {"verdict": args.verdict, "low": args.low, "high": args.high,
                               "note": args.note}.items() if v is not None}
    try:
        file, stamp = commit_prediction(lesson, given)
    except PredictionTooLate as exc:
        print(exc)
        return 1
    except PredictionMissing as exc:
        print(exc)
        return 2
    except ValueError as exc:
        print(exc)
        return 2
    print(f"recorded your prediction for {lesson.id} at {stamp}: {file}")
    print(f"  it can't be changed now. Next: nanoscope learn check {lesson.id}")
    return 0


def _author_check(args: argparse.Namespace) -> int:
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"{args.folder} is not a folder")
        return 2
    if args.queue:
        from nanoscope import queue

        job = queue.enqueue("author-check", {"folder": str(folder), "variant": args.variant},
                            lane="interactive")
        print(f"queued author-check job #{job} for {folder}")
        print("  watch it: nanoscope jobs    (needs a worker: nanoscope worker)")
        return 0
    from nanoscope.learn.authoring import author_check, format_author_check

    doc = author_check(folder, args.variant)
    if args.json:
        import json

        print(json.dumps(doc, indent=2))
    else:
        print(format_author_check(doc))
    return 0 if doc["state"] == "ready" else 1


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
        print("usage: nanoscope learn list | start | predict | check <lesson>")
        return 0
    if command == "list":
        print(format_paths(load_paths(args.path)))
        return 0
    if command == "start":
        lesson = loader.load_lesson(args.lesson)
        first = not unlocks.exists()
        if first:
            unlocks.set_policy("open" if args.open else "guided")
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
        if first:
            print("  bigger blocks stay locked until you build them in a lesson; to unlock "
                  "everything now: nanoscope learn unlock --all" if not args.open else
                  "  gating is off (--open); turn it on any time: nanoscope learn lock --reset")
        return 0
    if command == "author-check":
        return _author_check(args)
    if command == "predict":
        return _predict(args)
    if command == "unlock":
        return _unlock(args)
    if command == "lock":
        return _lock(args)
    if command == "status":
        print(format_status())
        return 0
    if command == "check":
        from nanoscope.learn.checks import run_lesson_checks

        lesson = loader.load_lesson(args.lesson)
        if args.variant not in lesson.compute:
            print(f"{lesson.id} has no {args.variant} variant; it has: "
                  f"{', '.join(lesson.compute)}")
            return 2
        if args.queue:
            from nanoscope import queue

            job = queue.enqueue("check", {"lesson": lesson.id, "variant": args.variant},
                                lane="interactive")
            print(f"queued check job #{job} for {lesson.id}")
            print("  watch it: nanoscope jobs    (needs a worker: nanoscope worker)")
            return 0
        doc = run_lesson_checks(lesson, variant=args.variant)
        return 0 if doc["passed"] else 1
    return 2
