"""`nanoscope learn author-check <dir>`: does a lesson folder an author wrote work?

It loads the folder as the curriculum loader does (every problem at once), then runs every
check twice: on `starter.py`, which should fail (it is what the learner fills in), and on
`solution.py`, which should pass. The result is an `author-check.v1` document; its `state` and
`summary` are the words the authoring page shows, so the page and the CLI agree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoscope.fsutil import write_json_atomic
from nanoscope.learn import checks as checks_module
from nanoscope.learn import loader, progress
from nanoscope.learn.authorstate import file_hashes, folder_id, result_file, short_where
from nanoscope.learn.checks import Context, Result
from nanoscope.learn.loader import CurriculumError, LessonSpec

STATES = ("ready", "does not load", "solution fails", "starter passes", "files missing")


@dataclass
class AuthorContext(Context):
    """A check context that reads the file it is given, not the learner's workspace copy."""

    file: Path | None = None

    @property
    def user_file(self) -> Path:
        assert self.file is not None
        return self.file


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _run_on(lesson: LessonSpec, file: Path, target: str, variant: str) -> list[Result]:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    ctx = AuthorContext(lesson, "local", variant, f"author-{target}-{lesson.slug}-{stamp}",
                        file=file)
    results: list[Result] = []
    blocked = False
    for check in lesson.checks:
        if blocked:
            result = Result(check.id, check.kind, False,
                            "skipped: fix the failing 'defines' check first", skipped=True)
        else:
            result = checks_module.run_one(ctx, check)
        results.append(result)
        blocked = blocked or (check.kind == "defines" and not result.passed)
    return results


def author_check(folder: str | Path, variant: str = "cpu") -> dict[str, Any]:
    """Check a lesson folder; returns the author-check.v1 document (and saves it)."""
    from nanoscope import __version__

    directory = Path(folder).resolve()
    started = time.perf_counter()
    doc: dict[str, Any] = {
        "schema": 1, "nanoscope": __version__, "folder": str(directory),
        "lesson": folder_id(directory), "title": None, "at": progress.now(),
        "variant": variant, "files": file_hashes(directory), "problems": [], "checks": [],
    }
    try:
        lesson = loader.load_lesson(directory)
    except CurriculumError as exc:
        prefix = f"{folder_id(directory)}/"
        doc["problems"] = [{"where": short_where(p.field, prefix), "message": p.message,
                            **({"hint": p.hint} if p.hint else {})} for p in exc.problems]
        doc["state"] = "does not load"
        doc["summary"] = (f"The curriculum loader found {_plural(len(doc['problems']), 'problem')}."
                          " Every one is listed; fix them and check again.")
        doc["seconds"] = round(time.perf_counter() - started, 3)
        return _save(directory, doc)

    doc["title"] = lesson.title
    if variant not in lesson.compute:
        raise ValueError(f"{lesson.id} has no {variant} variant; "
                         f"it has: {', '.join(lesson.compute)}")
    missing = [n for n in ("starter.py", "solution.py") if not (directory / n).is_file()]
    if missing:
        doc["state"] = "files missing"
        doc["summary"] = (f"The folder loads, but {' and '.join(missing)} "
                          f"{'is' if len(missing) == 1 else 'are'} missing: "
                          "the checks run on both.")
        doc["seconds"] = round(time.perf_counter() - started, 3)
        return _save(directory, doc)

    starter = _run_on(lesson, directory / "starter.py", "starter", variant)
    solution = _run_on(lesson, directory / "solution.py", "solution", variant)
    n = len(lesson.checks)
    starter_failed = sum(not r.passed for r in starter)
    solution_failed = sum(not r.passed for r in solution)
    doc["checks"] = [{"id": c.id, "kind": c.kind,
                      "starter": {"passed": s.passed, "skipped": s.skipped, "reason": s.reason},
                      "solution": {"passed": t.passed, "skipped": t.skipped, "reason": t.reason}}
                     for c, s, t in zip(lesson.checks, starter, solution, strict=True)]
    if solution_failed:
        doc["state"] = "solution fails"
        doc["summary"] = ("The folder loads, but solution.py fails "
                          + ("its check" if n == 1 else f"{solution_failed} of {n} checks")
                          + ": a learner who solves the lesson would fail too.")
    elif not starter_failed:
        doc["state"] = "starter passes"
        doc["summary"] = ("The folder loads, but starter.py already passes "
                          + ("its check" if n == 1 else "every check")
                          + ": the learner has nothing to do.")
    else:
        doc["state"] = "ready"
        doc["summary"] = ("The folder loads, starter.py fails "
                          + ("its check" if n == 1 else f"{starter_failed} of {n} checks")
                          + " and solution.py passes "
                          + ("it" if n == 1 else "both" if n == 2 else "all of them") + ".")
    doc["seconds"] = round(time.perf_counter() - started, 3)
    return _save(directory, doc)


def _save(directory: Path, doc: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(result_file(directory), doc)
    return doc


def format_author_check(doc: dict[str, Any]) -> str:
    lines = [f"{doc['lesson']}: {doc['state']}", doc["summary"]]
    for p in doc["problems"]:
        lines.append(f"  {p['where']}: {p['message']}" if p["where"] else f"  {p['message']}")
    for c in doc["checks"]:
        for target in ("starter", "solution"):
            r = c[target]
            mark = "pass" if r["passed"] else "skip" if r["skipped"] else "FAIL"
            lines.append(f"  [{mark}] {c['id']} ({c['kind']}) on {target}.py: {r['reason']}")
    return "\n".join(lines)
