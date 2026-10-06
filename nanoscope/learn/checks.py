"""A lesson's checks: small experiments that decide whether the learner got it.

Each kind in `loader.CHECK_KINDS` has a function in `CHECKERS`:

    def check_defines(ctx: Context, check: Check) -> Result

`run_lesson_checks` runs a lesson's checks in order, tells the learner each verdict and its
reason as it goes, and writes `learn/checks/<id>.json` (`check.v1`). Every verdict carries a
readable reason, pass or fail. If a `defines` check fails, the rest are skipped: they would
only fail for the same reason.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths
from nanoscope.fsutil import write_json_atomic
from nanoscope.learn import progress
from nanoscope.learn.loader import Check, LessonSpec
from nanoscope.log import info


@dataclass
class Result:
    id: str
    kind: str
    passed: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "passed": self.passed, "skipped": self.skipped,
                "reason": self.reason, "evidence": self.evidence,
                "seconds": round(self.seconds, 3)}


@dataclass
class Context:
    """What a check can see: the lesson, the learner's file, and whose progress this is."""

    lesson: LessonSpec
    owner: str = "local"
    variant: str = "cpu"  # which [compute.*] variant to run
    shared: dict[str, Any] = field(default_factory=dict)  # checks pass things on (the class)

    @property
    def user_file(self) -> Path:
        from nanoscope.learn.cli import workspace_lesson_dir

        return workspace_lesson_dir(self.lesson) / "starter.py"


Checker = Callable[[Context, Check], Result]
CHECKERS: dict[str, Checker] = {}


def checker(kind: str) -> Callable[[Checker], Checker]:
    def register(fn: Checker) -> Checker:
        CHECKERS[kind] = fn
        return fn
    return register


def check_id(lesson: LessonSpec) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{lesson.path}-{lesson.slug}-{stamp}"


def checks_dir(owner: str = "local") -> Path:
    return paths.learn_dir(owner) / "checks"


def run_one(ctx: Context, check: Check) -> Result:
    started = time.perf_counter()
    fn = CHECKERS.get(check.kind)
    try:
        if fn is None:
            result = Result(check.id, check.kind, False, f"no checker for {check.kind!r} yet")
        else:
            result = fn(ctx, check)
    except Exception as exc:  # the learner's code, or ours: say which line, never a traceback
        frames = traceback.extract_tb(exc.__traceback__)
        where = f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
        result = Result(check.id, check.kind, False,
                        f"crashed: {type(exc).__name__}: {exc}{where}")
    result.seconds = time.perf_counter() - started
    return result


def run_lesson_checks(lesson: LessonSpec, owner: str = "local",
                      variant: str = "cpu") -> dict[str, Any]:
    """Run every check of `lesson`, print each verdict, record progress and the result file.
    Returns the check.v1 document."""
    cid = check_id(lesson)
    progress.mark(lesson.id, "checking", check_id=cid, owner=owner)
    ctx = Context(lesson, owner, variant)
    n = len(lesson.checks)
    info(f"checking {lesson.id} ({n} check{'s' if n != 1 else ''})")
    results: list[Result] = []
    blocked = False
    for check in lesson.checks:
        if blocked:
            result = Result(check.id, check.kind, False,
                            "skipped: fix the failing 'defines' check first", skipped=True)
        else:
            result = run_one(ctx, check)
        results.append(result)
        info(f"  [{'pass' if result.passed else 'skip' if result.skipped else 'FAIL'}] "
             f"{check.id} ({check.kind}): {result.reason}")
        blocked = blocked or (check.kind == "defines" and not result.passed)
    passed = bool(results) and all(r.passed for r in results)
    doc = {
        "schema": 1, "nanoscope": __version__, "id": cid, "lesson": lesson.id,
        "at": progress.now(), "variant": variant, "passed": passed,
        "checks": [r.to_dict() for r in results],
    }
    write_json_atomic(checks_dir(owner) / f"{cid}.json", doc)
    progress.mark(lesson.id, "passed" if passed else "failed", check_id=cid, owner=owner)
    done = sum(r.passed for r in results)
    info(f"result: {done} of {len(results)} checks passed"
         + ("" if passed else ": read the reasons above, edit your file, and check again"))
    return doc
