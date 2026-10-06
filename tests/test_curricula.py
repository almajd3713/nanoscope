"""The shipped lessons, checked against their reference solutions in tests/solutions/.

Offline: every lesson's starter fails and its solution passes. The lessons that train a model
use `fake_data` and a tiny preset, with the targets that only mean something on real text left
out. The network-marked tests run the real thing on TinyStories (cached after the first run).
"""

import dataclasses
import time
from pathlib import Path

import pytest
from fakes import tiny
from learn_helpers import set_preset

from nanoscope.learn import gating, progress
from nanoscope.learn.checks import run_lesson_checks
from nanoscope.learn.cli import start_lesson, workspace_lesson_dir
from nanoscope.learn.loader import LessonSpec, load_lesson

SOLUTIONS = Path(__file__).parent / "solutions"
REAL_ONLY = ("reproduces", "verdict")  # these compare against numbers from real text


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "workspace"))
    gating.reload()


def solution(lesson_id: str) -> str:
    return (SOLUTIONS / f"{lesson_id}.py").read_text(encoding="utf-8")


def offline(lesson: LessonSpec) -> LessonSpec:
    """The lesson as it can run on the synthetic stories: no checks that need real numbers,
    and training targets loose enough for a tiny model on tiny data."""
    checks = []
    for check in lesson.checks:
        if check.kind in REAL_ONLY:
            continue
        if check.kind == "trains":
            check = dataclasses.replace(check, args={**check.args, "threshold": 100.0})
        checks.append(check)
    return dataclasses.replace(lesson, checks=checks)


def attempt(lesson: LessonSpec, code: str | None = None) -> dict:
    """Start the lesson (so the starter is in the workspace), put `code` in the learner's file
    if given, and run its checks. Returns the check.v1 document."""
    start_lesson(lesson)
    file = workspace_lesson_dir(lesson) / "starter.py"
    if code is not None:
        file.write_text(code, encoding="utf-8")
    return run_lesson_checks(lesson)


def verdicts(doc: dict) -> dict[str, bool]:
    return {c["id"]: c["passed"] for c in doc["checks"]}


def reasons(doc: dict) -> dict[str, str]:
    return {c["id"]: c["reason"] for c in doc["checks"]}


def small_cpu_preset(monkeypatch, lesson: LessonSpec, **kw) -> None:
    """Make the lesson's cpu preset a tiny one for this test."""
    set_preset(monkeypatch, tiny(name=lesson.compute["cpu"].preset, **kw))


@pytest.mark.usefixtures("fake_data")
def test_f01_bigram(monkeypatch):
    lesson = load_lesson("foundations/01-bigram")
    assert [c.kind for c in lesson.checks] == ["defines", "trains", "reproduces"]
    small_cpu_preset(monkeypatch, lesson)
    starter = attempt(offline(lesson))
    assert verdicts(starter) == {"built": True, "learns": False}  # builds, but cannot train yet
    assert "NotImplementedError" in reasons(starter)["learns"] or "parameter" in reasons(
        starter)["learns"]
    doc = attempt(offline(lesson), solution("foundations/01-bigram"))
    assert doc["passed"], reasons(doc)
    assert progress.state(lesson.id) == "passed"


@pytest.mark.network
def test_f01_bigram_real_data_under_two_minutes(home):
    lesson = load_lesson("foundations/01-bigram")
    started = time.perf_counter()
    doc = attempt(lesson, solution("foundations/01-bigram"))
    elapsed = time.perf_counter() - started
    assert doc["passed"], reasons(doc)
    assert verdicts(doc) == {"built": True, "learns": True, "matches-baseline": True}
    assert elapsed < 120, f"F01 took {elapsed:.0f} s on this CPU (budget: 2 minutes)"


@pytest.mark.usefixtures("fake_data")
def test_f02_mlp(monkeypatch):
    lesson = load_lesson("foundations/02-mlp")
    small_cpu_preset(monkeypatch, lesson)
    starter = attempt(offline(lesson))
    assert verdicts(starter)["built"] and not verdicts(starter)["beats-bigram"]
    doc = attempt(offline(lesson), solution("foundations/02-mlp"))
    assert doc["passed"], reasons(doc)


@pytest.mark.network
def test_f02_mlp_real_data(home):
    lesson = load_lesson("foundations/02-mlp")
    doc = attempt(lesson, solution("foundations/02-mlp"))
    assert doc["passed"], reasons(doc)
    assert doc["checks"][1]["evidence"]["mean"] <= 1.6
