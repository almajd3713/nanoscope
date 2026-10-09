"""The honest-ablations path: every starter fails its checks and every solution passes.

Offline on the synthetic stories (`fake_data`) with a tiny preset. The statements that only
mean something on real text (what the A/A spread is, which component is null) are covered by
the network-marked tests at the end, which train for real on TinyStories.
"""

import dataclasses

import pytest
from test_curricula import attempt, reasons, small_cpu_preset, solution, verdicts

from nanoscope.learn import gating
from nanoscope.learn.loader import load_lesson


def quick(lesson, steps=20):
    """The lesson with every `max_steps` in its checks cut down, so the structure of each check
    runs in seconds on the synthetic stories."""
    def cut(value):
        if isinstance(value, dict):
            return {k: steps if k == "max_steps" else cut(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cut(v) for v in value]
        return value

    return dataclasses.replace(lesson, checks=[
        dataclasses.replace(c, args=cut(c.args)) for c in lesson.checks])


@pytest.fixture(autouse=True)
def workspace(home, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "workspace"))
    gating.reload()


@pytest.mark.usefixtures("fake_data")
def test_ha01_starter_fails_on_size_and_solution_passes(monkeypatch):
    lesson = quick(load_lesson("honest-ablations/01-seed-noise"))
    assert [c.kind for c in lesson.checks] == ["defines", "matches", "verdict"]
    assert lesson.compute["cpu"].estimate_minutes <= 15
    small_cpu_preset(monkeypatch, lesson)
    starter = attempt(lesson)
    assert verdicts(starter)["built"] and not verdicts(starter)["same-size"]
    assert "MyGPT has" in reasons(starter)["same-size"]
    done = attempt(lesson, solution("honest-ablations/01-seed-noise"))
    assert done["passed"], reasons(done)
    assert "unpaired" in reasons(done)["noise"]
