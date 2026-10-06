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

from nanoscope.learn import gating, progress, unlocks
from nanoscope.learn.checks import run_lesson_checks
from nanoscope.learn.cli import start_lesson, workspace_lesson_dir
from nanoscope.learn.loader import LessonSpec, load_lesson

SOLUTIONS = Path(__file__).parent / "solutions"


@pytest.fixture(autouse=True)
def workspace(home, tmp_path, monkeypatch):  # `home`: progress and checks stay in the test
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "workspace"))
    gating.reload()


def solution(lesson_id: str) -> str:
    return (SOLUTIONS / f"{lesson_id}.py").read_text(encoding="utf-8")


def offline(lesson: LessonSpec) -> LessonSpec:
    """The lesson as it can run on the synthetic stories: a `reproduces` check becomes a plain
    training check (the baseline is real-text numbers), `verdict` is left out, and training
    targets are loose enough for a tiny model on tiny data."""
    checks = []
    for check in lesson.checks:
        if check.kind == "verdict":
            continue
        if check.kind == "reproduces":  # still trains the model, against a loose target
            check = dataclasses.replace(check, kind="trains", args={
                "class": check.args["class"], "metric": check.args.get("metric", "val_bpb"),
                "threshold": 100.0})
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
    assert verdicts(starter) == {"built": True, "learns": False,
                                 "matches-baseline": False}  # builds, but cannot train yet
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


def test_f03_attention_head():
    lesson = load_lesson("foundations/03-attention-head")
    starter = attempt(lesson)
    assert verdicts(starter) == {"built": True, "same-as-reference": False,
                                 "built-by-hand": True}
    assert "has no parameter named 'q.weight'" in reasons(starter)["same-as-reference"]
    doc = attempt(lesson, solution("foundations/03-attention-head"))
    assert doc["passed"], reasons(doc)
    assert doc["checks"][1]["evidence"]["max_abs_diff"] < 1e-5
    # the one-liner that skips the point of the lesson is refused, line by line
    shortcut = solution("foundations/03-attention-head").replace(
        "        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))\n"
        "        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), "
        "diagonal=1)\n"
        "        weights = scores.masked_fill(future, float(\"-inf\")).softmax(dim=-1)\n"
        "        return self.out(weights @ v)\n",
        "        import torch.nn.functional as F\n"
        "        return self.out(F.scaled_dot_product_attention(q, k, v, is_causal=True))\n")
    cheat = attempt(lesson, shortcut)
    assert verdicts(cheat) == {"built": True, "same-as-reference": True, "built-by-hand": False}
    assert "line 19: torch.nn.functional.scaled_dot_product_attention" in reasons(cheat)[
        "built-by-hand"]
    # a plausible mistake is caught with numbers: no mask, so the model sees the future
    unmasked = solution("foundations/03-attention-head").replace(
        "scores.masked_fill(future, float(\"-inf\"))", "scores")
    wrong = attempt(lesson, unmasked)
    assert not verdicts(wrong)["same-as-reference"]
    assert "output differs from the reference: max abs diff" in reasons(wrong)[
        "same-as-reference"]


def test_f04_multi_head(home):
    lesson = load_lesson("foundations/04-multi-head")
    assert lesson.unlocks == ["block:Attention"]
    unlocks.set_policy("guided")  # a learner who started with gating on
    starter = attempt(lesson)
    assert not starter["passed"] and not verdicts(starter)["same-as-reference"]
    assert not unlocks.is_unlocked("block:Attention")
    doc = attempt(lesson, solution("foundations/04-multi-head"))
    assert doc["passed"], reasons(doc)
    assert doc["checks"][1]["evidence"]["max_abs_diff"] < 1e-5
    entry = unlocks.read()["unlocks"]["block:Attention"]
    assert entry["how"] == "earned" and entry["lesson"] == lesson.id
    assert entry["evidence"] == f"learn/checks/{doc['id']}.json"
    assert unlocks.is_unlocked("block:Attention")
    # a head that mixes up the split fails with numbers
    mixed = solution("foundations/04-multi-head").replace(
        "t.view(B, T, self.n_heads, head_dim).transpose(1, 2)",
        "t.view(B, self.n_heads, T, head_dim)")
    wrong = attempt(lesson, mixed)
    assert not verdicts(wrong)["same-as-reference"]


def test_f05_block(home):
    lesson = load_lesson("foundations/05-block")
    assert lesson.unlocks == ["block:Block"]
    unlocks.set_policy("guided")
    starter = attempt(lesson)
    assert verdicts(starter) == {"built": True, "same-as-reference": False,
                                 "built-by-hand": True}
    doc = attempt(lesson, solution("foundations/05-block"))
    assert doc["passed"], reasons(doc)
    assert unlocks.read()["unlocks"]["block:Block"]["how"] == "earned"
    assert unlocks.is_unlocked("block:Block")
    # forgetting a residual connection, or sharing one norm for both places, is caught
    no_residual = solution("foundations/05-block").replace(
        "        x = x + self.attn(self.ln1(x))\n", "        x = self.attn(self.ln1(x))\n")
    assert not verdicts(attempt(lesson, no_residual))["same-as-reference"]
    post_norm = solution("foundations/05-block").replace(
        "x = x + self.attn(self.ln1(x))", "x = self.ln1(x + self.attn(x))")
    assert "output differs" in reasons(attempt(lesson, post_norm))["same-as-reference"]


@pytest.mark.usefixtures("fake_data")
def test_f06_gpt2(monkeypatch):
    lesson = load_lesson("foundations/06-gpt2")
    assert lesson.unlocks == ["block:Decoder"]
    unlocks.set_policy("guided")
    small_cpu_preset(monkeypatch, lesson, context_length=16)
    starter = attempt(offline(lesson))
    assert verdicts(starter)["built"] and not starter["passed"]
    assert not unlocks.is_unlocked("block:Decoder")
    doc = attempt(offline(lesson), solution("foundations/06-gpt2"))
    assert doc["passed"], reasons(doc)
    assert unlocks.read()["unlocks"]["block:Decoder"]["how"] == "earned"


@pytest.mark.network
def test_f06_gpt2_real_data(home):
    lesson = load_lesson("foundations/06-gpt2")
    doc = attempt(lesson, solution("foundations/06-gpt2"))
    assert doc["passed"], reasons(doc)
    low, high = doc["checks"][1]["evidence"]["interval"]
    assert low < doc["checks"][1]["evidence"]["value"] < high
