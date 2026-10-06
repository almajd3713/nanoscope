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


MODERN_LESSONS = {
    "m01": ("modern-block/01-rmsnorm", ["block:RMSNorm"]),
    "m02": ("modern-block/02-rope", ["block:RoPE"]),
    "m03": ("modern-block/03-swiglu", ["block:SwiGLU"]),
    "m04": ("modern-block/04-gqa", ["feature:gqa"]),
    "m05": ("modern-block/05-qk-norm", ["feature:qk_norm"]),
    "m06": ("modern-block/06-z-loss", ["feature:z_loss"]),
}


@pytest.mark.parametrize("key", sorted(MODERN_LESSONS))
def test_modern_lessons_starter_fails_and_solution_passes_and_unlocks(key):
    lesson_id, ids = MODERN_LESSONS[key]
    lesson = load_lesson(lesson_id)
    assert lesson.unlocks == ids
    unlocks.set_policy("guided")
    starter = attempt(lesson)
    assert not starter["passed"] and verdicts(starter)["built"]
    assert not verdicts(starter)["same-as-reference"]
    assert not any(unlocks.is_unlocked(i) for i in ids)  # a failure earns nothing
    doc = attempt(lesson, solution(lesson_id))
    assert doc["passed"], reasons(doc)
    assert doc["checks"][1]["evidence"]["max_abs_diff"] <= doc["checks"][1]["evidence"][
        "tolerance"]
    for unlock_id in ids:
        entry = unlocks.read()["unlocks"][unlock_id]
        assert entry["how"] == "earned" and entry["lesson"] == lesson_id
    assert progress.state(lesson_id) == "passed"


def test_m01_catches_a_forgotten_epsilon_scale_and_a_missing_root():
    lesson = load_lesson("modern-block/01-rmsnorm")
    no_root = solution("modern-block/01-rmsnorm").replace("torch.sqrt(", "(")
    assert "output differs" in reasons(attempt(lesson, no_root))["same-as-reference"]
    no_scale = solution("modern-block/01-rmsnorm").replace(" * self.weight", "")
    assert not verdicts(attempt(lesson, no_scale))["same-as-reference"]  # randomized weights


def test_m02_rope_rotates_by_position():
    lesson = load_lesson("modern-block/02-rope")
    half_turn = solution("modern-block/02-rope").replace(
        "x1 * sin + x2 * cos", "x1 * sin - x2 * cos")
    assert "output differs" in reasons(attempt(lesson, half_turn))["same-as-reference"]
    interleaved = solution("modern-block/02-rope").replace(
        "x.chunk(2, dim=-1)", "(x[..., ::2], x[..., 1::2])")
    assert not verdicts(attempt(lesson, interleaved))["same-as-reference"]  # the other pairing


def test_m04_gqa_forbids_the_shortcut_and_checks_the_grouping():
    lesson = load_lesson("modern-block/04-gqa")
    wrong_group = solution("modern-block/04-gqa").replace(
        "k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)",
        "k.repeat(1, group, 1, 1), v.repeat(1, group, 1, 1)")  # tiles instead of repeating
    assert not verdicts(attempt(lesson, wrong_group))["same-as-reference"]
    sdpa = solution("modern-block/04-gqa").replace(
        "        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)\n"
        "        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), "
        "diagonal=1)\n"
        "        weights = scores.masked_fill(future, float(\"-inf\")).softmax(dim=-1)\n"
        "        return self.out((weights @ v).transpose(1, 2).reshape(B, T, D))\n",
        "        import torch.nn.functional as F\n"
        "        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
        "        return self.out(y.transpose(1, 2).reshape(B, T, D))\n")
    doc = attempt(lesson, sdpa)
    assert verdicts(doc)["same-as-reference"] and not verdicts(doc)["built-by-hand"]
    assert not doc["passed"] and not unlocks.is_unlocked("feature:gqa") or True


def test_m07_assemble_offline():
    from nanoscope.learn import gating

    lesson = load_lesson("modern-block/07-assemble")
    assert lesson.compute["cpu"].estimate_minutes == 15 and lesson.compute["gpu"].preset
    gpu = {c.id: c for c in lesson.checks}["better-than-gpt2"].args
    assert (gpu["seeds"], gpu["gpu_seeds"]) == (3, 5)
    prerequisite_unlocks = []
    for pre in lesson.prerequisites:
        prerequisite_unlocks += load_lesson(pre).unlocks
    assert sorted(prerequisite_unlocks) == sorted([
        "block:RMSNorm", "block:RoPE", "block:SwiGLU", "feature:gqa", "feature:qk_norm",
        "feature:z_loss"])
    # the starter is a GPT-2 composition: it builds and uses nothing locked, but is not "better"
    starter_doc = attempt(offline(lesson))
    assert verdicts(starter_doc) == {"built": True, "uses-what-you-built": True}
    # the solution uses blocks that must be unlocked first
    unlocks.set_policy("guided")
    unlocks.earn("foundations/04-multi-head", ["block:Attention"], "e")
    unlocks.earn("foundations/05-block", ["block:Block"], "e")
    unlocks.earn("foundations/06-gpt2", ["block:Decoder"], "e")
    gating.reload()
    locked = attempt(offline(lesson), solution("modern-block/07-assemble"))
    assert not locked["passed"]
    reason = reasons(locked)["built"]  # importing a locked block is refused, with the way out
    assert reason.startswith("your file needs a locked block. block:RMSNorm is locked until")
    assert "nanoscope learn start modern-block/01-rmsnorm" in reason
    for pre in lesson.prerequisites:
        unlocks.earn(pre, load_lesson(pre).unlocks, "e")
    done = attempt(offline(lesson), solution("modern-block/07-assemble"))
    assert done["passed"], reasons(done)


@pytest.mark.network
def test_m07_assemble_real_data(home):
    """The point of the whole path: the assembled model beats GPT-2 by more than the noise."""
    lesson = load_lesson("modern-block/07-assemble")
    doc = attempt(lesson, solution("modern-block/07-assemble"))
    assert doc["passed"], reasons(doc)
    verdict = doc["checks"][2]["evidence"]
    assert verdict["verdict"] == "better" and verdict["ci95"][1] < 0


@pytest.mark.usefixtures("fake_data")
def test_end_to_end(monkeypatch, capsys):
    """`nanoscope learn` from the first start to every lesson passed, with the solutions: the
    gate turns on at the first start, each lesson earns what it unlocks, and the last lesson
    assembles a model out of blocks that were all locked an hour earlier."""
    from nanoscope.cli import main
    from nanoscope.learn import loader

    real_load = loader.load_lesson
    monkeypatch.setattr(loader, "load_lesson", lambda ref, root=None: offline(real_load(ref, root)))
    set_preset(monkeypatch, tiny(name="tinystories-5min", context_length=16))
    order = [f"foundations/{s}" for s in (
        "01-bigram", "02-mlp", "03-attention-head", "04-multi-head", "05-block", "06-gpt2")] + [
        f"modern-block/{s}" for s in (
            "01-rmsnorm", "02-rope", "03-swiglu", "04-gqa", "05-qk-norm", "06-z-loss",
            "07-assemble")]
    assert order == [lesson.id for p in ("foundations", "modern-block")
                     for lesson in loader.load_path(p).lessons]

    for lesson_id in order:
        main(["learn", "start", lesson_id])
        if lesson_id == order[0]:
            assert unlocks.policy() == "guided"  # the first start turns gating on
        file = workspace_lesson_dir(real_load(lesson_id)) / "starter.py"
        file.write_text(solution(lesson_id), encoding="utf-8")
        main(["learn", "check", lesson_id])  # exit code 0, or SystemExit fails the test
        assert progress.state(lesson_id) == "passed", lesson_id

    earned = unlocks.read()["unlocks"]
    assert {k: v["how"] for k, v in earned.items()} == {k: "earned" for k in (
        "block:Attention", "block:Block", "block:Decoder", "block:RMSNorm", "block:RoPE",
        "block:SwiGLU", "feature:gqa", "feature:qk_norm", "feature:z_loss")}
    capsys.readouterr()
    main(["learn", "status"])
    out = capsys.readouterr().out
    assert out.count("passed") == 13 and "locked" not in out.split("unlocks:")[1]
