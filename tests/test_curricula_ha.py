"""The honest-ablations path: every starter fails its checks and every solution passes.

Offline on the synthetic stories (`fake_data`) with a tiny preset. The statements that only
mean something on real text (what the A/A spread is, which component is null) are covered by
the network-marked tests at the end, which train for real on TinyStories.
"""

import dataclasses

import pytest
from test_curricula import attempt, reasons, small_cpu_preset, solution, verdicts

from nanoscope.learn import gating
from nanoscope.learn.cli import start_lesson
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


def without_verdicts(lesson):
    """The lesson minus its real-text expectations: on the synthetic stories a 'better' or
    'within noise' verdict means nothing, so the structure is tested here and the verdicts in
    the network-marked tests."""
    return dataclasses.replace(lesson, checks=[c for c in lesson.checks if c.kind != "verdict"])


@pytest.mark.usefixtures("fake_data")
def test_ha02_starter_fails_and_solution_passes(monkeypatch):
    lesson = load_lesson("honest-ablations/02-paired")
    assert [c.kind for c in lesson.checks] == [
        "function", "function", "defines", "verdict", "verdict"]
    assert [c.args.get("pairing") for c in lesson.checks if c.kind == "verdict"] == [
        "paired", "unpaired"]
    small_cpu_preset(monkeypatch, lesson)
    starter = attempt(without_verdicts(lesson))
    assert not verdicts(starter)["paired-formula"] and not verdicts(starter)["unpaired-formula"]
    assert "raised NotImplementedError" in reasons(starter)["paired-formula"]
    done = attempt(without_verdicts(lesson), solution("honest-ablations/02-paired"))
    assert done["passed"], reasons(done)
    assert "largest difference" in reasons(done)["paired-formula"]


@pytest.mark.usefixtures("fake_data")
def test_ha02_names_the_part_that_is_off(monkeypatch):
    lesson = without_verdicts(load_lesson("honest-ablations/02-paired"))
    small_cpu_preset(monkeypatch, load_lesson("honest-ablations/02-paired"))
    wrong_df = solution("honest-ablations/02-paired").replace(
        "t.ppf(0.975, n - 1)", "t.ppf(0.975, n)")  # the classic: n instead of n - 1
    doc = attempt(lesson, wrong_df)
    assert not verdicts(doc)["paired-formula"] and verdicts(doc)["unpaired-formula"]
    assert reasons(doc)["paired-formula"].startswith("case 1, paired_ci(")
    assert "low is" in reasons(doc)["paired-formula"]


def test_ha03_starter_is_a_third_too_big_and_the_solution_is_matched(monkeypatch):
    lesson = load_lesson("honest-ablations/03-matching")
    assert [c.kind for c in lesson.checks] == ["defines", "matches", "matches"]
    starter = attempt(lesson)
    assert verdicts(starter) == {"built": True, "same-params": False, "same-flops": False}
    assert "(+36." in reasons(starter)["same-params"] or "(+35." in reasons(starter)["same-params"]
    assert "MySwiGLU has more" in reasons(starter)["same-params"]
    done = attempt(lesson, solution("honest-ablations/03-matching"))
    assert done["passed"], reasons(done)
    assert "matched" in reasons(done)["same-params"]


def test_ha03_too_small_is_also_unfair(monkeypatch):
    lesson = load_lesson("honest-ablations/03-matching")
    small = solution("honest-ablations/03-matching").replace("ffn_hidden=171", "ffn_hidden=100")
    doc = attempt(lesson, small)
    assert not verdicts(doc)["same-params"]
    assert "MySwiGLU has fewer" in reasons(doc)["same-params"]


@pytest.mark.usefixtures("fake_data")
def test_ha04_needs_a_prediction_first_and_a_miss_still_counts(monkeypatch):
    from nanoscope.learn import progress
    from nanoscope.learn.checks import PredictionTooLate, commit_prediction

    lesson = quick(load_lesson("honest-ablations/04-predict"))
    assert [c.kind for c in lesson.checks] == ["defines", "verdict", "predicted"]
    assert lesson.checks[2].args["must_hit"] is False
    small_cpu_preset(monkeypatch, lesson)
    # no prediction: the experiment still runs, but the lesson cannot pass
    first = attempt(lesson, solution("honest-ablations/04-predict"))
    assert "no prediction was recorded" in reasons(first)["committed-first"]
    assert not first["passed"]
    # and now it is too late to predict for this lesson
    with pytest.raises(PredictionTooLate):
        commit_prediction(lesson, {"verdict": "better"})
    assert progress.entry(lesson.id)["first_checked_at"] is not None


@pytest.mark.usefixtures("fake_data")
def test_ha04_a_prediction_made_first_passes_even_if_the_verdict_differs(monkeypatch):
    from nanoscope.learn.checks import commit_prediction

    lesson = quick(load_lesson("honest-ablations/04-predict"))
    small_cpu_preset(monkeypatch, lesson)
    start_lesson(lesson)
    commit_prediction(lesson, {"verdict": "worse", "note": "a guess on the wrong side"})
    doc = attempt(lesson, solution("honest-ablations/04-predict"))
    assert verdicts(doc)["committed-first"], reasons(doc)
    assert reasons(doc)["committed-first"].startswith("your prediction was recorded first")
    # the starter is the same model twice: the experiment compares a model with itself
    starter = attempt(lesson)
    assert "NoRoPE" in reasons(starter)["rope-gap"] or "identical" in reasons(starter)["rope-gap"]


def capstone_study(monkeypatch, lesson, seeds=3):
    """The eight variants of the lesson's study, as tiny Bigram widths so they train in seconds:
    'no-zloss' is almost the full model (a null component), the others are much worse."""
    from fakes import tiny

    from nanoscope import Study
    from nanoscope.models import Bigram

    preset = tiny(name=lesson.compute["cpu"].preset, max_steps=30)
    set_preset(monkeypatch, preset)
    study = Study("ha05-capstone", preset=preset, seeds=seeds)
    widths = {"gpt2": 2, "modern": 32, "no-rope": 4, "no-qk-norm": 5, "no-swiglu": 6,
              "no-rmsnorm": 7, "no-zloss": 33, "no-gqa": 8}
    for name, width in widths.items():
        study.add(name, Bigram, d_model=width)
    return study


def set_preset(monkeypatch, preset):
    from learn_helpers import set_preset as install

    install(monkeypatch, preset)


@pytest.mark.usefixtures("fake_data")
def test_ha05_reads_your_study_and_holds_your_answers_against_it(monkeypatch):
    lesson = load_lesson("honest-ablations/05-capstone")
    assert lesson.checks[0].kind == "ablation"
    assert lesson.compute["cpu"].estimate_minutes > 15 and lesson.compute["gpu"].preset
    assert lesson.checks[0].args["seeds"] == 3 and lesson.checks[0].args["gpu_seeds"] == 5
    original = solution("honest-ablations/05-capstone")

    def answers(low, high, null):
        return (original.replace("GAP = (-1.0, 0.0)", f"GAP = ({low}, {high})")
                .replace('NULL_COMPONENT = "no-zloss"', f"NULL_COMPONENT = {null}"))

    # the starter has nothing filled in
    starter = attempt(lesson)
    assert not starter["passed"] and "GAP is not set" in reasons(starter)["gap-and-null"]
    assert "NULL_COMPONENT must be one of no-rope" in reasons(starter)["gap-and-null"]

    # answers without a study: it says what to run
    blank = attempt(lesson, answers(-1.0, 0.0, '"no-zloss"'))
    assert "has no variant" in reasons(blank)["gap-and-null"]

    capstone_study(monkeypatch, lesson).run(devices=["cpu"])
    probe = attempt(lesson, answers(-1e3, 1e3, '"no-zloss"'))
    delta = probe["checks"][0]["evidence"]["delta"]
    done = attempt(lesson, answers(delta - 0.1, delta + 0.1, '"no-zloss"'))
    assert done["passed"], reasons(done)
    evidence = done["checks"][0]["evidence"]
    assert evidence["drops"]["no-zloss"] == "within noise"
    assert evidence["drops"]["no-rope"] == "worse"

    wrong = attempt(lesson, answers(delta - 0.1, delta + 0.1, '"no-rope"'))
    assert not wrong["passed"] and "no-rope is not null" in reasons(wrong)["gap-and-null"]


def lesson_study(lesson_dir, path="honest-ablations", edits=()):
    """The study code printed in a lesson's lesson.md, run as the learner's studies file."""
    import re
    from pathlib import Path

    from nanoscope.learn.loader import curricula_dir

    text = (Path(curricula_dir()) / path / lesson_dir / "lesson.md").read_text()
    code = re.search(r"```python\n# studies/\w+\.py\n(.*?)```", text, re.S).group(1)
    for old, new in edits:
        assert old in code, old
        code = code.replace(old, new)
    namespace: dict = {}
    exec(compile(code, f"{lesson_dir}-study.py", "exec"), namespace)
    return namespace["study"]


def test_ha05_study_in_the_lesson_text_builds_eight_matched_variants():
    study = lesson_study("05-capstone")
    assert list(study.variants) == [
        "gpt2", "modern", "no-rope", "no-qk-norm", "no-swiglu", "no-rmsnorm", "no-zloss",
        "no-gqa"]
    lesson = load_lesson("honest-ablations/05-capstone")
    assert list(study.variants)[2:] == lesson.checks[0].args["drops"]
    sizes = study.sizes()
    base = sizes["gpt2"]["non_embedding_params"]
    for name, row in sizes.items():
        assert abs(row["non_embedding_params"] - base) / base < 0.02, name


@pytest.mark.network
def test_ha01_real_data(home):
    lesson = load_lesson("honest-ablations/01-seed-noise")
    doc = attempt(lesson, solution("honest-ablations/01-seed-noise"))
    assert doc["passed"], reasons(doc)
    evidence = doc["checks"][2]["evidence"]
    assert evidence["verdict"] == "within noise" and evidence["paired"] is False
    low, high = evidence["ci95"]
    print(f"HA01 A/A interval: {evidence['delta']:+.4f} [{low:+.4f}, {high:+.4f}]")


@pytest.mark.network
def test_ha02_real_data_pairing_sees_what_unpaired_misses(home):
    lesson = load_lesson("honest-ablations/02-paired")
    doc = attempt(lesson, solution("honest-ablations/02-paired"))
    assert doc["passed"], reasons(doc)
    paired, unpaired = doc["checks"][3]["evidence"], doc["checks"][4]["evidence"]
    assert paired["verdict"] == "better" and unpaired["verdict"] == "within noise"
    pw, uw = paired["ci95"][1] - paired["ci95"][0], unpaired["ci95"][1] - unpaired["ci95"][0]
    print(f"HA02 paired {paired['delta']:+.4f} width {pw:.4f}; "
          f"unpaired {unpaired['delta']:+.4f} width {uw:.4f}")
    assert pw < uw / 3


@pytest.mark.network
def test_ha04_real_data_rope_helps_and_the_prediction_is_scored(home):
    from nanoscope.learn.checks import commit_prediction

    lesson = load_lesson("honest-ablations/04-predict")
    start = attempt  # noqa: F841 - the starter must exist before a prediction is recorded
    from nanoscope.learn.cli import start_lesson

    start_lesson(lesson)
    commit_prediction(lesson, {"verdict": "better", "low": -0.4, "high": -0.02, "note": "test"})
    doc = attempt(lesson, solution("honest-ablations/04-predict"))
    assert doc["passed"], reasons(doc)
    print("HA04", reasons(doc)["rope-gap"], "|", reasons(doc)["committed-first"])


@pytest.mark.network
def test_ha05_real_data_the_study_in_the_lesson_text(home, monkeypatch):
    lesson = load_lesson("honest-ablations/05-capstone")
    study = lesson_study("05-capstone")
    study.run(devices=["cpu"])
    original = solution("honest-ablations/05-capstone")

    def answers(low, high, null):
        return (original.replace("GAP = (-1.0, 0.0)", f"GAP = ({low}, {high})")
                .replace('NULL_COMPONENT = "no-zloss"', f"NULL_COMPONENT = {null}"))

    probe = attempt(lesson, answers(-1e3, 1e3, "None"))
    evidence = probe["checks"][0]["evidence"]
    print("HA05 gap", evidence["delta"], evidence["ci95"], evidence["drops"])
    nulls = [n for n, v in evidence["drops"].items() if v == "within noise"]
    delta = evidence["delta"]
    done = attempt(lesson, answers(delta - 0.05, delta + 0.05, repr(nulls[0]) if nulls else "None"))
    assert done["passed"], reasons(done)


# -- the other paths --------------------------------------------------------------------------

def function_lesson_passes(monkeypatch, lesson_id, solution_id):
    """A lesson made of `function` checks: the starter raises, the solution passes."""
    lesson = load_lesson(lesson_id)
    assert {c.kind for c in lesson.checks} == {"function"}
    starter = attempt(lesson)
    assert not starter["passed"]
    assert all("raised NotImplementedError" in r for r in reasons(starter).values())
    done = attempt(lesson, solution(solution_id))
    assert done["passed"], reasons(done)
    return done


def test_ev01_eval_noise(monkeypatch):
    function_lesson_passes(monkeypatch, "evaluation/01-eval-noise", "evaluation/01-eval-noise")


def test_ev02_variance(monkeypatch):
    done = function_lesson_passes(monkeypatch, "evaluation/02-variance", "evaluation/02-variance")
    assert "largest difference" in reasons(done)["split"]
    # dividing by the wrong count is the usual slip: it is named
    lesson = load_lesson("evaluation/02-variance")
    wrong = solution("evaluation/02-variance").replace("(rows - 1)", "rows")
    doc = attempt(lesson, wrong)
    assert not doc["passed"] and "init is" in reasons(doc)["split"]


def test_sc01_flops(monkeypatch):
    function_lesson_passes(monkeypatch, "scaling-lite/01-flops", "scaling-lite/01-flops")
    lesson = load_lesson("scaling-lite/01-flops")
    wrong = solution("scaling-lite/01-flops").replace("6 * n_params * tokens",
                                                       "2 * n_params * tokens")
    doc = attempt(lesson, wrong)
    assert not verdicts(doc)["cost"] and verdicts(doc)["optimal"]


@pytest.mark.usefixtures("fake_data")
def test_sc02_ladder_two_steps(monkeypatch):
    from fakes import tiny

    lesson = load_lesson("scaling-lite/02-ladder")
    set_preset(monkeypatch, tiny(name=lesson.compute["cpu"].preset, max_steps=30))
    original = solution("scaling-lite/02-ladder")

    def answer(low, high):
        return original.replace("PREDICTION = (0.0, 1.0)", f"PREDICTION = ({low}, {high})")

    starter = attempt(lesson)
    assert "PREDICTION is not set" in reasons(starter)["extrapolate"]
    # step 1: only the three smaller sizes exist
    lesson_study("02-ladder", "scaling-lite").run(devices=["cpu"])
    early = attempt(lesson, answer(0.1, 9.0))
    assert "has no variant w96" in reasons(early)["extrapolate"]
    # step 2: set HOLDOUT and run again; finished runs are skipped
    full = lesson_study("02-ladder", "scaling-lite", [("HOLDOUT = False", "HOLDOUT = True")])
    assert list(full.variants) == ["w32", "w48", "w64", "w96"]
    full.run(devices=["cpu"])
    probe = attempt(lesson, answer(0.01, 100.0))
    measured = probe["checks"][0]["evidence"]["measured"]
    done = attempt(lesson, answer(measured - 0.2, measured + 0.2))
    assert done["passed"], reasons(done)
    assert "It landed" in reasons(done)["extrapolate"]


def test_ef01_bench(monkeypatch):
    function_lesson_passes(monkeypatch, "efficiency/01-bench", "efficiency/01-bench")
    lesson = load_lesson("efficiency/01-bench")
    # forgetting the cap, and flipping the comparison, are the usual slips
    uncapped = solution("efficiency/01-bench").replace("min(kernel_ms / step_ms, 1.0)",
                                                       "kernel_ms / step_ms")
    assert not verdicts(attempt(lesson, uncapped))["busy"]
    flipped = solution("efficiency/01-bench").replace("< threshold", "> threshold")
    assert not verdicts(attempt(lesson, flipped))["bound"]


def test_ef02_compile(monkeypatch):
    function_lesson_passes(monkeypatch, "efficiency/02-compile", "efficiency/02-compile")
    lesson = load_lesson("efficiency/02-compile")
    # counting only the steps, not the compile time
    free = solution("efficiency/02-compile").replace("compile_seconds + steps", "steps")
    assert not verdicts(attempt(lesson, free))["decision"]


def test_ef04_roofline(monkeypatch):
    function_lesson_passes(monkeypatch, "efficiency/04-roofline", "efficiency/04-roofline")
    lesson = load_lesson("efficiency/04-roofline")
    # forgetting that bandwidth is in GB/s and the result in TFLOP/s
    units = solution("efficiency/04-roofline").replace("bandwidth_gbs / 1000", "bandwidth_gbs")
    assert not verdicts(attempt(lesson, units))["roofline"]


@pytest.mark.network
def test_sc02_real_data_ladder_and_holdout(home):
    lesson = load_lesson("scaling-lite/02-ladder")
    original = solution("scaling-lite/02-ladder")
    swap = [("HOLDOUT = False", "HOLDOUT = True")]
    lesson_study("02-ladder", "scaling-lite", swap).run(devices=["cpu"])

    def answer(low, high):
        return original.replace("PREDICTION = (0.0, 1.0)", f"PREDICTION = ({low}, {high})")

    probe = attempt(lesson, answer(0.01, 100.0))
    evidence = probe["checks"][0]["evidence"]
    print("SC02", {k: evidence[k] for k in evidence if k != "points"})
    measured = evidence["measured"]
    done = attempt(lesson, answer(measured - 0.2, measured + 0.2))
    assert done["passed"], reasons(done)


def test_in01_induction(monkeypatch):
    lid = "interpretability/01-induction"
    function_lesson_passes(monkeypatch, lid, lid)
    lesson = load_lesson(lid)
    # the off-by-one slip: attending to the twin itself, not the token after it
    wrong = solution(lid).replace("i - half + 1", "i - half")
    assert not verdicts(attempt(lesson, wrong))["induction"]


def test_in02_logit_lens(monkeypatch):
    lid = "interpretability/02-logit-lens"
    function_lesson_passes(monkeypatch, lid, lid)
    lesson = load_lesson(lid)
    # forgetting "and stays": the first layer that ever guesses right
    wrong = solution(lid).replace("all(top_token(row) == target for row in layer_logits[k:])",
                                  "top_token(layer_logits[k]) == target")
    assert not verdicts(attempt(lesson, wrong))["settled"]


def test_ex01_certified_block(monkeypatch):
    lid = "extend/01-certified-block"
    lesson = load_lesson(lid)
    assert not attempt(lesson)["passed"]
    done = attempt(lesson, solution(lid))
    assert done["passed"], reasons(done)
    # forgetting the reference: registered, but nothing to certify against
    bare = solution(lid).replace("@register_block(reference=squared_relu_mlp, family=\"mlp\")",
                                 "@register_block(family=\"mlp\")")
    assert "not registered with a reference" in reasons(attempt(lesson, bare))["certified"]
    # one wrong formula in the block and the reference passes its own check, not the lesson's
    twice = solution(lid).replace("(h * h)", "h")
    assert "not the lesson's" in reasons(attempt(lesson, twice))["certified"]


def test_ex02_preset(monkeypatch):
    import nanoscope.presets as presets

    lid = "extend/02-preset"
    lesson = load_lesson(lid)
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    assert "no preset named" in reasons(attempt(lesson))["registered"]
    done = attempt(lesson, solution(lid))
    assert done["passed"], reasons(done)
    # override of the steps only: warmup is still 50 of 60 steps
    lazy = solution(lid).replace("warmup_steps=10,", "")
    assert "warmup_steps is 50 of 60" in reasons(attempt(lesson, lazy))["registered"]
    # changing a data field breaks the cached tokens
    moved = solution(lid).replace("max_steps=60,", "max_steps=60, context_length=128,")
    assert "context_length" in reasons(attempt(lesson, moved))["registered"]
    # a run too short for its evaluation interval never reports a loss
    never = solution(lid).replace("max_steps=60", "max_steps=30").replace(
        "warmup_steps=10", "warmup_steps=5").replace("eval_interval=20", "eval_interval=50")
    assert "never evaluate" in reasons(attempt(lesson, never))["registered"]


def test_ex04_moe(monkeypatch):
    lid = "extend/04-moe"
    lesson = load_lesson(lid)
    assert lesson.unlocks == ["block:MoE"]
    starter = attempt(lesson)
    assert not starter["passed"]
    done = attempt(lesson, solution(lid))
    assert done["passed"], reasons(done)
    # raw probabilities, not rescaled to sum to 1
    raw = solution(lid).replace("weight = weight / weight.sum(-1, keepdim=True)", "pass")
    assert "differs from the reference" in reasons(attempt(lesson, raw))["same-as-reference"]
