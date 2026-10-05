from dataclasses import fields

import pytest
from helpers import assert_valid

from nanoscope import Preset, get_preset
from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.specs import ModelSpec, PresetSpec, Problem, validate_run_request


@pytest.mark.parametrize("cls", [Bigram, GPT2, Modern])
def test_model_spec_lists_params_defaults_and_from_data_flags(cls):
    spec = ModelSpec.from_class(cls)
    assert spec.name == cls.__name__ and spec.ref.endswith(f":{cls.__name__}")
    by_name = {p.name: p for p in spec.params}
    assert by_name["vocab_size"].from_data and by_name["vocab_size"].required
    if "context_length" in by_name:  # Bigram doesn't take one
        assert by_name["context_length"].from_data
    tunable = {p.name for p in spec.tunable()}
    assert "vocab_size" not in tunable and "d_model" in tunable
    d_model = by_name["d_model"]
    assert d_model.default is not None and d_model.annotation == "int" and not d_model.required
    assert spec.to_dict()["params"][0]["name"] == spec.params[0].name


def test_model_spec_carries_the_docstring():
    import inspect

    for cls in (Bigram, GPT2, Modern):
        assert ModelSpec.from_class(cls).doc == (inspect.getdoc(cls) or "")


def test_every_preset_field_has_help():
    for f in fields(Preset):
        assert f.metadata.get("help"), f"{f.name} has no help text"
    spec = PresetSpec.from_preset(get_preset("tinystories-5min"))
    assert [f.name for f in spec.fields] == [f.name for f in fields(Preset)]
    by_name = {f.name: f for f in spec.fields}
    assert by_name["max_steps"].required and not by_name["weight_decay"].required
    assert by_name["weight_decay"].default == 0.1


def test_problem_schema():
    problem = Problem("unknown_keyword", "d_modle", "no such parameter", "did you mean d_model?")
    assert_valid("problem", problem.to_dict())
    assert_valid("problem", Problem("bad_seeds", None, "nope").to_dict())


def test_three_problems_come_back_at_once():
    problems = validate_run_request(
        Bigram, "tinystories-5min", {"d_modle": 8, "learning_rate": "fast"}, seeds=0)
    assert [(p.code, p.field) for p in problems] == [
        ("unknown_keyword", "d_modle"), ("wrong_type", "learning_rate"), ("bad_seeds", "seeds")]
    assert "learning_rate must be float, got str 'fast'" in problems[1].message
    for problem in problems:
        assert_valid("problem", problem.to_dict())


def test_unknown_preset_lists_the_choices():
    (problem,) = validate_run_request(Bigram, "no-such-preset", {})
    assert problem.code == "unknown_preset" and "tinystories-5min" in problem.hint


@pytest.mark.parametrize("seeds", [3, [0, 1], (4,)])
def test_good_requests_have_no_problems(seeds):
    assert validate_run_request(Bigram, "tinystories-5min",
                                {"d_model": 16, "learning_rate": 1, "train_docs": None},
                                seeds=seeds) == []


@pytest.mark.parametrize("seeds", [0, -1, [], [1.5], "3", True])
def test_bad_seeds_are_flagged(seeds):
    (problem,) = validate_run_request(Bigram, "tinystories-5min", {}, seeds=seeds)
    assert problem.code == "bad_seeds"
