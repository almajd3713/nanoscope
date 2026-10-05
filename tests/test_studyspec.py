import json
import sys
from pathlib import Path

import pytest
from fakes import tiny
from helpers import assert_valid

from nanoscope import Study, Tokens
from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.studyspec import StudySpec

pytestmark = pytest.mark.usefixtures("fake_data")


def toy():
    study = Study("toy", preset=tiny(), seeds=[0, 1, 2], budget=Tokens(4 * 32 * 6),
                  baseline="small", tolerance=0.05)
    study.add("small", Bigram, d_model=8)
    study.add("wide", Bigram, d_model=32)
    study.predict("wide", val_bpb=1.5)
    return study


def test_toml_roundtrip():
    spec = toy().to_spec()
    text = spec.to_toml()
    assert 'name = "toy"' in text and "[[variants]]" in text
    again = StudySpec.from_toml(text)
    assert again == spec
    assert_valid("studyspec", again.to_dict())


def test_from_spec_builds_the_same_study():
    original = toy()
    rebuilt = Study.from_spec(StudySpec.from_toml(original.to_spec().to_toml()))
    assert rebuilt.preset == original.preset  # a custom preset survives, tuples included
    assert rebuilt.seeds == original.seeds and rebuilt.baseline == "small"
    assert rebuilt.tolerance == 0.05 and rebuilt.predictions == original.predictions
    assert {n: (v.model_cls, v.kwargs) for n, v in rebuilt.variants.items()} == {
        n: (v.model_cls, v.kwargs) for n, v in original.variants.items()}
    assert [(j.variant.name, j.seed, j.preset.max_steps) for j in rebuilt.jobs()] == [
        (j.variant.name, j.seed, j.preset.max_steps) for j in original.jobs()]


def test_a_registered_preset_is_named_with_only_its_overrides():
    study = Study("named", preset="tinystories-5min", seeds=2, budget=Tokens(1e6),
                  eval_interval=25)
    study.add("a", Bigram)
    spec = study.to_spec()
    assert spec.preset == "tinystories-5min" and spec.overrides == {"eval_interval": 25}
    assert spec.custom_preset is None
    assert Study.from_spec(spec).preset == study.preset


def test_specs_refuse_what_toml_cannot_hold_and_unknown_keys():
    study = Study("nulls", preset=tiny(), seeds=1)
    study.add("a", Bigram, d_model=None)
    with pytest.raises(ValueError, match=r"variant 'a': d_model=None can't be written"):
        study.to_spec()
    with pytest.raises(ValueError, match="unknown study spec keys: colour"):
        StudySpec.from_toml('name = "x"\ncolour = "red"\n')


def test_match_knob_resolves_widths_like_the_m1_study():
    sys.path.insert(0, str(Path(__file__).parent.parent / "studies"))
    try:
        import m1_ablation
    finally:
        sys.path.pop(0)
    spec = StudySpec.from_toml("""
name = "m1-ablation"
preset = "tinystories-5min"
seeds = [0, 1, 2]
match = "params"
match_knob = "ffn_hidden"
match_to = "gpt2"
range = [64, 1024, 8]
baseline = "modern"
budget = { tokens = 4000000.0 }

[[variants]]
name = "gpt2"
model = "nanoscope.models.gpt2:GPT2"

[[variants]]
name = "modern"
model = "nanoscope.models.modern:Modern"

[[variants]]
name = "no-rope"
model = "nanoscope.models.modern:Modern"
kwargs = { rope = false }

[[variants]]
name = "no-gqa"
model = "nanoscope.models.modern:Modern"
kwargs = { n_kv_heads = 4 }
""")
    declared = Study.from_spec(spec)
    for name in ("modern", "no-rope", "no-gqa"):
        assert declared.variants[name].kwargs["ffn_hidden"] == (
            m1_ablation.study.variants[name].kwargs["ffn_hidden"]), name
    assert declared.variants["no-rope"].kwargs["rope"] is False
    assert "ffn_hidden" not in declared.variants["gpt2"].kwargs
    declared.sizes()  # builds every variant, so match="params" would refuse a mismatch


def test_match_knob_needs_a_range_and_a_target():
    spec = StudySpec(name="x", match_knob="ffn_hidden")
    with pytest.raises(ValueError, match="needs range"):
        Study.from_spec(spec)
    spec = StudySpec.from_toml(
        'name = "x"\nmatch_knob = "ffn_hidden"\nmatch_to = "nope"\nrange = [8, 64, 8]\n'
        '[[variants]]\nname = "a"\nmodel = "nanoscope.models.modern:Modern"\n')
    with pytest.raises(ValueError, match="match_to 'nope' is not a variant"):
        Study.from_spec(spec)


def test_m1_roundtrip():
    sys.path.insert(0, str(Path(__file__).parent.parent / "studies"))
    try:
        import m1_ablation
    finally:
        sys.path.pop(0)
    original = m1_ablation.study
    rebuilt = Study.from_spec(StudySpec.from_toml(original.to_spec().to_toml()))
    assert json.dumps([(j.variant.name, j.variant.model_cls.__name__, j.variant.kwargs, j.seed,
                        j.preset.max_steps, str(j.output_dir)) for j in rebuilt.jobs()]) == (
        json.dumps([(j.variant.name, j.variant.model_cls.__name__, j.variant.kwargs, j.seed,
                     j.preset.max_steps, str(j.output_dir)) for j in original.jobs()]))
    assert GPT2 and Modern  # the refs resolve to these classes


def test_spec_command_prints_toml_that_loads_back(tmp_path, capsys):
    from nanoscope.cli import main

    study_file = Path(__file__).parent.parent / "studies" / "m1_ablation.py"
    main(["spec", str(study_file)])
    text = capsys.readouterr().out
    from nanoscope.studyspec import tomllib  # tomli on Python 3.10

    parsed = tomllib.loads(text)
    assert parsed["name"] == "m1-ablation" and len(parsed["variants"]) == 8
    assert StudySpec.from_toml(text).to_toml() == text
