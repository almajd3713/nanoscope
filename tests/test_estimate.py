import json

import pytest
from fakes import tiny

from nanoscope import paths
from nanoscope.estimate import estimate_seconds
from nanoscope.models import Modern


def bench_row(**kw):
    return {"schema": 1, "model": "Modern", "preset": "test-tiny", "device": "cuda:0",
            "tokens_per_sec": 10_000.0, "at": "2026-10-06T10:00:00+00:00", **kw}


def write_bench(*rows):
    path = paths.hardware_dir() / "bench.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")


def test_estimate_uses_the_measured_speed(home):
    preset = tiny()  # 20 steps x 4 x 32 = 2560 tokens
    write_bench(bench_row(tokens_per_sec=1000.0), bench_row(tokens_per_sec=2560.0))
    est = estimate_seconds(Modern, preset, "cuda:1")
    assert est.source == "bench" and est.seconds == 1.0  # the newest row, same kind of device
    assert "2,560 tokens/s on cuda:0" in est.detail
    assert str(est).startswith("about 1 s (measured")
    assert estimate_seconds("modern", preset, "cuda").seconds == 1.0  # a name works too


def test_estimate_prefers_the_same_preset_and_matches_the_device_and_model(home):
    preset = tiny()
    write_bench(bench_row(preset="test-tiny", tokens_per_sec=2560.0),
                bench_row(preset="other", tokens_per_sec=256.0),
                bench_row(device="cpu", tokens_per_sec=25.6),
                bench_row(model="GPT2", tokens_per_sec=1.0))
    assert estimate_seconds(Modern, preset, "cuda:0").seconds == 1.0
    assert estimate_seconds(Modern, preset, "cpu").seconds == 100.0
    assert estimate_seconds(Modern, tiny(name="never-benched"), "cuda:0").seconds == 10.0


def test_estimate_falls_back_to_the_declared_minutes_then_to_unknown(home):
    declared = estimate_seconds(Modern, tiny(), "cpu", declared_minutes=3)
    assert (declared.source, declared.seconds) == ("declared", 180.0)
    assert str(declared).startswith("about 3 min (the lesson's estimate")
    nothing = estimate_seconds(Modern, tiny(), "cpu")
    assert nothing.source == "unknown" and nothing.seconds is None
    assert "nanoscope bench modern --save" in str(nothing)
    write_bench(bench_row(device="cuda:0"))  # a measurement on another kind of device
    assert estimate_seconds(Modern, tiny(), "cpu", declared_minutes=1).source == "declared"


def test_study_estimate_shares_runs_over_devices(home):
    from nanoscope.estimate import estimate_study

    write_bench(bench_row(device="cuda:0", tokens_per_sec=10_000.0))
    preset = tiny(max_steps=10)  # 10 steps x 4 x 32 tokens
    seconds = 10 * preset.batch_size * preset.context_length / 10_000.0
    runs = [("Modern", preset)] * 4

    one = estimate_study(runs, ["cuda:0"])
    assert one["complete"] and one["wall_seconds"] == pytest.approx(4 * seconds)
    two = estimate_study(runs, ["cuda:0", "cuda:1"])
    assert two["wall_seconds"] == pytest.approx(2 * seconds)
    assert two["total_seconds"] == pytest.approx(4 * seconds)

    mixed = estimate_study([*runs, ("Gpt2Unmeasured", preset)], ["cuda:0"])
    assert not mixed["complete"] and mixed["unknown_runs"] == 1
    assert "left out" in mixed["text"]
    none = estimate_study(runs, ["cpu"])
    assert none["wall_seconds"] is None and "nanoscope bench" in none["text"]


@pytest.mark.usefixtures("fake_data")
def test_study_estimate_from_a_spec_and_from_a_study(home):
    from nanoscope import Study, Tokens
    from nanoscope.estimate import estimate_study, spec_runs
    from nanoscope.models import Bigram

    write_bench(bench_row(model="Bigram", device="cpu", tokens_per_sec=1000.0))
    study = Study("est", preset=tiny(), seeds=2, budget=Tokens(4 * 32 * 5))
    study.add("a", Bigram, d_model=8)
    study.add("b", Bigram, d_model=16)

    from_study = study.estimate(["cpu"])
    assert from_study["runs"] == 4
    assert from_study["wall_seconds"] == pytest.approx(4 * 5 * 128 / 1000)
    runs = spec_runs(study.to_spec())
    assert len(runs) == 4 and runs[0][1].max_steps == 5
    assert estimate_study(runs, ["cpu"])["wall_seconds"] == from_study["wall_seconds"]
