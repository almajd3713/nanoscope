import math

import pytest

from nanoscope.statistics import multiple_comparison_note, noise_floor, precision_plan


def test_precision_plan_uses_the_baseline_seed_spread():
    plan = precision_plan("val_bpb", "tinystories-5min", 3)
    assert plan["source"] == "baselines/tinystories-5min"
    assert plan["models"] == ["bigram", "gpt2", "modern"]
    assert 0.005 < plan["sd"] < 0.05  # the shipped seeds differ by a few hundredths of a bit
    t3 = 4.302652729911275  # t(0.975, 2)
    assert plan["half_width"] == pytest.approx(t3 * plan["sd"] / math.sqrt(3))
    more = precision_plan("val_bpb", "tinystories-5min", 12)
    assert more["sd"] == pytest.approx(plan["sd"]) and more["half_width"] < plan["half_width"] / 2
    assert more["half_width"] == pytest.approx(2.200985 * plan["sd"] / math.sqrt(12), rel=1e-5)


def test_precision_plan_says_why_there_is_no_number():
    few = precision_plan("val_bpb", "tinystories-5min", 2)
    assert few["half_width"] is None and "at least 3 seeds" in few["note"]
    assert few["sd"] is None
    none = precision_plan("val_bpb", "no-such-preset", 5)
    assert none["half_width"] is None and none["source"] is None
    assert "no shipped baselines with 3 or more seeds for preset 'no-such-preset'" in none["note"]


def test_noise_floor_is_the_baseline_seed_sd():
    floor = noise_floor("tinystories-5min", "val_bpb")
    plan = precision_plan("val_bpb", "tinystories-5min", 3)
    assert floor["sd"] == pytest.approx(plan["sd"]) and floor["models"] == plan["models"]
    assert f"{floor['sd']:.3f} bpb" in floor["text"]
    none = noise_floor("no-such-preset")
    assert none["sd"] is None and "no shipped baselines" in none["note"]


def test_multiple_comparison_note_starts_above_three_variants():
    assert multiple_comparison_note(3) is None
    note = multiple_comparison_note(4)
    assert "4 variants" in note and "19%" in note
