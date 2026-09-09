from __future__ import annotations

import math

import pytest

from nanoscope.eval.statistics import curve_summaries, paired_summaries, summarize


def test_known_student_t_interval_and_small_samples():
    result = summarize([-0.1, -0.2, -0.3])
    margin = 4.302652729911275 * 0.1 / math.sqrt(3)
    assert result["mean"] == pytest.approx(-0.2)
    assert result["sample_sd"] == pytest.approx(0.1)
    assert result["ci95_low"] == pytest.approx(-0.2 - margin)
    assert result["ci95_high"] == pytest.approx(-0.2 + margin)
    for values in ([0.1], [0.1, 0.2], [0.1] * 3):
        assert summarize(values)["ci95_low"] is None
    assert "degenerate" in summarize([0.1] * 3)["status"]


def test_pairing_uses_seed_identity_and_differences():
    rows = [
        {"variant": "base", "seed": seed, "cross_entropy_nats": float(seed)} for seed in [3, 1, 2]
    ]
    rows += [
        {"variant": "new", "seed": seed, "cross_entropy_nats": seed - seed / 10}
        for seed in [2, 3, 1]
    ]
    result = paired_summaries(rows, ["base", "new"], "base")
    assert result[0]["seeds"] == [1, 2, 3]
    assert result[0]["deltas"] == pytest.approx([-0.1, -0.2, -0.3])
    assert result[0]["sample_sd"] == pytest.approx(0.1)
    assert result[1]["relation"] == "previous"
    with pytest.raises(ValueError, match="identical seed"):
        paired_summaries(rows[:-1], ["base", "new"], "base")


def test_curve_bands_use_only_shared_measured_budgets():
    points = [
        {
            "variant": "base",
            "seed": seed,
            "tokens_seen": amount,
            "cumulative_training_flops": amount * 100,
            "cross_entropy_nats": seed / 10,
        }
        for seed in (1, 2, 3)
        for amount in (10, 20)
    ]
    points.pop()  # Third seed lacks budget 20.
    result = curve_summaries(points, ["base"])
    assert [(row["axis"], row["budget"]) for row in result] == [
        ("tokens_seen", 10),
        ("cumulative_training_flops", 1000),
    ]
    assert all(row["n"] == 3 and row["ci95_low"] is not None for row in result)
