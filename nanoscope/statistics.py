"""Seed-level uncertainty. Each seed is one replicate; validation tokens are not."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any


def summarize(values: list[float]) -> dict[str, Any]:
    """Mean with a 95% t-interval. The interval needs at least three seeds."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary requires finite seed-level measurements")
    n = len(values)
    average = mean(values)
    sd = stdev(values) if n > 1 else None
    low = high = None
    status = "exploratory: fewer than three seeds"
    if n >= 3:
        if sd == 0:
            status = "degenerate: no observed seed variation"
        else:
            from scipy.stats import t

            assert sd is not None
            margin = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
            low, high = average - margin, average + margin
            status = "estimated"
    return {
        "n": n,
        "mean": average,
        "sample_sd": sd,
        "ci95_low": low,
        "ci95_high": high,
        "status": status,
    }


def paired_difference(values: list[float], reference: list[float]) -> dict[str, Any]:
    """Difference when both sides used the same seeds: summarize the per-seed deltas."""
    return {**summarize([a - b for a, b in zip(values, reference, strict=True)]), "paired": True}


def unpaired_difference(values: list[float], reference: list[float]) -> dict[str, Any]:
    """Welch's t-interval for a difference of means with unequal variances."""
    diff = mean(values) - mean(reference)
    out: dict[str, Any] = {
        "n": min(len(values), len(reference)), "mean": diff, "sample_sd": None,
        "ci95_low": None, "ci95_high": None, "paired": False,
        "status": "exploratory: fewer than three seeds",
    }
    if len(values) < 3 or len(reference) < 3:
        return out
    va, vb = stdev(values) ** 2 / len(values), stdev(reference) ** 2 / len(reference)
    if va + vb == 0:
        return {**out, "status": "degenerate: no observed seed variation"}
    from scipy.stats import t

    df = (va + vb) ** 2 / (va**2 / (len(values) - 1) + vb**2 / (len(reference) - 1))
    margin = float(t.ppf(0.975, df)) * math.sqrt(va + vb)
    return {**out, "ci95_low": diff - margin, "ci95_high": diff + margin, "status": "estimated"}


def score_prediction(
    actual_mean: float, ci95: tuple[float, float] | None, *, low: float | None = None,
    high: float | None = None, verdict: str | None = None, actual_verdict: str | None = None,
    max_width_ratio: float = 10.0,
) -> dict[str, Any]:
    """Score a prediction made before the run against what the run found.

    A predicted interval [low, high] is a hit when it contains the observed mean difference.
    It is *sharp enough* when it is no wider than `max_width_ratio` times the observed 95%
    interval (a prediction that covers everything predicts nothing). A predicted verdict word
    is right when it equals the observed one. `passed` needs every part that was predicted."""
    parts: dict[str, Any] = {}
    if low is not None and high is not None:
        hit = low <= actual_mean <= high
        width = high - low
        observed_width = None if ci95 is None else ci95[1] - ci95[0]
        sharp = None if not observed_width else width <= max_width_ratio * observed_width
        parts.update(hit=hit, width=width, observed_width=observed_width, sharp=sharp)
    if verdict is not None:
        parts["verdict_right"] = verdict == actual_verdict
    checks = [v for k, v in parts.items() if k in ("hit", "sharp", "verdict_right")
              and v is not None]
    return {"passed": bool(checks) and all(checks), "actual_mean": actual_mean,
            "ci95": None if ci95 is None else list(ci95), **parts}
