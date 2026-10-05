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
