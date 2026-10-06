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


def reproduction_interval(baseline: list[float], n_new: int = 1) -> tuple[float, float] | None:
    """Where a new result should land if it reproduces a baseline of several seeds: the
    baseline mean +- t * sd * sqrt(1/n_new + 1/n). A single new run is compared with a
    prediction interval, not the (much narrower) interval of the baseline's mean. None when
    the baseline has fewer than three seeds."""
    n = len(baseline)
    if n < 3:
        return None
    average, sd = mean(baseline), stdev(baseline)
    from scipy.stats import t

    margin = float(t.ppf(0.975, n - 1)) * sd * math.sqrt(1 / n_new + 1 / n)
    return average - margin, average + margin


def precision_plan(metric: str, preset: str, n_seeds: int) -> dict[str, Any]:
    """How precisely `n_seeds` seeds would pin down a mean on this preset: the expected half-width
    of a 95% interval, t * s / sqrt(n), where s is the seed-to-seed spread of the shipped
    baselines (pooled over the models that have 3 or more seeds). Before spending compute, this
    says whether a difference you hope to see is even detectable.

    Without baselines for the preset there is no spread to borrow: `half_width` is None."""
    import sys

    from scipy.stats import t

    import nanoscope.compare  # noqa: F401  (loads the module; `nanoscope.compare` the name is a function)

    compare = sys.modules["nanoscope.compare"]
    plan: dict[str, Any] = {"metric": metric, "preset": preset, "n_seeds": n_seeds, "sd": None,
                            "half_width": None, "source": None, "models": []}
    if n_seeds < 3:
        plan["note"] = "a confidence interval needs at least 3 seeds"
        return plan
    root = compare.BASELINES_DIR / preset
    variances, models = [], []
    for folder in sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []:
        try:
            values = [r.final(metric) for r in compare.load_runs(folder)]
        except (OSError, KeyError, ValueError):
            continue
        if len(values) >= 3:
            variances.append(stdev(values) ** 2)
            models.append(folder.name)
    if not variances:
        plan["note"] = f"no shipped baselines with 3 or more seeds for preset {preset!r}"
        return plan
    sd = math.sqrt(sum(variances) / len(variances))
    plan.update(sd=sd, source=f"baselines/{preset}", models=models)
    plan["half_width"] = float(t.ppf(0.975, n_seeds - 1)) * sd / math.sqrt(n_seeds)
    return plan
