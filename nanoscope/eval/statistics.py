"""Seed-level uncertainty; checkpoints and validation tokens are not replicates."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any


def summarize(values: list[float]) -> dict[str, Any]:
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
            try:
                from scipy.stats import t
            except ImportError as exc:
                raise RuntimeError("confidence intervals require: uv sync --extra eval") from exc
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


def paired_summaries(
    rows: list[dict[str, Any]], variants: list[str], baseline: str
) -> list[dict[str, Any]]:
    by_variant = {
        name: {row["seed"]: row for row in rows if row["variant"] == name} for name in variants
    }
    summaries = []
    for index, name in enumerate(variants):
        for relation, reference in (
            ("baseline", baseline),
            ("previous", variants[index - 1] if index else None),
        ):
            if reference is None or reference == name:
                continue
            seeds = sorted(by_variant[name])
            if seeds != sorted(by_variant[reference]):
                raise ValueError("paired summaries require identical seed sets")
            deltas = [
                by_variant[name][seed]["cross_entropy_nats"]
                - by_variant[reference][seed]["cross_entropy_nats"]
                for seed in seeds
            ]
            summaries.append(
                {
                    "variant": name,
                    "reference": reference,
                    "relation": relation,
                    "seeds": seeds,
                    "deltas": deltas,
                    **summarize(deltas),
                }
            )
    return summaries


def curve_summaries(curves: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for name in variants:
        points = [row for row in curves if row["variant"] == name]
        seeds = sorted({row["seed"] for row in points})
        for axis in ("tokens_seen", "cumulative_training_flops"):
            grids = [
                {row[axis] for row in points if row["seed"] == seed and row[axis] is not None}
                for seed in seeds
            ]
            for amount in sorted(set.intersection(*grids)):
                values = []
                for seed in seeds:
                    matches = [row for row in points if row["seed"] == seed and row[axis] == amount]
                    if len(matches) != 1:
                        raise ValueError("ambiguous curve measurements at one seed/budget")
                    values.append(matches[0]["cross_entropy_nats"])
                summaries.append(
                    {
                        "variant": name,
                        "axis": axis,
                        "budget": amount,
                        "seeds": seeds,
                        **summarize(values),
                    }
                )
    return summaries
