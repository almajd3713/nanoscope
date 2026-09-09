from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any

from nanoscope.eval.artifacts import fingerprint
from nanoscope.eval.config import check_keys, read_yaml, relative_path
from nanoscope.eval.runner import load_result
from nanoscope.eval.statistics import curve_summaries, paired_summaries


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def recipe_differences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Ignore run identity and operational paths, but retain scientific recipe fields."""
    sections = ("data", "tokenizer", "model", "optimizer", "scheduler")
    a = {key: left[key] for key in sections}
    b = {key: right[key] for key in sections}
    training_fields = (
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "precision",
        "deterministic",
        "grad_clip",
    )
    a["train"] = {key: left["train"][key] for key in training_fields}
    b["train"] = {key: right["train"][key] for key in training_fields}
    flat_a, flat_b = _flatten(a), _flatten(b)
    return {
        key: {"baseline": flat_a.get(key), "variant": flat_b.get(key)}
        for key in sorted(flat_a.keys() | flat_b.keys())
        if flat_a.get(key) != flat_b.get(key)
    }


def _same_score_profile(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("corpus_hash", "evaluator_version", "evaluator_code_hash", "precision"):
        if left["identity"][key] != right["identity"][key]:
            raise ValueError(f"incompatible evaluation {key}")
    # Different batching/devices are recorded, but not different arithmetic policies.
    for key in (
        "torch",
        "float32_matmul_precision",
        "cudnn_allow_tf32",
        "deterministic_algorithms",
    ):
        if left["identity"]["runtime"].get(key) != right["identity"]["runtime"].get(key):
            raise ValueError(f"incompatible evaluation runtime setting: {key}")


def _speed_profile(result: dict[str, Any]) -> dict[str, Any]:
    config = result["run"]["config"]
    provenance = result["run"]["provenance"]
    return {
        "hardware": {
            key: provenance.get(key)
            for key in ("device", "gpu", "world_size", "torch", "cuda", "cudnn")
        },
        "precision": config["train"]["precision"],
        "deterministic": config["train"]["deterministic"],
        "batch_size": config["train"]["batch_size"],
        "accumulation": config["train"]["gradient_accumulation_steps"],
        "timing": result["training"]["timing_definition"],
    }


def build_comparison(study_path: str | Path) -> dict[str, Any]:
    study = read_yaml(study_path)
    check_keys(
        study,
        {
            "name",
            "baseline",
            "axis",
            "budget",
            "tolerance",
            "parameter_tolerance",
            "allow_changes",
            "variants",
        },
        {"name", "baseline", "axis", "budget", "variants"},
    )
    axis = {"tokens": "tokens_seen", "flops": "cumulative_training_flops"}.get(study["axis"])
    if axis is None:
        raise ValueError("study axis must be tokens or flops")
    budget = study["budget"]
    if type(budget) not in {int, float} or not math.isfinite(budget) or budget <= 0:
        raise ValueError("study budget must be finite and positive")
    tolerance = study.get("tolerance", 0.0)
    parameter_tolerance = study.get("parameter_tolerance")
    for name, value in (("tolerance", tolerance), ("parameter_tolerance", parameter_tolerance)):
        if value is not None and (
            type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value < 1
        ):
            raise ValueError(f"{name} must be a fraction in [0, 1)")
    if axis == "cumulative_training_flops" and parameter_tolerance is None:
        raise ValueError("FLOP comparisons require an explicit parameter_tolerance")
    allowed = study.get("allow_changes", ["model"])
    if not isinstance(allowed, list) or any(not isinstance(key, str) or not key for key in allowed):
        raise ValueError("allow_changes must be a list of config field paths")
    variants = study["variants"]
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("a comparison requires at least two variants")
    selected: dict[str, dict[int, dict[str, Any]]] = {}
    curves: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("each variant must be a mapping")
        check_keys(variant, {"name", "results"}, {"name", "results"})
        label = variant["name"]
        if not isinstance(label, str) or not label or label in selected:
            raise ValueError("variant names must be non-empty and unique")
        if not isinstance(variant["results"], list) or not variant["results"]:
            raise ValueError(f"{label}: results must be a non-empty list of paths/globs")
        paths: set[Path] = set()
        for pattern in variant["results"]:
            resolved = relative_path(pattern, study_path)
            matches = (
                sorted(resolved.glob("*.json"))
                if resolved.is_dir()
                else [Path(item) for item in sorted(glob.glob(str(resolved)))]
            )
            if not matches:
                raise ValueError(f"{label}: no evaluation results match {pattern}")
            paths.update(matches)
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for path in sorted(paths):
            result = load_result(path)
            if not result["training"]["held_out_verified"]:
                raise ValueError(f"{label}: held-out exclusion is unverified for {path}")
            inputs[str(path)] = result["result_hash"]
            seed = result["run"]["seed"]
            by_seed.setdefault(seed, []).append(result)
        selected[label] = {}
        for seed, results in sorted(by_seed.items()):
            if len({row["run"]["id"] for row in results}) != 1:
                raise ValueError(f"{label}: duplicate seed {seed} across different runs")
            reference = results[0]
            steps: dict[int, dict[str, Any]] = {}
            for result in results:
                _same_score_profile(reference, result)
                if (
                    result["identity"]["training_config_digest"]
                    != (reference["identity"]["training_config_digest"])
                ):
                    raise ValueError(f"{label}: one run has inconsistent training recipes")
                step = result["checkpoint"]["step"]
                if step in steps and steps[step]["result_id"] != result["result_id"]:
                    raise ValueError(f"{label}: ambiguous evaluations at seed {seed}, step {step}")
                steps[step] = result
            candidates = []
            for step, result in sorted(steps.items()):
                amount = result["training"].get(axis)
                if type(amount) not in {int, float} or not math.isfinite(amount) or amount < 0:
                    raise ValueError(f"{label}: missing or invalid {axis} at step {step}")
                curves.append(
                    {
                        "variant": label,
                        "seed": seed,
                        "step": step,
                        "tokens_seen": result["training"]["tokens_seen"],
                        "cumulative_training_flops": result["training"].get(
                            "cumulative_training_flops"
                        ),
                        "cross_entropy_nats": result["metrics"]["cross_entropy_nats"],
                    }
                )
                if budget * (1 - tolerance) <= amount <= budget:
                    candidates.append(result)
            if not candidates:
                raise ValueError(
                    f"{label}, seed {seed}: missing evaluated checkpoint at or below "
                    f"budget {budget} within tolerance {tolerance}"
                )
            selected[label][seed] = max(
                candidates, key=lambda row: (row["training"][axis], row["checkpoint"]["step"])
            )
        variant_anchor = next(iter(selected[label].values()))
        for result in selected[label].values():
            if recipe_differences(variant_anchor["run"]["config"], result["run"]["config"]):
                raise ValueError(f"{label}: seeds within one variant have different recipes")
    baseline = study["baseline"]
    if baseline not in selected:
        raise ValueError("baseline must name one of the variants")
    baseline_seeds = set(selected[baseline])
    for label, runs in selected.items():
        if set(runs) != baseline_seeds:
            raise ValueError(
                f"{label}: seed set differs from baseline; "
                f"missing={sorted(baseline_seeds - set(runs))}, "
                f"extra={sorted(set(runs) - baseline_seeds)}"
            )
    anchor = next(iter(selected[baseline].values()))
    rows = []
    diffs = []
    for index, (label, runs) in enumerate(selected.items()):
        previous = list(selected)[index - 1] if index else None
        for seed, result in runs.items():
            reference = selected[baseline][seed]
            _same_score_profile(anchor, result)
            # Also validate all curve points, not just the selected endpoint.
            differences = recipe_differences(anchor["run"]["config"], result["run"]["config"])
            unexpected = [
                key
                for key in differences
                if not any(key == prefix or key.startswith(prefix + ".") for prefix in allowed)
            ]
            if unexpected:
                raise ValueError(f"{label}: undeclared training recipe differences: {unexpected}")
            if result["training"]["source"] != anchor["training"]["source"]:
                raise ValueError(f"{label}: training dataset revision/content differs")
            if parameter_tolerance is not None:
                ref_n = reference["training"]["non_embedding_parameters"]
                n = result["training"]["non_embedding_parameters"]
                if ref_n <= 0 or abs(n - ref_n) / ref_n > parameter_tolerance:
                    raise ValueError(f"{label}: non-embedding parameter count exceeds tolerance")
            if axis == "cumulative_training_flops":
                estimator = result["training"]["flop_estimator"]
                if estimator is None or estimator != anchor["training"]["flop_estimator"]:
                    raise ValueError(f"{label}: unknown or incompatible FLOP estimator")
            loss = result["metrics"]["cross_entropy_nats"]
            diffs.append({"variant": label, "seed": seed, "changes": differences})
            speed_comparable = (
                bool(result["run"]["provenance"])
                and bool(reference["run"]["provenance"])
                and _speed_profile(result) == _speed_profile(reference)
            )
            rows.append(
                {
                    "variant": label,
                    "seed": seed,
                    "run_id": result["run"]["id"],
                    "step": result["checkpoint"]["step"],
                    "result_id": result["result_id"],
                    "checkpoint": result["checkpoint"]["path"],
                    "tokens_seen": result["training"]["tokens_seen"],
                    "estimated_flops": result["training"]["cumulative_training_flops"],
                    "total_parameters": result["training"]["total_parameters"],
                    "non_embedding_parameters": result["training"]["non_embedding_parameters"],
                    "cross_entropy_nats": loss,
                    "perplexity": result["metrics"]["perplexity"],
                    "delta_baseline": loss - reference["metrics"]["cross_entropy_nats"],
                    "delta_previous": None
                    if previous is None
                    else (loss - selected[previous][seed]["metrics"]["cross_entropy_nats"]),
                    "training_tokens_per_second": result["training"]["tokens_per_second"],
                    "training_peak_bytes": result["training"]["gpu_peak_allocated_bytes"],
                    "speed_comparable": speed_comparable,
                    "budget_shortfall_fraction": (budget - result["training"][axis]) / budget,
                }
            )
    return {
        "schema_version": 1,
        "study": study,
        "study_hash": fingerprint(study),
        "inputs": inputs,
        "rows": rows,
        "curves": curves,
        "config_differences": diffs,
        "paired_summaries": paired_summaries(rows, list(selected), baseline),
        "curve_summaries": curve_summaries(curves, list(selected)),
        "notes": [
            "Negative loss deltas mean improvement.",
            "Paired 95% Student-t intervals use independent training seeds, not "
            "tokens/checkpoints.",
            "Fewer than three pairs or zero observed variation: exploratory; no interval.",
            "Curve bands are pointwise at shared measured budgets; no interpolation or "
            "multiple-comparison correction. Intervals describe seed variation on this "
            "fixed corpus.",
            "FLOPs are estimates; 6ND omits architecture-dependent work.",
            "Timing definitions are recorded per result; v2 excludes "
            "evaluation/checkpoint/logging work.",
        ],
    }
