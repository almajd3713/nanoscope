from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from nanoscope.eval.artifacts import write_json


def write_report(comparison: dict[str, Any], output: Path, *, plots: bool = False) -> Path:
    pyplot = None
    if plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as pyplot
        except ImportError as exc:
            raise RuntimeError("plots require the eval extra: uv sync --extra eval") from exc
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", comparison)
    rows = comparison["rows"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {comparison['study']['name']}",
        "",
        f"Baseline: `{comparison['study']['baseline']}`. "
        f"Budget: {comparison['study']['budget']} {comparison['study']['axis']}.",
        "",
        "| Variant | Seed | Step | Tokens | Estimated FLOPs | CE (nats) | PPL | "
        "Delta baseline | Delta previous | Speed comparable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [
            row[key]
            for key in (
                "variant",
                "seed",
                "step",
                "tokens_seen",
                "estimated_flops",
                "cross_entropy_nats",
                "perplexity",
                "delta_baseline",
                "delta_previous",
                "speed_comparable",
            )
        ]
        formatted = [
            "unknown"
            if value is None
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value).replace("|", "\\|").replace("\n", " ")
            for value in values
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    summaries = comparison["paired_summaries"]
    if summaries:
        with (output / "paired-summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        lines.extend(
            [
                "",
                "## Paired seed differences",
                "",
                "Negative values favor the variant. Intervals measure seed variation on "
                "the fixed corpus.",
                "",
                "| Variant | Reference | Relation | Pairs | Mean delta | Sample SD | 95% "
                "CI | Status |",
                "|---|---|---|---:|---:|---:|---|---|",
            ]
        )
        for summary in summaries:
            interval = (
                "unavailable"
                if summary["ci95_low"] is None
                else f"[{summary['ci95_low']:.6g}, {summary['ci95_high']:.6g}]"
            )
            sd = "unknown" if summary["sample_sd"] is None else f"{summary['sample_sd']:.6g}"
            fields = [
                summary["variant"],
                summary["reference"],
                summary["relation"],
                str(summary["n"]),
                f"{summary['mean']:.6g}",
                sd,
                interval,
                summary["status"],
            ]
            lines.append(
                "| "
                + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in fields)
                + " |"
            )
    lines.extend(
        [
            "",
            *[f"- {note}" for note in comparison["notes"]],
            "",
            "Full parameter/cost measurements are in `comparison.csv`; resolved config "
            "differences and exact input hashes are in `comparison.json`.",
            "",
        ]
    )
    if pyplot is not None:
        for axis, label in (
            ("tokens_seen", "Training prediction tokens"),
            ("cumulative_training_flops", "Estimated training FLOPs"),
        ):
            figure, axes = pyplot.subplots(figsize=(8, 5))
            for variant in comparison["study"]["variants"]:
                name = variant["name"]
                seeds = sorted(
                    {point["seed"] for point in comparison["curves"] if point["variant"] == name}
                )
                for seed in seeds:
                    points = sorted(
                        (
                            point
                            for point in comparison["curves"]
                            if point["variant"] == name
                            and point["seed"] == seed
                            and point[axis] is not None
                        ),
                        key=lambda point: point[axis],
                    )
                    if points:
                        axes.plot(
                            [p[axis] for p in points],
                            [p["cross_entropy_nats"] for p in points],
                            marker="o",
                            label=f"{name}, seed {seed}",
                        )
                aggregate = sorted(
                    (
                        point
                        for point in comparison["curve_summaries"]
                        if point["variant"] == name and point["axis"] == axis
                    ),
                    key=lambda point: point["budget"],
                )
                if aggregate and aggregate[0]["n"] > 1:
                    (line,) = axes.plot(
                        [p["budget"] for p in aggregate],
                        [p["mean"] for p in aggregate],
                        linewidth=2.5,
                        label=f"{name}, seed mean",
                    )
                    # NaNs break bands across budgets without an estimable interval.
                    axes.fill_between(
                        [p["budget"] for p in aggregate],
                        [
                            p["ci95_low"] if p["ci95_low"] is not None else float("nan")
                            for p in aggregate
                        ],
                        [
                            p["ci95_high"] if p["ci95_high"] is not None else float("nan")
                            for p in aggregate
                        ],
                        color=line.get_color(),
                        alpha=0.18,
                    )
            axes.set(xlabel=label, ylabel="Held-out cross-entropy (nats/token)")
            if axes.lines:
                axes.legend()
            axes.grid(alpha=0.2)
            figure.tight_layout()
            stem = f"loss-vs-{axis}"
            figure.savefig(output / f"{stem}.png", dpi=150)
            figure.savefig(output / f"{stem}.svg")
            pyplot.close(figure)
            lines.extend([f"![Loss against {label}]({stem}.png)", ""])
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
