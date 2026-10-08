"""A study's results read from its files: the report, the per-seed finals and the bundle.

Everything here works from the run folders and a StudySpec alone, never importing a model, so
the API can serve it and `Study.report()` can share it. A study that is still running gives a
partial answer from the runs that exist.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths, store
from nanoscope.compare import METRICS, Comparison, RunSet, compare, load_runs
from nanoscope.schemas.upgrade import read_json
from nanoscope.statistics import summarize
from nanoscope.studyspec import StudySpec


def study_dir(name: str) -> Path:
    return paths.runs_dir() / "studies" / name


def variant_dirs(folder: Path) -> list[Path]:
    """The variant folders of a study that hold at least one run."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and (
        any(p.glob("seed-*/config.json")) or (p / "config.json").exists()))


def finals(folder: Path, variants: list[str] | None = None) -> dict[str, Any]:
    """Every finished run's final metrics: {variant: {seed: {metric: value}}}."""
    out: dict[str, Any] = {}
    for v in variant_dirs(folder):
        if variants is not None and v.name not in variants:
            continue
        out[v.name] = {str(r.seed): {m: r.final(m) for m in METRICS if r.curve(m)}
                       for r in load_runs(v)}
    return out


def results(name: str, spec: StudySpec | None) -> tuple[Comparison, list[dict[str, Any]],
                                                          dict[str, Any] | None]:
    """The comparison of the study's variants against its baseline, its scored predictions and
    its study.json. Raises ValueError when fewer than two variants have runs."""
    folder = study_dir(name)
    variants = variant_dirs(folder)
    if len(variants) < 2:
        raise ValueError(f"study {name!r} needs finished runs of at least 2 variants "
                         f"(it has {len(variants)})")
    baseline = None
    if spec and spec.baseline:
        baseline = next((v for v in variants if v.name == spec.baseline), None)
    by_name = {v.name: RunSet(store.ref_of(v), load_runs(v), label=v.name) for v in variants}
    base = by_name.get(baseline.name) if baseline else None
    comparison = compare(*by_name.values(), baseline=base or next(iter(by_name.values())))
    predictions: list[dict[str, Any]] = []
    for variant, expected in (spec.predictions if spec else {}).items():
        match = next((v for v in variants if v.name == variant), None)
        if match is None:
            continue
        for metric, value in expected.items():
            actual = summarize([r.final(metric) for r in load_runs(match)])
            predictions.append({"variant": variant, "metric": metric, "predicted": value,
                                "actual": actual, "error": (actual["mean"] - value) / value})
    manifest_path = folder / "study.json"
    manifest = read_json(manifest_path, "study") if manifest_path.exists() else None
    return comparison, predictions, manifest


def results_doc(name: str, mode: str, comparison: Comparison, predictions: list[dict[str, Any]],
                manifest: dict[str, Any] | None) -> dict[str, Any]:
    """results.v1: what `results.json` holds."""
    return {"schema": 1, "nanoscope": __version__, "study": name, "mode": mode,
            "manifest": manifest, "rows": comparison.rows, "notes": comparison.notes,
            "predictions": predictions}


def render_report(name: str, mode: str, preset: str, seeds: list[int],
                  budget: tuple[str, float] | None, manifest: dict[str, Any] | None,
                  comparison: Comparison, predictions: list[dict[str, Any]]) -> str:
    """The Markdown report."""
    lines = [f"# Study: {name}", "", f"- mode: {mode}",
             f"- preset: {preset}, seeds: {', '.join(map(str, seeds))}"]
    if budget is not None:
        lines.append(f"- budget: {budget[0]}({budget[1]:.3g}) per run")
    if manifest:
        m = manifest
        lines.append(f"- code: commit {m['commit'][:10]}")
        lines.append(f"- preregistered: {m['study_file']} committed in "
                     f"{m['study_file_commit'][:10]} at {m['study_file_committed_at']}; "
                     f"first run started {m['started_at']}")
        if m.get("committed_via") == "nanoscope":
            lines.append("- preregistration committed via nanoscope")
    lines += ["", "## Results", "", "```", str(comparison), "```"]
    if predictions:
        lines += ["", "## Predictions", "",
                  "| variant | metric | predicted | actual (95% CI) | error |",
                  "|---|---|---|---|---|"]
        for p in predictions:
            a = p["actual"]
            ci = (f" [{a['ci95_low']:.3f}, {a['ci95_high']:.3f}]"
                  if a["ci95_low"] is not None else "")
            lines.append(f"| {p['variant']} | {p['metric']} | {p['predicted']:.3f} | "
                         f"{a['mean']:.3f}{ci} | {p['error']:+.1%} |")
    return "\n".join(lines) + "\n"


def write_bundle(path: str | Path, name: str, report_md: str, results_json: dict[str, Any],
                 spec_name: str, spec_text: str) -> Path:
    """A zip of everything needed to check or re-run the study: the report, the results, the
    spec, study.json and plan.json when they exist, and `finals.json` with every seed's final
    metrics."""
    folder = study_dir(name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "report.md": report_md,
        "results.json": json.dumps(results_json, indent=2, default=str),
        spec_name: spec_text,
        "finals.json": json.dumps({"schema": 1, "nanoscope": __version__, "study": name,
                                   "finals": finals(folder)}, indent=2),
    }
    for fname in ("study.json", "plan.json"):
        if (folder / fname).exists():
            files[fname] = (folder / fname).read_text(encoding="utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, text in files.items():
            zf.writestr(fname, text)
    return path
