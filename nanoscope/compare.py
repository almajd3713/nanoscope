"""Did it help? Compare runs on the same held-out text, with seed-level confidence intervals.

    compare(my_result, "gpt2")                  # your run against the shipped GPT-2 baseline
    compare(modern, no_rope, gpt2, baseline=gpt2)

Anything that names runs works: a RunResult, a RunGroup (from run(..., seeds=3)), a run
folder, or a run name looked up under runs/<preset>/ and then the baselines shipped with
nanoscope. Each seed is one measurement; with three or more seeds per side you get a 95%
confidence interval on the difference.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from nanoscope import __version__, paths, store
from nanoscope.schemas.upgrade import read_json
from nanoscope.statistics import paired_difference, summarize, unpaired_difference

BASELINES_DIR = paths.baselines_dir()
METRICS = {"val_bpb": "bits per byte", "val_loss": "loss (nats per token)"}


@dataclass
class SeedRun:
    run_dir: Path
    config: dict[str, Any]
    metrics: list[dict[str, Any]]

    @classmethod
    def load(cls, run_dir: Path) -> SeedRun:
        config = read_json(run_dir / "config.json", "config")
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        return cls(run_dir, config, [json.loads(line) for line in lines])

    @property
    def seed(self) -> int:
        return self.config["seed"]

    @property
    def preset(self) -> dict[str, Any]:
        return self.config["preset"]

    @property
    def tokens_per_step(self) -> int:
        return self.preset["batch_size"] * self.preset["context_length"]

    @property
    def final_step(self) -> int:
        return self.metrics[-1]["step"] if self.metrics else 0

    def curve(self, metric: str) -> list[tuple[int, float]]:
        return [(r["step"] * self.tokens_per_step, r[metric]) for r in self.metrics if metric in r]

    def final(self, metric: str) -> float:
        points = self.curve(metric)
        if not points:
            raise ValueError(f"{self.run_dir} has no {metric} measurements yet")
        return points[-1][1]


IDENTITY_KEYS = ("ref", "rebuildable", "source_sha256")  # where a model came from, not what it is


def _without_seed(config: dict[str, Any]) -> dict[str, Any]:
    plain = {k: v for k, v in config.items() if k not in ("seed", "stats")}
    if isinstance(plain.get("model"), dict):
        plain["model"] = {k: v for k, v in plain["model"].items() if k not in IDENTITY_KEYS}
    return plain


@dataclass
class RunSet:
    """All seeds of one configuration."""

    source: str
    runs: list[SeedRun]
    label: str = ""

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError(f"no runs found in {self.source}")
        first = _without_seed(self.runs[0].config)
        for r in self.runs[1:]:
            if _without_seed(r.config) != first:
                raise ValueError(f"{self.source}: seeds were trained with different configs")
        self.runs.sort(key=lambda r: r.seed)

    @property
    def config(self) -> dict[str, Any]:
        return self.runs[0].config

    @property
    def seeds(self) -> list[int]:
        return [r.seed for r in self.runs]

    @property
    def mode(self) -> str:
        return self.config.get("study", {}).get("mode", "explore")


def load_runs(path: str | Path) -> list[SeedRun]:
    """A seed folder (holding config.json), or a folder of seed-* folders."""
    path = Path(path)
    if (path / "config.json").exists():
        return [SeedRun.load(path)]
    seeds = sorted(p for p in path.glob("seed-*") if (p / "config.json").exists())
    if not seeds:
        raise FileNotFoundError(f"no runs in {path}")
    return [SeedRun.load(p) for p in seeds]


def _holds_runs(path: Path) -> bool:
    return (path / "config.json").exists() or any(path.glob("seed-*/config.json"))


def _resolve(item: Any, preset: str | None) -> RunSet:
    from nanoscope.run import RunGroup, RunResult

    if isinstance(item, RunResult):
        return RunSet(str(item.run_dir), [SeedRun.load(item.run_dir)])
    if isinstance(item, RunGroup):
        return RunSet(str(item.results[0].run_dir.parent),
                      [SeedRun.load(r.run_dir) for r in item.results])
    if isinstance(item, RunSet):
        return item
    try:  # a ref: a path under the runs folder (or baselines/...)
        where = store.resolve(item)
    except (ValueError, FileNotFoundError):
        where = None
    if where is not None and _holds_runs(where):
        return RunSet(store.ref_of(where), load_runs(where))
    path = Path(item)
    if path.exists():
        return RunSet(str(path), load_runs(path))
    if preset is None:
        raise ValueError(f"can't find runs named {item!r}: pass a run folder, or preset=...")
    for root in (paths.runs_dir(), BASELINES_DIR):
        if (root / preset / str(item)).exists():
            where = root / preset / str(item)
            return RunSet(str(where), load_runs(where))
    available = sorted({p.name for root in (paths.runs_dir(), BASELINES_DIR)
                        for p in (root / preset).glob("*") if p.is_dir()})
    raise FileNotFoundError(
        f"no runs named {item!r} for preset {preset!r}; available: {', '.join(available) or '-'}"
    )


def _label(target: RunSet, sets: list[RunSet]) -> str:
    """Class name, plus whichever settings tell this set apart from the others."""
    cls = target.config["model"]["class"]
    kwargs = {k: v for k, v in target.config["model"]["kwargs"].items()
              if k not in ("vocab_size", "context_length")}
    same_class = [s for s in sets if s.config["model"]["class"] == cls]
    parts = [f"{k}={v}" for k, v in kwargs.items()
             if any(s.config["model"]["kwargs"].get(k) != v for s in same_class)]
    preset = target.config["preset"]
    parts += [f"{k}={v}" for k, v in preset.items()
              if any(s.config["preset"].get(k) != v for s in sets)]
    return f"{cls}({', '.join(parts)})" if parts else cls


def _fmt(x: float | None, sign: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+.3f}".replace("-", "−") if sign else f"{x:.3f}"


def verdict_of(delta: dict[str, Any] | None) -> str:
    """What a row's difference from the baseline says (lower is better for both metrics):
    "better" or "worse" when the 95% interval excludes zero, "within noise" when it
    includes it, "no CI" with fewer than three seeds, "baseline" for the baseline itself."""
    if delta is None:
        return "baseline"
    if delta["ci95_low"] is None:
        return "no CI"
    if delta["ci95_high"] < 0:
        return "better"
    if delta["ci95_low"] > 0:
        return "worse"
    return "within noise"


@dataclass
class Comparison:
    metric: str
    baseline: str
    rows: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    sets: list[RunSet] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        preset = self.sets[0].config["preset"]
        header = (f"{METRICS[self.metric]} on the first {preset['eval_docs']} validation "
                  f"documents of {preset['dataset']} (lower is better)")
        table = [["model", "seeds", "non-emb params", "tokens", self.metric,
                  f"Δ vs {self.baseline}", ""]]
        for row in self.rows:
            s, d = row["summary"], row["delta"]
            value = _fmt(s["mean"])
            if s["ci95_low"] is not None:
                value += f" ± {(s['ci95_high'] - s['ci95_low']) / 2:.3f}"
            verdict = row["verdict"]
            if d is None:
                delta, verdict = "(baseline)", ""
            elif d["ci95_low"] is None:
                delta = _fmt(d["mean"], sign=True)
                verdict += ": need 3+ seeds each"
            else:
                delta = (f"{_fmt(d['mean'], True)} "
                         f"[{_fmt(d['ci95_low'], True)}, {_fmt(d['ci95_high'], True)}]")
                if not d["paired"]:
                    verdict += " (unpaired)"
            table.append([row["label"], str(s["n"]), f"{row['params'] / 1e6:.2f}M",
                          f"{row['tokens'] / 1e6:.1f}M", value, delta, verdict])
        widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
        lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
                 for r in table]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join([header, "", *lines, *[f"note: {n}" for n in self.notes]])

    __repr__ = __str__

    def to_dict(self) -> dict[str, Any]:
        """The comparison as plain data (the comparison.v1 schema)."""
        out: dict[str, Any] = {
            "schema": 1, "nanoscope": __version__, "metric": self.metric,
            "baseline": self.baseline, "rows": self.rows, "notes": self.notes}
        if self.sets:
            from nanoscope.statistics import precision_plan

            out["curves"] = [
                {"label": s.label, "source": s.source,
                 "seeds": [{"seed": r.seed, "points": [[x, y] for x, y in r.curve(self.metric)]}
                           for r in s.runs]}
                for s in self.sets]
            first = self.sets[0]
            out["precision_plan"] = precision_plan(
                self.metric, first.config["preset"]["name"], min(len(s.runs) for s in self.sets))
        return out

    def plot(self, save: str | Path | None = None) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (row, s) in enumerate(zip(self.rows, self.sets, strict=True)):
            color = f"C{i}"
            curves = [r.curve(self.metric) for r in s.runs]
            for c in curves:
                ax.plot(*zip(*c, strict=True), color=color, alpha=0.25, linewidth=1)
            common = sorted(set.intersection(*[{t for t, _ in c} for c in curves]))
            means = [mean(dict(c)[t] for c in curves) for t in common]
            ax.plot(common, means, color=color, linewidth=2,
                    label=f"{row['label']} (n={len(curves)})")
        ax.set_xlabel("training tokens")
        ax.set_ylabel(METRICS[self.metric])
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        return fig


def compare(
    *runs: Any, baseline: Any = None, preset: str | None = None, metric: str = "val_bpb",
) -> Comparison:
    """Compare runs against a baseline (default: the last one given).

    Names are looked up under runs/<preset>/ and then the shipped baselines; `preset` is the
    preset's folder name, taken from the first run passed as an object when left out.
    """
    if len(runs) < 2:
        raise ValueError("compare needs at least two sets of runs")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    if preset is None:  # look names up next to the first run given as an object
        for item in runs:
            if not isinstance(item, (str, Path)):
                preset = Path(_resolve(item, None).runs[0].run_dir).parent.parent.name
                break
    sets = [_resolve(item, preset) for item in runs]
    base = sets[-1] if baseline is None else _resolve(baseline, preset)
    if baseline is not None and all(s.source != base.source for s in sets):
        sets.append(base)
    base_index = next(i for i, s in enumerate(sets) if s.source == base.source)

    text = {(s.config["preset"]["dataset"], s.config["preset"]["eval_docs"]) for s in sets}
    if len(text) > 1:
        raise ValueError(f"these runs were evaluated on different text: {sorted(text)}")
    if metric == "val_loss" and len({s.config["tokenizer"] for s in sets}) > 1:
        raise ValueError("val_loss is per token, so it can't compare tokenizers; use val_bpb")

    notes = []
    tokens = {s.runs[0].final_step * s.runs[0].tokens_per_step for s in sets}
    if len(tokens) > 1:
        notes.append("runs trained on different numbers of tokens; this compares the "
                     "training budgets as much as the models")
    for s in sets:
        for r in s.runs:
            if r.final_step < r.preset["max_steps"]:
                notes.append(
                    f"{r.run_dir} stopped at step {r.final_step} of {r.preset['max_steps']}"
                )

    modes = {s.mode for s in sets}
    if len(modes) > 1:
        notes.append("mixes record and explore runs; only record runs belong in a claim")
    for s in sets:
        s.label = s.label or _label(s, sets)
        if len(modes) > 1 and s.mode == "explore":
            s.label += " [explore]"
    ref = {r.seed: r.final(metric) for r in sets[base_index].runs}
    rows = []
    for i, s in enumerate(sets):
        values = {r.seed: r.final(metric) for r in s.runs}
        delta = None
        if i != base_index:
            if sorted(values) == sorted(ref):
                delta = paired_difference([values[k] for k in sorted(values)],
                                          [ref[k] for k in sorted(ref)])
            else:
                delta = unpaired_difference(list(values.values()), list(ref.values()))
        stats = s.config["stats"]
        rows.append({
            "label": s.label,
            "source": s.source,
            "seeds": s.seeds,
            # studies match non-embedding parameters, so that is the number to show
            "params": stats.get("n_non_embedding_params", stats["n_params"]),
            "tokens": s.runs[0].final_step * s.runs[0].tokens_per_step,
            "summary": summarize(list(values.values())),
            "delta": delta,
            "verdict": verdict_of(delta),
        })
    return Comparison(metric, sets[base_index].label, rows, notes, sets)


def export_baseline(item: Any, dest: str | Path = BASELINES_DIR, every: int = 10) -> Path:
    """Copy a run set's config, samples and a thinned metrics log into the baselines folder."""
    s = _resolve(item, None)
    out = Path(dest) / Path(s.source).parent.name / Path(s.source).name
    for r in s.runs:
        seed_dir = out / r.run_dir.name
        seed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(r.run_dir / "config.json", seed_dir / "config.json")
        if (r.run_dir / "samples.txt").exists():
            shutil.copy(r.run_dir / "samples.txt", seed_dir / "samples.txt")
        kept = [{k: v for k, v in row.items() if k != "sample"} for row in r.metrics
                if row["step"] % every == 0 or "val_loss" in row]
        (seed_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8"
        )
    return out
