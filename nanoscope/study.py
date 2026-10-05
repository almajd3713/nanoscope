"""Studies: several model variants, the same seeds and budget, one committed plan.

    # studies/m1_ablation.py
    from nanoscope import Study, Tokens
    from nanoscope.models import GPT2, Modern

    study = Study("m1-ablation", preset="tinystories-5min", seeds=3, budget=Tokens(4e6),
                  match="params", baseline="gpt2", mode="record")
    study.add("gpt2", GPT2)
    study.add("modern", Modern, ffn_hidden=384)
    study.add("modern-no-rope", Modern, ffn_hidden=384, rope=False)
    study.predict("modern", val_bpb=1.10)   # preregistered: committed before the first run

Then `nanoscope study studies/m1_ablation.py --devices cuda:0,cuda:1` trains every
(variant, seed) pair, one run per device at a time, skipping finished runs, and
`nanoscope report studies/m1_ablation.py` writes experiments/m1-ablation/.

mode="record" adds guarantees and nothing else: a clean git tree whose commit is stored
in every run, a study file committed before its first run, predictions frozen once runs
start, and results that are never overwritten.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nanoscope.compare import METRICS, Comparison, RunSet, compare, load_runs
from nanoscope.dataset import load_tokenizer
from nanoscope.hardware import check_gpu_fits, cpu_threads, probe_memory
from nanoscope.presets import Preset, get_preset
from nanoscope.progress import one_line, snapshot
from nanoscope.run import RUNS_DIR, run
from nanoscope.sizing import build_on_meta, count_params, flops_per_token
from nanoscope.statistics import summarize

REPORTS_DIR = Path("experiments")
PROGRESS_EVERY = 30  # seconds between progress lines while workers run


@dataclass(frozen=True)
class Tokens:
    """Train every variant on this many tokens."""

    n: float


@dataclass(frozen=True)
class FLOPs:
    """Train every variant for this much compute; cheaper models see more tokens."""

    n: float


@dataclass
class Variant:
    name: str
    model_cls: type[nn.Module]
    kwargs: dict[str, Any]


@dataclass
class Job:
    variant: Variant
    seed: int
    preset: Preset
    output_dir: Path


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _caller_file() -> Path | None:
    frame = inspect.stack()[2]
    path = Path(frame.filename)
    return path.resolve() if path.suffix == ".py" and path.exists() else None


class Study:
    def __init__(
        self,
        name: str,
        preset: str | Preset = "tinystories-5min",
        seeds: int | list[int] = 3,
        budget: Tokens | FLOPs | None = None,
        match: str | None = None,
        baseline: str | None = None,
        mode: str = "explore",
        tolerance: float = 0.02,
        push_to_hub: str | None = None,
        compile: bool | str = False,
        **preset_overrides: Any,
    ) -> None:
        if mode not in ("explore", "record"):
            raise ValueError("mode must be 'explore' or 'record'")
        if match not in (None, "params"):
            raise ValueError("match must be None or 'params'")
        self.name = name
        base = get_preset(preset) if isinstance(preset, str) else preset
        self.preset = base.override(**preset_overrides)
        self.seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
        self.budget = budget
        self.match = match
        self.baseline = baseline
        self.mode = mode
        self.tolerance = tolerance
        self.push_to_hub = push_to_hub
        self.compile = compile
        self.source = _caller_file()
        self.variants: dict[str, Variant] = {}
        self.predictions: dict[str, dict[str, float]] = {}
        if mode == "record" and len(self.seeds) < 3:
            warnings.warn(f"study {name!r}: fewer than 3 seeds gives no confidence interval",
                          stacklevel=2)

    def add(self, name: str, model_cls: type[nn.Module], **kwargs: Any) -> Variant:
        if name in self.variants:
            raise ValueError(f"variant {name!r} added twice")
        if "/" in name or name.startswith("."):
            raise ValueError(f"variant name {name!r} must be usable as a folder name")
        self.variants[name] = Variant(name, model_cls, kwargs)
        return self.variants[name]

    def predict(self, variant: str, **values: float) -> None:
        """Write down what you expect before running; the report checks it."""
        unknown = set(values) - set(METRICS)
        if unknown:
            raise ValueError(f"predict takes {', '.join(METRICS)}; got {', '.join(unknown)}")
        self.predictions[variant] = {k: float(v) for k, v in values.items()}

    @property
    def dir(self) -> Path:
        return RUNS_DIR / "studies" / self.name

    def sizes(self) -> dict[str, dict[str, int]]:
        """Parameters and FLOPs per token of every variant, without training anything."""
        vocab_size = load_tokenizer(self.preset).vocab_size
        out = {}
        for v in self.variants.values():
            params = inspect.signature(v.model_cls).parameters
            from_data = {"vocab_size": vocab_size, "context_length": self.preset.context_length}
            kwargs = {k: x for k, x in from_data.items() if k in params} | v.kwargs
            model = build_on_meta(v.model_cls, **kwargs)
            out[v.name] = {
                "non_embedding_params": count_params(model)[1],
                "flops_per_token": flops_per_token(model, self.preset.context_length),
            }
        return out

    def _check_match(self, sizes: dict[str, dict[str, int]]) -> None:
        if self.match != "params":
            return
        counts = {name: s["non_embedding_params"] for name, s in sizes.items()}
        if max(counts.values()) > min(counts.values()) * (1 + self.tolerance):
            table = "\n".join(f"  {name}: {n:,}" for name, n in counts.items())
            raise ValueError(
                f"match='params' needs non-embedding parameter counts within "
                f"{self.tolerance:.0%}:\n{table}\nnanoscope.sizing.match_params can pick a "
                "width that matches."
            )

    def _steps(self, flops_per_tok: int) -> int:
        tokens_per_step = self.preset.batch_size * self.preset.context_length
        if self.budget is None:
            return self.preset.max_steps
        if isinstance(self.budget, Tokens):
            return math.ceil(self.budget.n / tokens_per_step)
        return math.ceil(self.budget.n / (flops_per_tok * tokens_per_step))

    def _job_preset(self, steps: int) -> Preset:
        # Study runs are judged on validation loss, so sampling text mid-run is wasted time
        # (generation has no KV cache). Each run still writes one sample at its last step.
        return self.preset.override(max_steps=steps, sample_interval=steps)

    def jobs(self) -> list[Job]:
        """Every (variant, seed) run, seed by seed, so early results are already paired."""
        if not self.variants:
            raise ValueError(f"study {self.name!r} has no variants; call study.add(...)")
        sizes = self.sizes()
        self._check_match(sizes)
        presets = {
            name: self._job_preset(self._steps(s["flops_per_token"]))
            for name, s in sizes.items()
        }
        return [
            Job(v, seed, presets[v.name], self.dir / v.name / f"seed-{seed}")
            for seed in self.seeds for v in self.variants.values()
        ]

    def _provenance(self) -> dict[str, Any]:
        """Record mode: check the git state and freeze the preregistration."""
        if self.source is None:
            raise ValueError("record mode needs the study to be defined in a .py file")
        cwd = self.source.parent
        try:
            root = Path(_git(["rev-parse", "--show-toplevel"], cwd))
        except subprocess.CalledProcessError as exc:
            raise ValueError("record mode needs the study file inside a git repository") from exc
        if not _git(["ls-files", str(self.source)], cwd):
            raise ValueError(f"commit {self.source.name} first: record mode preregisters it")
        dirty = _git(["status", "--porcelain"], cwd)
        if dirty:
            listing = "\n".join(dirty.splitlines()[:10])
            raise ValueError("record mode needs a clean git tree, so every result maps to one "
                             f"commit. Uncommitted changes:\n{listing}")
        file_commit, file_time = _git(
            ["log", "-1", "--format=%H %cI", "--", str(self.source)], cwd
        ).split()
        provenance = {
            "commit": _git(["rev-parse", "HEAD"], cwd),
            "study_file": str(self.source.relative_to(root)),
            "study_file_commit": file_commit,
            "study_file_committed_at": file_time,
        }
        manifest_path = self.dir / "study.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["predictions"] != self.predictions:
                raise ValueError(
                    f"the predictions in {self.source.name} changed after the study started "
                    f"on {manifest['started_at']}; preregistered predictions are frozen."
                )
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({
                **provenance,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "predictions": self.predictions,
            }, indent=2), encoding="utf-8")
        return provenance

    def _check_memory(self, devices: list[str]) -> None:
        """Refuse to put more workers on a GPU than its free memory holds."""
        workers = {d: devices.count(d) for d in set(devices) if d.startswith("cuda")}
        if not any(n > 1 for n in workers.values()):
            return
        costliest = max(self.variants.values(), key=lambda v: self._flops_per_token(v))
        vocab_size = load_tokenizer(self.preset).vocab_size
        params = inspect.signature(costliest.model_cls).parameters
        from_data = {"vocab_size": vocab_size, "context_length": self.preset.context_length}
        kwargs = {k: x for k, x in from_data.items() if k in params} | costliest.kwargs
        for device, n in workers.items():
            per_worker = probe_memory(lambda: costliest.model_cls(**kwargs), self.preset,
                                      torch.device(device))
            check_gpu_fits(device, n, per_worker)

    def _flops_per_token(self, variant: Variant) -> int:
        return self.sizes()[variant.name]["flops_per_token"]

    def run(
        self, devices: list[str] | None = None, shard: tuple[int, int] | None = None,
        workers_per_device: int = 1, threads: int | None = None,
    ) -> None:
        """Train every job not yet finished. Several workers run jobs side by side.

        workers_per_device > 1 puts that many runs on each device at once, which keeps a GPU
        busy when one small model can't (the CPU is the limit, not the GPU). threads caps the
        CPU threads each worker uses; the default splits the cores between CPU workers.
        """
        if workers_per_device > 1:
            devices = [d for d in (devices or ["cuda:0" if torch.cuda.is_available() else "cpu"])
                       for _ in range(workers_per_device)]
        if threads:
            torch.set_num_threads(threads)
        if shard is None and devices:
            self._check_memory(devices)
        provenance = self._provenance() if self.mode == "record" else {}
        jobs = self.jobs()
        if shard is None:
            self._write_plan(jobs)
        if devices and len(devices) > 1 and shard is None:
            self._run_parallel(devices, threads)
            return
        if shard is not None:
            index, count = shard
            jobs = jobs[index::count]
        device = devices[0] if devices else None
        for i, job in enumerate(jobs, 1):
            label = f"run {i}/{len(jobs)}: {job.variant.name} seed {job.seed}"
            print(f"[nanoscope] {label}", flush=True)
            result = run(
                job.variant.model_cls, job.preset, seed=job.seed, device=device,
                output_dir=job.output_dir, push_to_hub=self.push_to_hub, compile=self.compile,
                study={"name": self.name, "variant": job.variant.name, "mode": self.mode,
                       **provenance},
                **job.variant.kwargs,
            )
            bpb = result.summary()["final_val_bpb"]
            print(f"[nanoscope] {label} -> {bpb:.3f} bits per byte", flush=True)

    def _write_plan(self, jobs: list[Job]) -> None:
        """List every planned run, so `nanoscope status` can show the ones not started yet."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "plan.json").write_text(json.dumps({
            "study": self.name,
            "runs": [{"dir": str(j.output_dir.relative_to(self.dir)),
                      "max_steps": j.preset.max_steps} for j in jobs],
        }, indent=2), encoding="utf-8")

    def _run_parallel(self, devices: list[str], threads: int | None) -> None:
        """One worker process per device, each taking every n-th job."""
        if self.source is None:
            raise ValueError("running on several devices needs the study in a .py file")
        logs = self.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        workers = []
        on_cpu = sum(d == "cpu" for d in devices)
        for i, device in enumerate(devices):
            log = logs / f"worker-{i}.log"
            cmd = [sys.executable, "-m", "nanoscope.cli", "study", str(self.source),
                   "--name", self.name, "--devices", device, "--shard", f"{i}/{len(devices)}"]
            if self.push_to_hub:
                cmd += ["--push-to-hub", self.push_to_hub]
            if self.compile:
                cmd += ["--compile", str(self.compile).lower()]
            worker_threads = threads or (cpu_threads(on_cpu) if device == "cpu" else None)
            if worker_threads:
                cmd += ["--threads", str(worker_threads)]
            print(f"[nanoscope] worker {i} on {device}, log: {log}", flush=True)
            with log.open("w") as fh:
                workers.append((log, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)))
        total = len(self.jobs())
        last = None
        while any(proc.poll() is None for _, proc in workers):
            line = one_line(snapshot(self.dir), total)
            if line != last:
                print(f"[nanoscope] {line}", flush=True)
                last = line
            time.sleep(PROGRESS_EVERY)
        print(f"[nanoscope] {one_line(snapshot(self.dir), total)}", flush=True)
        failed = [log for log, proc in workers if proc.returncode != 0]
        if failed:
            tail = "\n".join(failed[0].read_text(encoding="utf-8").splitlines()[-15:])
            raise RuntimeError(f"{len(failed)} worker(s) failed; end of {failed[0]}:\n{tail}")

    def report(self, write: bool = True) -> StudyReport:
        sets = []
        for v in self.variants:
            if (self.dir / v).exists():
                sets.append(RunSet(str(self.dir / v), load_runs(self.dir / v), label=v))
        if len(sets) < 2:
            raise ValueError(f"study {self.name!r} needs finished runs of at least 2 variants")
        by_name = {s.label: s for s in sets}
        base = by_name.get(self.baseline or "", sets[0])
        comparison = compare(*sets, baseline=base)

        predictions = []
        for variant, expected in self.predictions.items():
            if variant not in by_name:
                continue
            for metric, value in expected.items():
                actual = summarize([r.final(metric) for r in by_name[variant].runs])
                predictions.append({
                    "variant": variant, "metric": metric, "predicted": value, "actual": actual,
                    "error": (actual["mean"] - value) / value,
                })
        manifest_path = self.dir / "study.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else None)
        report = StudyReport(self, comparison, predictions, manifest)
        if write:
            report.write(REPORTS_DIR / self.name)
        return report


@dataclass
class StudyReport:
    study: Study
    comparison: Comparison
    predictions: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None

    def __str__(self) -> str:
        s = self.study
        lines = [f"# Study: {s.name}", ""]
        lines.append(f"- mode: {s.mode}")
        lines.append(f"- preset: {s.preset.name}, seeds: {', '.join(map(str, s.seeds))}")
        if s.budget is not None:
            lines.append(f"- budget: {type(s.budget).__name__}({s.budget.n:.3g}) per run")
        if self.manifest:
            m = self.manifest
            lines.append(f"- code: commit {m['commit'][:10]}")
            lines.append(f"- preregistered: {m['study_file']} committed in "
                         f"{m['study_file_commit'][:10]} at {m['study_file_committed_at']}; "
                         f"first run started {m['started_at']}")
        lines += ["", "## Results", "", "```", str(self.comparison), "```"]
        if self.predictions:
            lines += ["", "## Predictions", "",
                      "| variant | metric | predicted | actual (95% CI) | error |",
                      "|---|---|---|---|---|"]
            for p in self.predictions:
                a = p["actual"]
                ci = (f" [{a['ci95_low']:.3f}, {a['ci95_high']:.3f}]"
                      if a["ci95_low"] is not None else "")
                lines.append(f"| {p['variant']} | {p['metric']} | {p['predicted']:.3f} | "
                             f"{a['mean']:.3f}{ci} | {p['error']:+.1%} |")
        return "\n".join(lines) + "\n"

    __repr__ = __str__

    def write(self, out: Path) -> Path:
        import matplotlib.pyplot as plt

        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(str(self), encoding="utf-8")
        (out / "results.json").write_text(json.dumps({
            "study": self.study.name,
            "mode": self.study.mode,
            "manifest": self.manifest,
            "rows": self.comparison.rows,
            "notes": self.comparison.notes,
            "predictions": self.predictions,
        }, indent=2, default=str), encoding="utf-8")
        fig = self.comparison.plot(save=out / "curves.png")
        plt.close(fig)
        return out


def load_study(path: str | Path, name: str | None = None) -> Study:
    """Import a study file and return its Study (by name when it defines several)."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"_nanoscope_study_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Don't write __pycache__ next to the study: record mode needs the tree untouched.
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
    studies = [v for v in vars(module).values() if isinstance(v, Study)]
    if name is not None:
        studies = [s for s in studies if s.name == name]
    if len(studies) != 1:
        found = ", ".join(s.name for s in studies) or "none"
        raise ValueError(f"{path} must define exactly one Study{f' named {name!r}' if name else ''}"
                         f"; found: {found}")
    return studies[0]
