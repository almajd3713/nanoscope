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
import os
import subprocess
import sys
import time
import warnings
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoscope import __version__, paths, queue, store
from nanoscope.compare import METRICS, Comparison, RunSet, compare, load_runs
from nanoscope.dataset import load_tokenizer
from nanoscope.jobs.payload import preset_fields
from nanoscope.log import info
from nanoscope.modelref import model_ref
from nanoscope.presets import Preset, get_preset
from nanoscope.progress import one_line, snapshot
from nanoscope.run import RunResult, run
from nanoscope.schemas.upgrade import read_json
from nanoscope.sizing import build_on_meta, count_params, flops_per_token
from nanoscope.statistics import summarize
from nanoscope.status import STOP_FILE
from nanoscope.studyspec import StudySpec

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

    def to_spec(self) -> StudySpec:
        """This study as data (see nanoscope.studyspec); what a TOML study file holds."""
        from nanoscope.studyspec import StudySpec, VariantSpec, check_tomlable

        def plain(value: Any) -> Any:
            return list(value) if isinstance(value, tuple) else value

        spec = StudySpec(self.name, seeds=list(self.seeds), match=self.match,
                         baseline=self.baseline, mode=self.mode, tolerance=self.tolerance)
        try:
            registered = get_preset(self.preset.name)
        except KeyError:
            registered = None
        if registered is None:
            spec.custom_preset = {k: plain(v) for k, v in asdict(self.preset).items()
                                  if v is not None}
        else:
            spec.preset = registered.name
            spec.overrides = {f.name: plain(getattr(self.preset, f.name))
                              for f in fields(self.preset)
                              if getattr(self.preset, f.name) != getattr(registered, f.name)
                              and getattr(self.preset, f.name) is not None}
        if isinstance(self.budget, Tokens):
            spec.budget = {"tokens": self.budget.n}
        elif isinstance(self.budget, FLOPs):
            spec.budget = {"flops": self.budget.n}
        for v in self.variants.values():
            ref, rebuildable = model_ref(v.model_cls)
            if not rebuildable:
                raise ValueError(f"variant {v.name!r}: {v.model_cls.__name__} was defined in a "
                                 "notebook or script, so a spec can't name it")
            check_tomlable(v.name, v.kwargs)
            spec.variants.append(
                VariantSpec(v.name, ref, {k: plain(x) for k, x in v.kwargs.items()}))
        spec.predictions = {k: dict(v) for k, v in self.predictions.items()}
        return spec

    @classmethod
    def from_spec(cls, spec: StudySpec, source: Path | None = None) -> Study:
        """Build a Study from its spec, resolving model refs and declarative size matching."""
        from nanoscope.studyspec import resolve_variants

        if spec.custom_preset is not None:
            data = {f.name: None for f in fields(Preset)
                    if f.default is MISSING and f.default_factory is MISSING}
            preset = Preset.from_dict({**data, **spec.custom_preset})
        else:
            preset = get_preset(spec.preset).override(**{
                k: tuple(v) if k == "betas" else v for k, v in spec.overrides.items()})
        budget = None
        if spec.budget:
            (kind, n), = spec.budget.items()
            budget = Tokens(n) if kind == "tokens" else FLOPs(n)
        study = cls(spec.name, preset=preset, seeds=list(spec.seeds), budget=budget,
                    match=spec.match, baseline=spec.baseline, mode=spec.mode,
                    tolerance=spec.tolerance)
        study.source = source
        for name, model_cls, kwargs in resolve_variants(spec, preset):
            study.add(name, model_cls, **kwargs)
        for variant, values in spec.predictions.items():
            study.predict(variant, **values)
        return study

    @property
    def dir(self) -> Path:
        return paths.runs_dir() / "studies" / self.name

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
            raise ValueError("record mode needs the study to be defined in a .py or .toml file")
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
            manifest = read_json(manifest_path, "study")
            if manifest["predictions"] != self.predictions:
                raise ValueError(
                    f"the predictions in {self.source.name} changed after the study started "
                    f"on {manifest['started_at']}; preregistered predictions are frozen."
                )
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({
                "schema": 1, "nanoscope": __version__,
                **provenance,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "predictions": self.predictions,
            }, indent=2), encoding="utf-8")
        return provenance

    def _refuse_locked(self) -> None:
        """Variants a learner wrote can't use blocks they haven't unlocked (shipped models
        are never refused)."""
        from nanoscope.learn.gating import refuse

        for variant in self.variants.values():
            refuse(variant.model_cls)

    def run(
        self, devices: list[str] | None = None, workers_per_device: int = 1,
        threads: int | None = None,
    ) -> None:
        """Train every job not yet finished.

        With one device (or none) the runs go one after another in this process. With several
        devices, or workers_per_device > 1, they go through the job queue: local workers
        (one process per device, workers_per_device runs at once on each) take them in turn,
        which keeps a GPU busy when one small model can't. threads caps the CPU threads each
        run uses; the default splits the cores between the runs sharing a CPU.
        """
        self._refuse_locked()
        if (devices and len(devices) > 1) or workers_per_device > 1:
            self._run_queued(devices or ["cuda:0" if torch.cuda.is_available() else "cpu"],
                             workers_per_device, threads)
            return
        if threads:
            torch.set_num_threads(threads)
        provenance = self._provenance() if self.mode == "record" else {}
        jobs = self.jobs()
        self._write_plan(jobs)
        device = devices[0] if devices else None
        for i, job in enumerate(jobs, 1):
            if (self.dir / STOP_FILE).exists():
                info(f"study stopped: skipping {len(jobs) - i + 1} remaining run(s)")
                return
            label = f"run {i}/{len(jobs)}: {job.variant.name} seed {job.seed}"
            info(f"{label}")
            result = cast(RunResult, run(
                job.variant.model_cls, job.preset, seed=job.seed, device=device,
                output_dir=job.output_dir, push_to_hub=self.push_to_hub, compile=self.compile,
                study={"name": self.name, "variant": job.variant.name, "mode": self.mode,
                       **provenance},
                **job.variant.kwargs,
            ))
            if result.train_result.stopped_early:
                state = read_json(result.run_dir / "status.json", "status")["state"]
                if state == "cancelled" and not (self.dir / STOP_FILE).exists():
                    # `nanoscope stop <run>`: that one run is cancelled, the study goes on.
                    info(f"{label} cancelled at step {result.final_step}; "
                          "continuing with the next run")
                    continue
                info(f"{label} stopped at step {result.final_step}; "
                      "skipping the rest")
                return
            bpb = result.summary()["final_val_bpb"]
            info(f"{label} -> {bpb:.3f} bits per byte")

    def _write_plan(self, jobs: list[Job]) -> None:
        """List every planned run, so `nanoscope status` can show the ones not started yet."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "plan.json").write_text(json.dumps({
            "schema": 1, "nanoscope": __version__,
            "study": self.name,
            "runs": [{"dir": str(j.output_dir.relative_to(self.dir)),
                      "max_steps": j.preset.max_steps} for j in jobs],
        }, indent=2), encoding="utf-8")

    def enqueue(self, lane: str = "batch") -> list[int]:
        """Put every unfinished run on the queue, seed by seed, and write plan.json.

        Returns the ids of the jobs that stand for this study's unfinished runs. A run that
        is already done gets no job; one already queued or running keeps its job."""
        self._refuse_locked()
        provenance = self._provenance() if self.mode == "record" else {}
        jobs = self.jobs()
        for job in jobs:
            _, rebuildable = model_ref(job.variant.model_cls)
            if not rebuildable:
                raise ValueError(
                    f"variant {job.variant.name!r}: {job.variant.model_cls.__name__} was "
                    "defined in a notebook or script, so a worker can't load it. Put the "
                    "model in a .py file and import it.")
        self._write_plan(jobs)
        ids = []
        for job in jobs:
            payload: dict[str, Any] = {
                "model": model_ref(job.variant.model_cls)[0], "seed": job.seed,
                "kwargs": job.variant.kwargs, "compile": self.compile,
                "study": {"name": self.name, "variant": job.variant.name, "mode": self.mode,
                          **provenance},
                **preset_fields(job.preset),
            }
            if self.push_to_hub:
                payload["push_to_hub"] = self.push_to_hub
            job_id = queue.enqueue("run", payload, lane=lane, ref=store.ref_of(job.output_dir))
            if job_id is not None:
                ids.append(job_id)
        return ids

    def _run_queued(self, devices: list[str], workers_per_device: int,
                    threads: int | None) -> None:
        """Enqueue, start one local worker process per device, and watch until all are done."""
        ids = self.enqueue()
        total = len(self.jobs())
        slots = {d: devices.count(d) * workers_per_device for d in dict.fromkeys(devices)}
        logs = self.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **({"NANOSCOPE_THREADS": str(threads)} if threads else {})}
        workers = []
        for i, (device, count) in enumerate(slots.items()):
            log = logs / f"worker-{i}.log"
            cmd = [sys.executable, "-m", "nanoscope.cli", "worker", "--device", device,
                   "--slots", str(count), "--exit-when-idle"]
            info(f"worker {i} on {device} ({count} at a time), log: {log}")
            with log.open("w") as fh:
                workers.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env))
        try:
            self._watch(ids, workers, total)
        except KeyboardInterrupt:
            info("study interrupted: stopping the workers (runs keep their checkpoints)")
            for proc in workers:
                proc.terminate()
            for proc in workers:
                proc.wait()
            queue.cancel_prefix(store.ref_of(self.dir))
            raise
        finally:
            for proc in workers:
                if proc.poll() is None:
                    proc.kill()
        failed = [queue.get(i) for i in ids if queue.get(i)["state"] == "failed"]
        if failed:
            first = failed[0]
            log = paths.job_logs_dir() / f"{first['id']}.log"
            tail = ("\n".join(log.read_text(encoding="utf-8").splitlines()[-15:])
                    if log.exists() else first["error"])
            raise RuntimeError(f"{len(failed)} job(s) failed ({first['error']}); "
                               f"end of {log}:\n{tail}")

    def _watch(self, ids: list[int], workers: list[subprocess.Popen[bytes]], total: int) -> None:
        """Print the progress line every PROGRESS_EVERY seconds until every job has ended."""
        last, last_print = None, float("-inf")
        while True:
            states = {queue.get(i)["state"] for i in ids}
            finished = states <= {"done", "failed", "cancelled"}
            if finished or time.time() - last_print >= PROGRESS_EVERY:
                line = one_line(snapshot(self.dir), total)
                if line != last or finished:
                    info(f"{line}")
                    last = line
                last_print = time.time()
            if finished:
                return
            if all(proc.poll() is not None for proc in workers):
                raise RuntimeError(
                    f"every worker exited but {len(ids)} job(s) are not finished; see the logs "
                    f"in {self.dir / 'logs'}")
            time.sleep(min(1.0, PROGRESS_EVERY))

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
        manifest = (read_json(manifest_path, "study")
                    if manifest_path.exists() else None)
        report = StudyReport(self, comparison, predictions, manifest)
        if write:
            report.write(paths.reports_dir() / self.name)
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

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data: exactly what `results.json` holds (results.v1)."""
        return {
            "schema": 1, "nanoscope": __version__,
            "study": self.study.name,
            "mode": self.study.mode,
            "manifest": self.manifest,
            "rows": self.comparison.rows,
            "notes": self.comparison.notes,
            "predictions": self.predictions,
        }

    def write(self, out: Path) -> Path:
        import matplotlib.pyplot as plt

        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(str(self), encoding="utf-8")
        (out / "results.json").write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        fig = self.comparison.plot(save=out / "curves.png")
        plt.close(fig)
        return out


def load_study(path: str | Path, name: str | None = None) -> Study:
    """Import a study file (.py, or a .toml spec) and return its Study."""
    path = Path(path).resolve()
    if path.suffix == ".toml":
        return Study.from_spec(StudySpec.load(path), source=path)
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
