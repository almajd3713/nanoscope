from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoscope import paths
from nanoscope.compare import IDENTITY_KEYS
from nanoscope.dataset import Data, load_data, tokenizer_id
from nanoscope.integrations import HubSync, chain, wandb_hook
from nanoscope.log import info
from nanoscope.modelref import class_source, model_ref, source_sha256
from nanoscope.presets import Preset, get_preset
from nanoscope.progress import ProgressBar
from nanoscope.runref import REQUIRED, preset_dir, run_name
from nanoscope.schemas.upgrade import read_json
from nanoscope.sizing import count_params, flops_per_token
from nanoscope.specs import validate_run_request
from nanoscope.status import STOP_FILE, StatusFile
from nanoscope.store import ref_of
from nanoscope.train_loop import OptimizerFactory, TrainResult, generate, train


@dataclass
class RunResult:
    preset: Preset
    model_name: str
    seed: int
    train_result: TrainResult
    device: str
    model: nn.Module
    data: Data

    @property
    def run_dir(self) -> Path:
        return self.train_result.run_dir

    @property
    def ref(self) -> str:
        """The run's name under the runs folder (see nanoscope.store)."""
        return ref_of(self.run_dir)

    @property
    def metrics(self) -> list[dict[str, Any]]:
        return self.train_result.metrics

    def to_dict(self, metrics: bool = False) -> dict[str, Any]:
        """The run as plain data: its ref, summary and (optionally) every metrics row."""
        out = {**self.summary(), "seed": self.seed, "device": self.device,
               "final_step": self.final_step, "stopped_early": self.train_result.stopped_early}
        return {**out, "metrics": self.metrics} if metrics else out

    @property
    def final_step(self) -> int:
        return self.train_result.final_step

    @property
    def train_losses(self) -> list[float]:
        return [r["loss"] for r in self.metrics]

    @property
    def val_losses(self) -> list[tuple[int, float]]:
        return [(r["step"], r["val_loss"]) for r in self.metrics if "val_loss" in r]

    @property
    def val_bpb(self) -> list[tuple[int, float]]:
        return [(r["step"], r["val_bpb"]) for r in self.metrics if "val_bpb" in r]

    @property
    def samples(self) -> list[tuple[int, str]]:
        return [(r["step"], r["sample"]) for r in self.metrics if "sample" in r]

    def plot(self, save: Path | str | None = None) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("plotting requires matplotlib: pip install matplotlib") from exc

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        steps = [r["step"] for r in self.metrics]
        losses = [r["loss"] for r in self.metrics]
        ax1.plot(steps, losses, alpha=0.3, label="train loss", color="C0")

        window = max(1, len(losses) // 20)
        if len(losses) > window:
            smoothed = []
            for i in range(len(losses)):
                start = max(0, i - window + 1)
                smoothed.append(sum(losses[start:i+1]) / (i - start + 1))
            ax1.plot(steps, smoothed, label="train (smoothed)", color="C0")

        val = self.val_losses
        if val:
            vs, vl = zip(*val, strict=True)
            ax1.plot(vs, vl, "o-", label="val loss", color="C1", markersize=4)

        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title(f"{self.model_name} on {self.preset.name} (seed={self.seed})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        return fig

    def print_samples(self, n: int = 3) -> None:
        for step, text in self.samples[-n:]:
            print(f"--- step {step} ---")
            print(text)
            print()

    def generate(
        self, prompt: str = "", max_new_tokens: int = 200, temperature: float = 0.8, seed: int = 42,
    ) -> str:
        return generate(
            self.model, self.data.tokenizer, prompt, max_new_tokens, temperature,
            self.preset.context_length, seed,
        )

    def summary(self) -> dict[str, Any]:
        val = self.val_losses
        bpb = self.val_bpb
        return {
            "model": self.model_name,
            "preset": self.preset.name,
            "seed": self.seed,
            "final_step": self.final_step,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": val[-1][1] if val else None,
            "final_val_bpb": bpb[-1][1] if bpb else None,
            "ref": self.ref,
            "run_dir": str(self.run_dir),
        }


@dataclass
class RunGroup:
    """The same configuration trained with several seeds."""

    results: list[RunResult]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, i: int) -> RunResult:
        return self.results[i]

    @property
    def seeds(self) -> list[int]:
        return [r.seed for r in self.results]

    def to_dict(self, metrics: bool = False) -> dict[str, Any]:
        """The set as plain data: the summary over seeds, plus each seed's own dict."""
        return {**self.summary(), "runs": [r.to_dict(metrics) for r in self.results]}

    @property
    def ref(self) -> str:
        """The set's name under the runs folder: the folder that holds the seeds."""
        return ref_of(self.results[0].run_dir.parent)

    def summary(self) -> dict[str, Any]:
        from nanoscope.statistics import summarize

        first = self.results[0]
        return {
            "model": first.model_name,
            "preset": first.preset.name,
            "seeds": self.seeds,
            "val_loss": summarize([r.summary()["final_val_loss"] for r in self.results]),
            "val_bpb": summarize([r.summary()["final_val_bpb"] for r in self.results]),
            "ref": self.ref,
            "run_dir": str(first.run_dir.parent),
        }

    def plot(self, save: Path | str | None = None) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, r in enumerate(self.results):
            ax.plot(*zip(*r.val_losses, strict=True), "o-", markersize=3, color=f"C{i}",
                    label=f"seed {r.seed}")
        first = self.results[0]
        ax.set_xlabel("step")
        ax.set_ylabel("val loss")
        ax.set_title(f"{first.model_name} on {first.preset.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        return fig


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Model arguments nanoscope fills in from the tokenizer and preset.
_FROM_DATA = ("vocab_size", "context_length")


def _split_kwargs(
    model_cls: type[nn.Module], preset: Preset, kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route each keyword to the model constructor or to the preset."""
    # a locked block is refused at build time with its own error, not as a TypeError here
    problems = [p for p in validate_run_request(model_cls, preset, kwargs) if p.code != "locked"]
    if problems:
        raise TypeError(problems[0].message)  # the first one; the API reports them all
    params = {
        name for name, p in inspect.signature(model_cls).parameters.items()
        if name not in _FROM_DATA and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k in params}
    overrides = {k: v for k, v in kwargs.items() if k not in params}
    return model_kwargs, overrides


def _preset_dir(preset: Preset) -> str:
    return preset_dir(preset)


def _run_name(
    model_cls: type[nn.Module], model_kwargs: dict[str, Any], given: Preset, preset: Preset,
    optimizer: str | None = None,
) -> str:
    """`bigram` for defaults; `bigram-1a2b3c4d` once anything differs (see nanoscope.runref)."""
    defaults = {name: REQUIRED if p.default is inspect.Parameter.empty else p.default
                for name, p in inspect.signature(model_cls).parameters.items()}
    return run_name(model_cls.__name__, defaults, model_kwargs, given, preset, optimizer)


class ConfigMismatch(ValueError):
    """The run folder holds a run with a different config, so resuming would mix two runs."""


def _refuse_record_overwrite(run_dir: Path, resume: bool) -> None:
    """Record-mode results are evidence: never start them over. Checked before anything is
    written to the folder, status included."""
    path = run_dir / "config.json"
    if not resume and path.exists():
        saved = read_json(path, "config")
        if saved.get("study", {}).get("mode") == "record":
            raise ValueError(f"{run_dir} is a record-mode result; record results are never "
                             "overwritten. Delete the folder by hand if you really mean it.")


def _check_config(run_dir: Path, config: dict[str, Any], resume: bool,
                  identity: dict[str, Any] | None = None) -> None:
    path = run_dir / "config.json"
    if resume and path.exists():
        saved = read_json(path, "config")
        for key in ("stats", "schema", "nanoscope"):  # facts about the file, not the run
            saved.pop(key, None)
        saved_model = dict(saved.get("model", {}))
        before = saved_model.get("source_sha256")
        for key in IDENTITY_KEYS:  # where the model came from, not what it is
            saved_model.pop(key, None)
        saved["model"] = saved_model
        after = (identity or {}).get("source_sha256")
        if before and after and before != after:
            _log(f"note: the source of {config['model']['class']} changed since this run "
                 f"started ({before[:8]} -> {after[:8]}); resuming with the current code")
        if saved != config:
            changed = sorted(k for k in config if saved.get(k) != config[k])
            raise ConfigMismatch(
                f"{run_dir} holds a run with a different config ({', '.join(changed)} changed). "
                "Pass resume=False to start it over, or use another output_dir."
            )


def _log(msg: str) -> None:
    info(msg)


def run(
    model_cls: type[nn.Module],
    preset: str | Preset = "tinystories-5min",
    *,
    seed: int = 0,
    seeds: int | list[int] | None = None,
    device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
    resume: bool = True,
    on_step: Any = None,
    on_eval: Any = None,
    wandb: bool | str = False,
    push_to_hub: str | None = None,
    progress: bool = True,
    compile: bool | str = False,
    block_stats: bool = False,
    checkpoint_steps: list[int] | None = None,
    optimizer: OptimizerFactory | None = None,
    study: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> RunResult | RunGroup:
    """Train model_cls on a preset; returns a RunResult, or a RunGroup when seeds= is given.

    Keywords that match model_cls's __init__ go to the model; the rest override preset fields.
    progress=False hides the progress bar. wandb=True (or a project name) logs live curves;
    push_to_hub="user/repo" mirrors the run (checkpoints included) in a private Hub repo and
    resumes from it when the local folder is empty, e.g. in a new Kaggle session.
    `study` is set by nanoscope.Study and recorded in config.json.
    block_stats=True records, at each eval step, every block's activation size, gradient norm,
    update-to-weight ratio and attention entropy in <run>/blockstats.jsonl
    (`nanoscope status --blocks <ref>` shows the latest).
    compile=True speeds up training with torch.compile (see train_loop.train); it doesn't
    change the run's identity, so a run can resume with it on or off.
    checkpoint_steps=[100, 500] keeps a full checkpoint at those steps in checkpoints/archive/
    (never pruned); it isn't part of the run's identity either.
    optimizer=make is a factory `(param_groups, preset) -> torch.optim.Optimizer` that replaces
    the default AdamW (see nanoscope.optim.muon for an example). Its ref goes into config.json
    and the run's name, so it is part of the run's identity; the learning-rate schedule still
    comes from the preset.
    """
    if seeds is not None:
        if output_dir is not None:
            raise ValueError("output_dir names one run; leave it out when passing seeds=")
        seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
        return RunGroup([
            cast(RunResult, run(
                model_cls, preset, seed=s, device=device, resume=resume, on_step=on_step,
                on_eval=on_eval, wandb=wandb, push_to_hub=push_to_hub, progress=progress,
                study=study, compile=compile, block_stats=block_stats,
                checkpoint_steps=checkpoint_steps, optimizer=optimizer,
                **model_kwargs))
            for s in seed_list
        ])
    if isinstance(preset, str):
        preset = get_preset(preset)
    given = preset
    model_kwargs, overrides = _split_kwargs(model_cls, preset, model_kwargs)
    preset = preset.override(**overrides)

    resolved_device = _resolve_device(device)
    optimizer_ref = model_ref(optimizer)[0] if optimizer else None
    name = _run_name(model_cls, model_kwargs, given, preset, optimizer_ref)
    if output_dir is None:
        run_dir = paths.runs_dir() / _preset_dir(given) / name / f"seed-{seed}"
    else:
        run_dir = Path(output_dir)

    late = [s for s in checkpoint_steps or [] if not 1 <= s <= preset.max_steps]
    if late:
        raise ValueError(f"checkpoint_steps {late} fall outside 1..{preset.max_steps} "
                         "(the number of training steps)")
    _refuse_record_overwrite(run_dir, resume)
    status = StatusFile(run_dir, preset.max_steps, device=str(resolved_device))
    status.write("preparing")
    try:
        data = load_data(preset)
        vocab_size = data.tokenizer.vocab_size
        from_data = {"vocab_size": vocab_size, "context_length": preset.context_length}
        model_params = inspect.signature(model_cls).parameters
        model_kwargs = {k: v for k, v in from_data.items() if k in model_params} | model_kwargs
        torch.manual_seed(seed)  # the seed decides the initial weights too, not just the batches
        model = model_cls(**model_kwargs)
        full_kwargs = {
            name: p.default for name, p in inspect.signature(model_cls).parameters.items()
            if p.default is not inspect.Parameter.empty
        } | model_kwargs

        config = json.loads(json.dumps({
            "model": {"class": model_cls.__name__, "kwargs": full_kwargs},
            "preset": asdict(preset),
            "tokenizer": tokenizer_id(preset),
            "seed": seed,
            **({"optimizer": {"ref": optimizer_ref}} if optimizer_ref else {}),
            **({"study": study} if study else {}),
        }, default=str))
        hub = None
        if push_to_hub:
            try:
                path_in_repo = run_dir.relative_to(paths.runs_dir()).as_posix()
            except ValueError:
                path_in_repo = "/".join(run_dir.parts[-3:])
            hub = HubSync(push_to_hub, run_dir, path_in_repo)
            if resume:
                hub.pull()
        ref, rebuildable = model_ref(model_cls)
        identity = {"ref": ref, "rebuildable": rebuildable,
                    "source_sha256": source_sha256(model_cls)}
        _check_config(run_dir, config, resume, identity)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not rebuildable and (source := class_source(model_cls)):
            (run_dir / "model_source.py").write_text(source, encoding="utf-8")
        n_params, n_non_embedding = count_params(model)
        stats = {
            "n_params": n_params,
            "n_non_embedding_params": n_non_embedding,
            "flops_per_token": flops_per_token(model, preset.context_length),
            "vocab_size": vocab_size,
            "device": (torch.cuda.get_device_name(resolved_device)
                       if resolved_device.type == "cuda" else resolved_device.type),
            "torch": torch.__version__,
        }
        from nanoscope import __version__

        (run_dir / "config.json").write_text(
            json.dumps({"schema": 1, "nanoscope": __version__,
                        **config, "model": {**config["model"], **identity}, "stats": stats},
                       indent=2),
            encoding="utf-8",
        )

        _log(f"{model_cls.__name__} ({stats['n_params']:,} params) on {preset.name}, "
             f"{resolved_device} -> {run_dir}")
        bar = None
        if progress:
            bar = ProgressBar(preset.max_steps, f"{model_cls.__name__} seed {seed}")
        on_step = chain(on_step, bar, status)
        finish = None
        if wandb:
            project = wandb if isinstance(wandb, str) else "nanoscope"
            wandb_step, finish = wandb_hook(project, "-".join(run_dir.parts[-3:]), config)
            on_step = chain(on_step, wandb_step)
        stop_file = run_dir / STOP_FILE
        stop_file.unlink(missing_ok=True)  # a leftover from an earlier cancel
        cancelled = False

        def should_stop() -> bool:
            nonlocal cancelled
            cancelled = cancelled or stop_file.exists()
            return cancelled

        if block_stats:
            from nanoscope.blockstats import BlockStats

            stats_hook = BlockStats(model, data, preset, run_dir, resolved_device)
            user_on_eval = on_eval

            def eval_hooks(step: int, val_loss: float) -> None:
                if user_on_eval:
                    user_on_eval(step, val_loss)
                stats_hook(step, val_loss)
            on_eval = eval_hooks
        result = train(
            model=model,
            data=data,
            preset=preset,
            run_dir=run_dir,
            device=resolved_device,
            seed=seed,
            resume=resume,
            on_step=on_step,
            on_eval=on_eval,
            on_checkpoint=hub,
            compile=compile,
            should_stop=should_stop,
            checkpoint_steps=checkpoint_steps,
            make_optimizer=optimizer,
        )
        if bar:
            bar.close()
        if finish:
            finish()
        if result.stopped_early:
            status.write("cancelled" if cancelled else "stopped", step=result.final_step)
        else:
            status.write("done", step=result.final_step)

        return RunResult(
            preset=preset,
            model_name=model_cls.__name__,
            seed=seed,
            train_result=result,
            device=str(resolved_device),
            model=model,
            data=data,
        )
    except BaseException as exc:
        if isinstance(exc, ConfigMismatch):  # refused before touching the run: leave it as it was
            status.restore()
        else:
            status.fail(exc)
        raise
