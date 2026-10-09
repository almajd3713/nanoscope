"""Run refs: the stable names under which runs are addressed.

A ref is a path relative to the runs root, which is already unique:
`tinystories-5min/bigram-1a2b3c4d/seed-0` is one run, `tinystories-5min/bigram-1a2b3c4d` is a
set of seeds, `studies/m1-ablation/no-rope/seed-2` is a study run. Refs starting with
`baselines/` point at the results shipped with the package (read-only).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from nanoscope import paths
from nanoscope.progress import RunState, snapshot

BASELINES_PREFIX = "baselines"


def _check(ref: str | Path) -> PurePosixPath:
    text = str(ref).replace("\\", "/")
    parts = PurePosixPath(text)
    if not text.strip("/"):
        raise ValueError("empty run ref")
    if parts.is_absolute() or (len(text) > 1 and text[1] == ":"):
        raise ValueError(f"run ref {text!r} must be relative to the runs folder, not absolute")
    if ".." in parts.parts:
        raise ValueError(f"run ref {text!r} may not contain '..'")
    return parts


def resolve(ref: str | Path, *, must_exist: bool = True) -> Path:
    """The folder a ref names. Refs can't escape the runs root (no '..', no symlink tricks)."""
    parts = _check(ref)
    if parts.parts[0] == BASELINES_PREFIX:
        root, rest = paths.baselines_dir(), parts.parts[1:]
    else:
        root, rest = paths.runs_dir(), parts.parts
    target = root.joinpath(*rest)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"run ref {str(ref)!r} points outside {root}")
    if must_exist and not target.exists():
        raise FileNotFoundError(f"no run at ref {str(ref)!r} (looked in {target})")
    return target


def ref_of(run_dir: str | Path) -> str:
    """The ref of a run folder; folders outside the known roots keep their own path."""
    path = Path(run_dir)
    for prefix, root in ((None, paths.runs_dir()), (BASELINES_PREFIX, paths.baselines_dir())):
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        parts = (prefix, *relative.parts) if prefix else relative.parts
        return "/".join(parts)
    return path.as_posix()


def list_runs(prefix: str = "", state: str | None = None) -> list[RunState]:
    """Every run under a ref prefix (all runs when empty), optionally only in one state."""
    root = resolve(prefix, must_exist=False) if prefix else paths.runs_dir()
    runs = snapshot(root)
    return [r for r in runs if state is None or r.state == state]


def list_sets(prefix: str = "") -> list[str]:
    """Refs of every set of seeds under a prefix (the folders that hold `seed-*` runs)."""
    sets = {r.run_dir.parent for r in list_runs(prefix) if r.run_dir.name.startswith("seed-")}
    return sorted(ref_of(p) for p in sets)


def locate(target: str | Path) -> Path:
    """The folder a CLI argument names: a ref, a study name, or a plain path."""
    for candidate in (str(target), f"studies/{target}"):
        try:
            return resolve(candidate)
        except (ValueError, FileNotFoundError):
            continue
    if Path(target).exists():
        return Path(target)
    raise FileNotFoundError(
        f"nothing to stop at {str(target)!r}: not a run ref, a study name or a folder "
        f"(runs are under {paths.runs_dir()})"
    )


def request_stop(target: str | Path) -> list[str]:
    """Ask every running run under target to stop, by writing a STOP file in its folder.

    A study (a folder with plan.json) also gets its own STOP file, so the jobs that have not
    started are skipped. Returns the refs it wrote to; the runs stop at their next step."""
    from nanoscope import queue
    from nanoscope.status import STOP_FILE

    root = locate(target)
    refs = []
    if paths.queue_db().exists():
        queue.cancel_prefix(ref_of(root))  # queued jobs never start; running ones get STOP
    if (root / "plan.json").exists():
        (root / STOP_FILE).write_text("", encoding="utf-8")
        refs.append(ref_of(root))
    for state in snapshot(root):
        if state.state in ("running", "preparing"):
            (state.run_dir / STOP_FILE).write_text("", encoding="utf-8")
            refs.append(ref_of(state.run_dir))
    return refs


class LoadedRun:
    """A trained run rebuilt from its folder: the model with its latest checkpoint loaded."""

    def __init__(self, ref: str, run_dir: Path, config: dict, model: Any, preset: Any,
                 tokenizer: Any, step: int) -> None:
        self.ref, self.run_dir, self.config = ref, run_dir, config
        self.model, self.preset, self.tokenizer, self.step = model, preset, tokenizer, step

    def generate(self, prompt: str = "", max_new_tokens: int = 200, temperature: float = 0.8,
                 seed: int = 42) -> str:
        from nanoscope.train_loop import generate

        return generate(self.model, self.tokenizer, prompt, max_new_tokens, temperature,
                        self.preset.context_length, seed)


def saved_steps(run_dir: Path) -> list[int]:
    """The steps a run has a checkpoint for: the kept ones and the archived ones."""
    folder = Path(run_dir) / "checkpoints"
    return sorted({int(p.stem.removeprefix("step_")) for p in folder.glob("**/step_*.pt")})


def no_checkpoint(ref: str, step: int, steps: list[int]) -> FileNotFoundError:
    return FileNotFoundError(
        f"{ref} has no checkpoint at step {step}; it has {', '.join(map(str, steps)) or 'none'}. "
        "Keep more with run(..., checkpoint_steps=[...])")


def _checkpoint_at(run_dir: Path, step: int, device: Any) -> dict[str, Any]:
    import torch

    name = f"step_{step:08d}.pt"
    for path in (run_dir / "checkpoints" / name, run_dir / "checkpoints" / "archive" / name):
        if path.exists():
            return torch.load(path, map_location=device, weights_only=False)
    raise no_checkpoint(ref_of(run_dir), step, saved_steps(run_dir))


def load_run(ref: str | Path, device: str = "cpu", step: int | None = None) -> LoadedRun:
    """Rebuild a trained run from its ref: import its model class, load the latest checkpoint
    (or the one at `step`, kept or archived)."""
    import torch

    from nanoscope.dataset import load_tokenizer
    from nanoscope.modelref import load_class
    from nanoscope.presets import Preset
    from nanoscope.schemas.upgrade import read_json
    from nanoscope.train_loop import _load_checkpoint

    run_dir = resolve(ref)
    if not (run_dir / "config.json").exists():
        seeds = sorted(p.name for p in run_dir.glob("seed-*") if (p / "config.json").exists())
        if not seeds:
            raise FileNotFoundError(f"{ref} holds no run (no config.json)")
        raise ValueError(f"{ref} is a set of {len(seeds)} seeds; load one, e.g. "
                         f"{str(ref).rstrip('/')}/{seeds[0]}")
    if str(ref).startswith(BASELINES_PREFIX):
        raise FileNotFoundError(
            f"{ref} is a shipped baseline: it keeps the curves, not the checkpoints. "
            "Load a run you trained.")
    config = read_json(run_dir / "config.json", "config")
    model_info = config["model"]
    if "ref" not in model_info:
        raise ValueError(
            f"{ref} was trained before runs recorded where their model came from; "
            f"build {model_info['class']}(**config['model']['kwargs']) yourself and load "
            f"the checkpoint from {run_dir / 'checkpoints'}")
    if not model_info.get("rebuildable", True):
        raise ValueError(
            f"{ref} was trained from a class defined in a notebook or script, so it can't be "
            f"imported again. Its source is saved in {run_dir / 'model_source.py'}: define the "
            f"class from that, then build it with config['model']['kwargs'].")
    state = (_load_checkpoint(run_dir, torch.device(device)) if step is None
             else _checkpoint_at(run_dir, step, torch.device(device)))
    if state is None:
        raise FileNotFoundError(f"{ref} has no checkpoint to load yet")
    cls = load_class(model_info["ref"])
    model = cls(**model_info["kwargs"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    preset = Preset.from_dict(config["preset"])
    return LoadedRun(str(ref), run_dir, config, model, preset, load_tokenizer(preset),
                     int(state["step"]))
