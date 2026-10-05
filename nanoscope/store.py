"""Run refs: the stable names under which runs are addressed.

A ref is a path relative to the runs root, which is already unique:
`tinystories-5min/bigram-1a2b3c4d/seed-0` is one run, `tinystories-5min/bigram-1a2b3c4d` is a
set of seeds, `studies/m1-ablation/no-rope/seed-2` is a study run. Refs starting with
`baselines/` point at the results shipped with the package (read-only).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

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
    from nanoscope.status import STOP_FILE

    root = locate(target)
    refs = []
    if (root / "plan.json").exists():
        (root / STOP_FILE).write_text("", encoding="utf-8")
        refs.append(ref_of(root))
    for state in snapshot(root):
        if state.state in ("running", "preparing"):
            (state.run_dir / STOP_FILE).write_text("", encoding="utf-8")
            refs.append(ref_of(state.run_dir))
    return refs
