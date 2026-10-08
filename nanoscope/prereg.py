"""Preregistration commits: commit a study's spec (which holds its predictions) before its
first record run, so the claim "this was written down first" has a git timestamp behind it.

`preview(study)` says exactly what a commit would contain without changing anything;
`commit(study, preview_hash)` makes that commit, and only that one: it recomputes the preview
and refuses if it no longer matches what the user was shown. A commit's evidence is local
(anyone can rewrite their own history); pushing to a remote you do not control is what makes
it stronger.

Git runs here, in whichever process calls these functions. The API never does: both are jobs
for a worker (`prereg-preview`, `commit`), so repository hooks run in the worker.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanoscope.study import Study

GIT = ["git", "-c", "core.fsmonitor=false"]
DIFF = ["--no-ext-diff", "--no-textconv"]


TRAILER = "Committed-via: nanoscope"


class PreregError(ValueError):
    """A preregistration commit that cannot or should not be made; the text says why."""


@dataclass
class Preview:
    root: str  # the repository's top folder
    files: list[str]  # repo-relative paths the commit would contain
    diff: str  # unified diff of those files against HEAD (all new lines for an untracked file)
    message: str
    unrelated: list[str]  # other changed or untracked files; a commit is refused while any exist
    identity: bool  # git has a user.name and user.email to commit as
    nothing_to_commit: bool  # the files are already committed unchanged
    head: str | None
    hash: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(args: list[str], cwd: Path, ok: tuple[int, ...] = (0,)) -> str:
    out = subprocess.run([*GIT, *args], cwd=cwd, capture_output=True, text=True, check=False)
    if out.returncode not in ok:
        raise PreregError(f"git {' '.join(args[:2])} failed: {out.stderr.strip() or out.stdout}")
    return out.stdout


def _config(key: str, cwd: Path) -> str:
    out = subprocess.run([*GIT, "config", "--get", key], cwd=cwd, capture_output=True,
                         text=True, check=False)
    return out.stdout.strip()


def _message(study: Study) -> str:
    spec_seeds = ", ".join(map(str, study.seeds))
    lines = [f"Preregister study {study.name}", "",
             f"mode: {study.mode}; preset: {study.preset.name}; seeds: {spec_seeds}",
             f"variants: {', '.join(study.variants)}"]
    if study.predictions:
        lines.append("predictions: " + "; ".join(
            f"{v} " + ", ".join(f"{m}={x:g}" for m, x in p.items())
            for v, p in study.predictions.items()))
    lines += ["", "Committed by nanoscope before the first record run.", "", TRAILER]
    return "\n".join(lines)


def preview(study: Study) -> Preview:
    """What committing this study's spec would do. Changes nothing."""
    if study.source is None:
        raise PreregError("this study was not loaded from a file, so there is nothing to commit")
    cwd = study.source.parent
    try:
        root = Path(_git(["rev-parse", "--show-toplevel"], cwd).strip())
    except PreregError as exc:
        raise PreregError("the study file is not inside a git repository") from exc
    rel = str(study.source.resolve().relative_to(root.resolve()))
    files = [rel]
    head_out = subprocess.run([*GIT, "rev-parse", "--verify", "-q", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    head = head_out.stdout.strip() or None

    tracked = bool(_git(["ls-files", "--", rel], root).strip())
    if tracked and head:
        diff = _git(["diff", *DIFF, "HEAD", "--", rel], root)
    else:
        diff = _git(["diff", *DIFF, "--no-index", "--", "/dev/null", rel], root, ok=(0, 1))
    status = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], root)
    changed = [entry[3:] for entry in status.split("\0") if len(entry) > 3]
    # a rename lists the old path as its own entry without a status code; keep real paths only
    unrelated = sorted(p for p in changed if p not in files)
    result = Preview(
        root=str(root), files=files, diff=diff, message=_message(study), unrelated=unrelated,
        identity=bool(_config("user.name", root) and _config("user.email", root)),
        nothing_to_commit=tracked and not diff, head=head)
    blob = "\0".join([result.root, *files, result.diff, result.message, *unrelated,
                      str(result.head), str(result.identity)])
    result.hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return result


def commit(study: Study, preview_hash: str) -> str:
    """Make the preregistration commit the user was shown and return its hash.

    Refuses, committing nothing, when the preview changed since `preview_hash` was taken, when
    other files have uncommitted changes (a record run needs a clean tree, and the commit must
    hold only the spec), when git has no identity, or when there is nothing to commit."""
    now = preview(study)
    if now.hash != preview_hash:
        raise PreregError("the preview changed since you looked at it (the file, the repository "
                          "or the message differs); preview again and confirm the new diff")
    if now.unrelated:
        listing = "\n".join(f"  {p}" for p in now.unrelated[:10])
        raise PreregError("other files have uncommitted changes; commit or stash them first, so "
                          f"the preregistration commit holds only the spec:\n{listing}")
    if not now.identity:
        raise PreregError("git has no identity to commit as; run git config user.name and "
                          "git config user.email")
    if now.nothing_to_commit:
        raise PreregError(f"{now.files[0]} is already committed and unchanged; nothing to commit")
    root = Path(now.root)
    _git(["add", "--", *now.files], root)
    _git(["commit", "-m", now.message, "--", *now.files], root)
    return _git(["rev-parse", "HEAD"], root).strip()
