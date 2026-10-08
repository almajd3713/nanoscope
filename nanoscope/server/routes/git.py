from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope.server import workspace

router = APIRouter(prefix="/api/git", tags=["git"])

# Read-only commands only. fsmonitor can run a program named in the repository's config, so it is
# turned off; GIT_OPTIONAL_LOCKS stops `status` from writing the index. Hooks do not run for any
# of these. Anything that commits is a worker job (nanoscope.prereg), never this process.
GIT = ["git", "-c", "core.fsmonitor=false", "--no-optional-locks"]


class GitStatus(BaseModel):
    repo: bool  # the workspace is inside a git repository
    root: str | None = None
    branch: str | None = None  # None when detached
    head: str | None = None
    clean: bool = True  # no uncommitted change anywhere in the repository
    changed: list[str] = []  # repo-relative paths with uncommitted changes (first 50)
    identity: bool = False  # git has a user.name and user.email to commit as
    path: str | None = None  # the file asked about
    path_committed: bool | None = None  # that file is tracked and unchanged since HEAD


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run([*GIT, *args], cwd=cwd, capture_output=True, text=True, check=False,
                          env=env, timeout=20)


@router.get("/status")
def git_status(path: str | None = None) -> GitStatus:
    """The workspace repository's state, read-only: whether a record study can run (a clean tree,
    its spec committed) and whether a preregistration commit could be made (an identity).
    `path` is a workspace file (e.g. `studies/m1.toml`) whose committed state is asked about."""
    base = workspace.root()
    top = _git(["rev-parse", "--show-toplevel"], base)
    if top.returncode != 0:
        return GitStatus(repo=False, path=path)
    root = Path(top.stdout.strip())
    branch = _git(["symbolic-ref", "--short", "-q", "HEAD"], root).stdout.strip() or None
    head = _git(["rev-parse", "--verify", "-q", "HEAD"], root).stdout.strip() or None
    status = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], root).stdout
    changed = [e[3:] for e in status.split("\0") if len(e) > 3]
    name = _git(["config", "--get", "user.name"], root).stdout.strip()
    email = _git(["config", "--get", "user.email"], root).stdout.strip()
    out = GitStatus(repo=True, root=str(root), branch=branch, head=head, clean=not changed,
                    changed=changed[:50], identity=bool(name and email), path=path)
    if path is not None:
        target = workspace.safe_path(path, must_exist=False)
        rel = target.resolve().relative_to(root.resolve()).as_posix()
        tracked = bool(_git(["ls-files", "--", rel], root).stdout.strip())
        out.path_committed = tracked and rel not in changed
    return out
