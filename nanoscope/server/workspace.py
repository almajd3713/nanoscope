"""The workspace folder as the API sees it: confined paths and ETags."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import HTTPException

from nanoscope import paths

IGNORED = {".git", "__pycache__", ".venv", "node_modules", ".ipynb_checkpoints"}


def root() -> Path:
    return paths.workspace_dir().resolve()


def safe_path(relative: str, *, must_exist: bool = True) -> Path:
    """`relative` inside the workspace, or an error. Refuses absolute paths, `..`, and anything
    that resolves (through symlinks) outside the folder."""
    text = relative.replace("\\", "/")
    if not text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise HTTPException(400, f"{relative!r} must be a path inside the workspace")
    if ".." in Path(text).parts:
        raise HTTPException(400, f"{relative!r} may not contain '..'")
    base = root()
    target = (base / text).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(400, f"{relative!r} points outside the workspace")
    if must_exist and not target.is_file():
        raise HTTPException(404, f"no file {relative!r} in the workspace")
    return target


def etag_of(path: Path) -> str:
    """Changes when the file's bytes or modification time do (a strong validator)."""
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f'"{stat.st_mtime_ns:x}-{digest}"'


def relative(path: Path) -> str:
    return path.relative_to(root()).as_posix()


def listing(pattern: str = "**/*") -> list[Path]:
    """Files matching a glob under the workspace, skipping caches and hidden tool folders."""
    base = root()
    if not base.exists():
        return []
    found = []
    for path in sorted(base.glob(pattern)):
        parts = path.relative_to(base).parts
        if not path.is_file() or any(p in IGNORED or p.startswith(".git") for p in parts):
            continue
        if path.resolve().is_relative_to(base):  # a symlink out of the folder is not listed
            found.append(path)
    return found


def write_atomic(target: Path, content: str) -> None:
    """Write so a reader (or the file watcher) never sees half a file: temp file, then replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)
