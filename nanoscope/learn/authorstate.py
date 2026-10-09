"""What `learn author-check` saved for a lesson folder, readable without importing torch (the
API reads it; only a worker runs the check itself, see nanoscope.learn.authoring)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nanoscope import paths

FILES = ("lesson.toml", "lesson.md", "starter.py", "solution.py", "notebook.py")


def folder_id(folder: Path) -> str:
    return f"{folder.parent.name}/{folder.name}"


def file_hashes(folder: Path) -> dict[str, str | None]:
    """sha256 of each lesson file (None when absent): a page compares these to say whether the
    files changed since a check."""
    return {name: (hashlib.sha256((folder / name).read_bytes()).hexdigest()
                   if (folder / name).is_file() else None) for name in FILES}


def result_file(folder: Path) -> Path:
    return paths.learn_dir("local") / "authoring" / f"{folder.parent.name}-{folder.name}.json"


def short_where(where: str | None, prefix: str) -> str:
    """A problem's location without the lesson's own folder prefix (the page names the folder)."""
    where = where or ""
    return where[len(prefix):] if where.startswith(prefix) else where


def last_check(folder: str | Path) -> dict[str, Any] | None:
    """The saved result for a folder, with `fresh`: whether its files are as they were."""
    directory = Path(folder).resolve()
    file = result_file(directory)
    if not file.exists():
        return None
    from nanoscope.schemas.upgrade import read_json

    doc = read_json(file, "author-check")
    return {**doc, "fresh": doc.get("files") == file_hashes(directory)}


