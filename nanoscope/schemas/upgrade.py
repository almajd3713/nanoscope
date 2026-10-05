"""Read the JSON files nanoscope writes, whatever version wrote them.

A file with no `schema` key is v0 (everything written before schemas existed). Readers accept
the current version N and N-1, upgrading older files in memory, and refuse anything newer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanoscope.schemas import CURRENT


def _stamp_v1(data: dict[str, Any]) -> dict[str, Any]:
    return {**data, "schema": 1}


# UPGRADES[kind][n] turns a version n document into version n + 1.
UPGRADES: dict[str, dict[int, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    kind: {0: _stamp_v1} for kind in CURRENT
}


def upgrade(data: dict[str, Any], kind: str, source: str | Path = "file") -> dict[str, Any]:
    """The document at the current schema version, or ValueError if it can't be read."""
    current = CURRENT[kind]
    version = data.get("schema", 0)
    if version > current:
        written_by = data.get("nanoscope", "an unknown version")
        raise ValueError(
            f"{source} was written by a newer nanoscope ({written_by}): it uses {kind} schema "
            f"{version} and this nanoscope reads up to {current}. Upgrade nanoscope."
        )
    if version < current - 1:
        raise ValueError(
            f"{source} uses {kind} schema {version}, too old for this nanoscope "
            f"(reads {current - 1} and {current}). Re-run it or re-export it."
        )
    while version < current:
        data = UPGRADES[kind][version](data)
        version += 1
    return data


def read_json(path: str | Path, kind: str) -> dict[str, Any]:
    """Load a JSON file of the given kind ('config', 'status', 'plan', 'study', 'results')."""
    path = Path(path)
    return upgrade(json.loads(path.read_text(encoding="utf-8")), kind, path)
