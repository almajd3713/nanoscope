"""JSON Schemas (2020-12) for every file nanoscope writes that other tools read.

`get("config")` returns the current version of a schema; `get("config", 1)` a specific one.
Readers go through `nanoscope.schemas.upgrade.read_json`, which accepts the current version
and the one before it (N and N-1).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

CURRENT = {
    "config": 1, "status": 1, "plan": 1, "study": 1, "results": 1, "problem": 1, "studyspec": 1,
    "prepare": 1, "bench": 1, "comparison": 1, "job": 1, "worker": 1, "graph": 1,
    "describe": 1, "blockstats": 1, "blocks": 1, "path": 1, "lesson": 1, "progress": 1,
    "check": 1, "unlocks": 1, "cert": 1, "card": 1, "inspect": 1,
}


@cache
def _load(name: str, version: int) -> dict[str, Any]:
    path = Path(__file__).parent / f"{name}.v{version}.json"
    if not path.exists():
        raise KeyError(f"no schema {name}.v{version}")
    return json.loads(path.read_text(encoding="utf-8"))


def get(name: str, version: int | None = None) -> dict[str, Any]:
    if name not in CURRENT:
        raise KeyError(f"unknown schema {name!r}; known: {', '.join(sorted(CURRENT))}")
    return _load(name, version or CURRENT[name])
