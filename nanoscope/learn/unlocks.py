"""What a learner has unlocked: `learn/unlocks.json`, written atomically.

    {"schema": 1, "policy": "guided", "unlocks": {
        "block:Attention": {"how": "earned", "lesson": "foundations/04-attention",
                            "at": "...", "evidence": "learn/checks/<id>.json"}}}

Without the file the policy is `open`: nothing is locked, so level 0 `run(Bigram)`, scripts
and notebooks never meet gating. The first `nanoscope learn start` writes `guided`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths
from nanoscope.fsutil import write_json_atomic
from nanoscope.learn.progress import now
from nanoscope.schemas.upgrade import upgrade

FILE = "unlocks.json"
POLICIES = ("guided", "open")
HOWS = ("earned", "skipped", "open")


def unlocks_path(owner: str = "local") -> Path:
    return paths.learn_dir(owner) / FILE


def exists(owner: str = "local") -> bool:
    return unlocks_path(owner).exists()


def read(owner: str = "local") -> dict[str, Any]:
    """The unlocks document; with no file, an open policy and nothing recorded."""
    path = unlocks_path(owner)
    if not path.exists():
        return {"schema": 1, "nanoscope": __version__, "policy": "open", "unlocks": {}}
    return upgrade(json.loads(path.read_text(encoding="utf-8")), "unlocks", path)


def _write(doc: dict[str, Any], owner: str) -> dict[str, Any]:
    doc["nanoscope"] = __version__
    write_json_atomic(unlocks_path(owner), doc)
    return doc


def policy(owner: str = "local") -> str:
    return read(owner)["policy"]


def set_policy(new: str, owner: str = "local") -> dict[str, Any]:
    if new not in POLICIES:
        raise ValueError(f"policy must be one of {', '.join(POLICIES)}, got {new!r}")
    doc = read(owner)
    doc["policy"] = new
    return _write(doc, owner)


def grant(unlock_id: str, how: str, lesson: str | None = None, evidence: str | None = None,
          owner: str = "local", *, doc: dict[str, Any] | None = None,
          save: bool = True) -> dict[str, Any]:
    """Record an unlock. Something already earned is never downgraded to skipped or open."""
    if how not in HOWS:
        raise ValueError(f"how must be one of {', '.join(HOWS)}, got {how!r}")
    doc = doc if doc is not None else read(owner)
    existing = doc["unlocks"].get(unlock_id)
    if existing and HOWS.index(existing["how"]) < HOWS.index(how):
        return doc  # earned beats skipped beats open
    doc["unlocks"][unlock_id] = {"how": how, "lesson": lesson, "at": now(),
                                 "evidence": evidence}
    return _write(doc, owner) if save else doc


def earn(lesson: str, ids: list[str], evidence: str, owner: str = "local") -> dict[str, Any]:
    """Several unlocks from one passed lesson, in one atomic write."""
    doc = read(owner)
    for unlock_id in ids:
        grant(unlock_id, "earned", lesson, evidence, owner, doc=doc, save=False)
    return _write(doc, owner)


def is_unlocked(unlock_id: str, owner: str = "local") -> bool:
    doc = read(owner)
    return doc["policy"] == "open" or unlock_id in doc["unlocks"]
