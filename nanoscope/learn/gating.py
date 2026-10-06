"""What is locked, and the one function that decides.

A block or feature is *lockable* exactly when some lesson lists it under `unlocks` in its
lesson.toml: the lock list comes from the curricula, not from a second list. Under policy
`guided` a lockable id is locked until it is earned (or skipped); with no `unlocks.json`
the policy is `open` and nothing is ever locked.

Gating limits what a learner *composes*, never what they *run*, and it is a learning aid, not
security: Python cannot stop determined code (see docs/learn.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from nanoscope.learn import loader, unlocks
from nanoscope.learn.loader import CurriculumError


@dataclass(frozen=True)
class Locked:
    id: str  # "block:Attention" or "feature:gqa"
    lesson: str  # the one lesson that unlocks it
    message: str


@cache
def _table(root: str) -> dict[str, str]:
    table: dict[str, str] = {}
    base = Path(root)
    for path_id in loader.list_path_ids(base):
        try:
            path = loader.load_path(path_id, base)
        except CurriculumError:
            continue  # broken content must never break an import
        for lesson in path.lessons:
            for unlock_id in lesson.unlocks:
                table.setdefault(unlock_id, lesson.id)
    return table


def lock_table() -> dict[str, str]:
    """Every lockable id and the lesson that unlocks it."""
    return dict(_table(str(loader.curricula_dir())))


def reload() -> None:
    """Forget the cached lock table (after curricula change on disk)."""
    _table.cache_clear()


def lockable(unlock_id: str) -> bool:
    return unlock_id in lock_table()


def message_for(unlock_id: str, lesson: str) -> str:
    return (f"{unlock_id} is locked until you build it yourself in the lesson {lesson}.\n"
            f"  start the lesson:      nanoscope learn start {lesson}\n"
            "  or unlock everything:  nanoscope learn unlock --all")


def check(ids: list[str], owner: str = "local") -> list[Locked]:
    """The ids among `ids` that are locked for this learner right now."""
    if unlocks.policy(owner) == "open":
        return []
    table = lock_table()
    earned = unlocks.read(owner)["unlocks"]
    return [Locked(i, table[i], message_for(i, table[i]))
            for i in ids if i in table and i not in earned]
