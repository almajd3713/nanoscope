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
from typing import Any

from nanoscope.learn import loader, unlocks
from nanoscope.learn.loader import CurriculumError


@dataclass(frozen=True)
class Locked:
    id: str  # "block:Attention" or "feature:gqa"
    lesson: str  # the one lesson that unlocks it
    message: str


class LockedBlockError(ImportError):
    """Using a block or feature the learner has not unlocked yet. An ImportError, so `from
    nanoscope.blocks import Attention` fails the way a missing name does. `locked` is the
    first locked id; `all` lists every one when a model uses several."""

    def __init__(self, locked: Locked | list[Locked]) -> None:
        everything = [locked] if isinstance(locked, Locked) else list(locked)
        message = everything[0].message
        if len(everything) > 1:
            names = ", ".join(x.id for x in everything)
            message = f"this model uses {len(everything)} locked parts: {names}\n" + "\n".join(
                x.message for x in everything)
        super().__init__(message)
        self.locked, self.all = everything[0], everything
        self.name = everything[0].id.split(":", 1)[-1]


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


@dataclass(frozen=True)
class LockedUse:
    """A locked block or feature a file uses, and where."""

    id: str
    line: int
    lesson: str
    message: str


def scan(path: str | Path, owner: str = "local") -> list[LockedUse]:
    """Locked uses in a file, found by reading it (ast only, nothing is imported or run):
    imports of locked names from `nanoscope.blocks`, locked block calls in a composition, and
    locked features (GQA, QK-norm, sliding window, z-loss). The editor shows these on save;
    `validate_run_request` refuses a run that has them."""
    import ast

    if unlocks.policy(owner) == "open":
        return []
    table = lock_table()
    earned = unlocks.read(owner)["unlocks"]
    found: dict[tuple[str, int], LockedUse] = {}

    def note(unlock_id: str, line: int) -> None:
        if unlock_id in table and unlock_id not in earned:
            found.setdefault((unlock_id, line), LockedUse(
                unlock_id, line, table[unlock_id], message_for(unlock_id, table[unlock_id])))

    source = Path(path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.module == "nanoscope.blocks":
            for alias in node.names:
                note(f"block:{alias.name}", node.lineno)

    from nanoscope.blocks.graph import parse

    def walk(node: dict[str, Any]) -> None:
        if node["kind"] == "list":
            for item in node["items"]:
                walk(item)
        elif node["kind"] == "block":
            line = node["span"]["line"]
            if not node.get("local"):
                note(f"block:{node['block']}", line)
            args = node["args"]
            if node["block"] == "Attention":
                heads, kv = args.get("n_heads"), args.get("n_kv_heads")
                if heads and kv and heads["kind"] == kv["kind"] == "literal" \
                        and kv["value"] < heads["value"]:
                    note("feature:gqa", kv["span"]["line"])
                if args.get("qk_norm", {}).get("value") is True:
                    note("feature:qk_norm", args["qk_norm"]["span"]["line"])
                window = args.get("window")
                if window and not (window["kind"] == "literal" and window["value"] is None):
                    note("feature:sliding_window", window["span"]["line"])
            for child in args.values():
                walk(child)

    for cls in parse(path)["classes"]:
        if not cls["representable"] or cls["kind"] != "decoder":
            continue
        for name, node in cls["args"].items():
            walk(node)
            if name == "z_loss" and node["kind"] == "literal" and node["value"]:
                note("feature:z_loss", node["span"]["line"])
    return sorted(found.values(), key=lambda u: (u.line, u.id))
