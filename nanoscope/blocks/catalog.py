"""The palette catalog: every block a model can be composed from, as data.

`nanoscope blocks --json` prints it (`blocks.v1`); the palette, `describe` and lesson gating
read the same table. Shipped blocks come from the registry; with a workspace folder, blocks
registered there with `register_block` are listed too, found by reading the files.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import nanoscope.blocks.registry as registry
from nanoscope import __version__
from nanoscope.blocks.spec import _options


def _args(cls: type) -> list[dict[str, Any]]:
    """The options a block takes: name, type as written, whether required, and the default."""
    out = []
    slots = getattr(cls, "SLOTS", None)
    if slots is not None:
        return [{"name": s, "type": "BlockSpec", "required": True} for s in slots]
    options = _options(cls) if issubclass(cls, registry_base()) else {
        k: p for k, p in inspect.signature(cls.__init__).parameters.items()  # type: ignore[misc]
        if k != "self"}  # a Decoder takes d_model and context_length itself
    for name, param in options.items():
        entry: dict[str, Any] = {
            "name": name,
            "type": (param.annotation if isinstance(param.annotation, str)
                     else getattr(param.annotation, "__name__", None))
            if param.annotation is not inspect.Parameter.empty else None,
            "required": param.default is inspect.Parameter.empty,
        }
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default if isinstance(
                param.default, (int, float, str, bool, type(None))) else repr(param.default)
        out.append(entry)
    return out


def registry_base() -> type:
    from nanoscope.blocks.spec import BlockModule
    return BlockModule


def _shipped(info: registry.BlockInfo) -> dict[str, Any]:
    cls = getattr(importlib.import_module(info.module), info.name)
    return {
        "name": info.name, "family": info.family, "tier": info.tier, "module": info.module,
        "args": _args(cls), "doc": inspect.getdoc(cls) or "", "reference": info.reference,
        "features": list(info.features), "user": info.user,
        # a shipped block with a reference is checked against it by its test in the repo
        "certified": info.reference is not None and not info.user, "file": None, "line": None,
    }


def catalog(workspace: str | Path | None = None) -> dict[str, Any]:
    """Every shipped block, plus the `register_block` blocks found under `workspace`."""
    blocks_pkg = importlib.import_module("nanoscope.blocks")
    for name in dir(blocks_pkg):  # importing each name registers its block
        getattr(blocks_pkg, name)
    blocks = [_shipped(info) for info in registry.all_blocks() if not info.user]
    errors: list[dict[str, Any]] = []
    if workspace is not None:
        from nanoscope.blocks.discover import discover

        found = discover(workspace)
        errors = found["errors"]
        for b in found["blocks"]:
            blocks.append({
                "name": b["name"], "family": b["family"], "tier": "composite", "module": None,
                "args": [{"name": o["name"], "type": o["annotation"], "required": o["required"],
                          **({"default": o["default"]} if "default" in o else {})}
                         for o in b["options"]],
                "doc": b["doc"], "reference": b["reference"], "features": [], "user": True,
                "certified": False, "file": b["file"], "line": b["line"]})
    return {"schema": 1, "nanoscope": __version__, "blocks": blocks, "errors": errors}


def format_catalog(doc: dict[str, Any]) -> str:
    """The catalog as a table grouped by family."""
    rows = [["block", "family", "tier", "options", "reference", "certified"]]
    for b in sorted(doc["blocks"], key=lambda b: (b["family"], b["name"])):
        options = ", ".join(a["name"] + ("" if a["required"] else "?") for a in b["args"])
        rows.append([b["name"] + (" (yours)" if b["user"] else ""), b["family"], b["tier"],
                     options or "-", b["reference"] or "-", "yes" if b["certified"] else "no"])
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows]
    return "\n".join([*lines, "", "options marked ? are optional"])
