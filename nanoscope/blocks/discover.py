"""What a workspace defines, found by reading it: blocks registered with `register_block` and
models that subclass `Decoder` or `Composite`. No file is imported or run, so a file that
raises on import, or one with a typo, cannot hurt the listing (plan 8.1).

    discover("workspace/")  # {"blocks": [...], "models": [...], "errors": [...]}
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from nanoscope.blocks.graph import parse

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "site-packages", ".tox"}
CONTEXT = ("self", "d_model", "context_length")


def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.py")
                  if not any(part in SKIP_DIRS or part.startswith(".")
                             for part in p.relative_to(root).parts[:-1]))


def _decorator(node: ast.expr) -> tuple[bool, dict[str, ast.expr]]:
    """(is it register_block, its keyword arguments) for one decorator expression."""
    call = node if isinstance(node, ast.Call) else None
    target = call.func if call else node
    name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
    return name == "register_block", {k.arg: k.value for k in (call.keywords if call else [])
                                       if k.arg}


def _options(init: ast.FunctionDef) -> list[dict[str, Any]]:
    args = init.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    pairs = list(zip(positional, defaults, strict=True))
    pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
    out = []
    for arg, default in pairs:
        if arg.arg in CONTEXT:
            continue
        entry: dict[str, Any] = {
            "name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
            "required": default is None}
        if default is not None:
            try:
                entry["default"] = ast.literal_eval(default)
            except (ValueError, TypeError):
                entry["default_source"] = ast.unparse(default)
        out.append(entry)
    return out


def _blocks_in(tree: ast.Module, file: Path) -> list[dict[str, Any]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            registered, kwargs = _decorator(deco)
            if not registered:
                continue
            init = next((s for s in node.body
                         if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None)
            family = kwargs.get("family")
            reference = kwargs.get("reference")
            found.append({
                "name": node.name, "file": str(file), "line": node.lineno,
                "family": family.value if isinstance(family, ast.Constant) else "custom",
                "reference": ast.unparse(reference) if reference is not None else None,
                "options": _options(init) if init else [],
                "doc": (ast.get_docstring(node) or "").strip(),
            })
    return found


def discover(root: str | Path) -> dict[str, Any]:
    """The user blocks and models under `root` (a folder or one file)."""
    blocks: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for file in _files(Path(root)):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            blocks += _blocks_in(tree, file)
            for cls in parse(file)["classes"]:
                models.append({"name": cls["name"], "kind": cls["kind"], "file": str(file),
                               "line": cls["line"], "representable": cls["representable"],
                               "reason": cls["reason"]})
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError) as exc:
            errors.append({"file": str(file), "error": str(exc)})
    return {"blocks": blocks, "models": models, "errors": errors}


# -- models ----------------------------------------------------------------------------------

def _module_bases(tree: ast.Module) -> tuple[dict[str, str], set[str]]:
    """(import table, names of classes that are models). A class is a model when a base is
    `torch.nn.Module`, a class from `nanoscope.models` or `nanoscope.blocks`, or another model
    class in the same file."""
    imports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0])
    models: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            dotted = ast.unparse(base)
            root, _, rest = dotted.partition(".")
            qualified = imports.get(root, root) + (f".{rest}" if rest else "")
            if (qualified.startswith("torch.nn.") and qualified.endswith("Module")) \
                    or qualified.startswith(("nanoscope.models", "nanoscope.blocks")) \
                    or qualified in models:
                models.add(node.name)
    return imports, models


def discover_models(root: str | Path) -> list[dict[str, Any]]:
    """Every class under `root` that looks like a model (a `torch.nn.Module` subclass, or built
    on `nanoscope.models` / `nanoscope.blocks`), with its constructor parameters, read from the
    source without importing it. Registered blocks (`@register_block`) are not models."""
    found: list[dict[str, Any]] = []
    for file in _files(Path(root)):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        _, models = _module_bases(tree)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in models:
                continue
            if any(_decorator(d)[0] for d in node.decorator_list):
                continue
            init = next((s for s in node.body
                         if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None)
            found.append({
                "name": node.name, "file": str(file), "line": node.lineno,
                "doc": (ast.get_docstring(node) or "").strip(),
                "params": _all_params(init) if init else [],
            })
    return found


def _all_params(init: ast.FunctionDef) -> list[dict[str, Any]]:
    """Every constructor parameter (the ones nanoscope fills from the data included)."""
    args = init.args
    positional = [*args.posonlyargs, *args.args][1:]  # drop self
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    pairs = list(zip(positional, defaults, strict=True))
    pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
    out = []
    for arg, default in pairs:
        entry: dict[str, Any] = {
            "name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
            "required": default is None, "default": None}
        if default is not None:
            try:
                entry["default"] = ast.literal_eval(default)
            except (ValueError, TypeError):
                entry["default"] = ast.unparse(default)
        out.append(entry)
    return out
