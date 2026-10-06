"""The architecture graph of a model file, read without running it.

    parse("my_model.py")   # {"schema": 1, "classes": [{"name": "MyLM", "kind": "decoder", ...}]}

Each `Decoder` or `Composite` subclass in the file becomes a class entry. A `Decoder`'s
`super().__init__(...)` arguments become a tree of nodes:

    literal   a Python literal (numbers, strings, None, lists of them)
    param     a name from the class's own `__init__` signature
    block     a call to a registered block, or to a `Composite` defined in the file
    list      a list or tuple that holds blocks (the `pattern=` of a Decoder)
    opaque    a call to anything else; shown, but its inside is not editable
    expr      any other expression, kept as source text

A class with a statement outside this subset is code-only: it still appears, with the reason
and the line. Every node carries its source span (1-based lines, 0-based columns), which
`emit` uses to patch the file and `describe` uses to point at a line.

Only `ast` is used, so parsing never imports or runs the file (plan 8.1).
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

from nanoscope import __version__
from nanoscope.blocks import registry

BLOCKS_MODULE = "nanoscope.blocks"
BASES = {"Decoder": "decoder", "Composite": "composite"}


def _known_blocks() -> dict[str, registry.BlockInfo]:
    blocks = importlib.import_module(BLOCKS_MODULE)
    for name in dir(blocks):  # importing each name registers its block
        getattr(blocks, name)
    return {info.name: info for info in registry.all_blocks()}


def _decoder_params() -> list[str]:
    from nanoscope.blocks.structure import Decoder
    return [p for p in inspect.signature(Decoder.__init__).parameters if p != "self"]


def _dotted(node: ast.expr) -> str | None:
    """`a.b.C` for a name or attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _imports(tree: ast.Module) -> dict[str, str]:
    """Local name -> the dotted name it stands for, from the module's top-level imports."""
    table: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                table[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                table[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0])
    return table


def _span(node: ast.AST) -> dict[str, int]:
    return {"line": node.lineno, "col": node.col_offset,  # type: ignore[attr-defined]
            "end_line": node.end_lineno or node.lineno,  # type: ignore[attr-defined]
            "end_col": node.end_col_offset or 0}  # type: ignore[attr-defined]


class _CodeOnly(Exception):
    def __init__(self, reason: str, line: int) -> None:
        super().__init__(reason)
        self.reason, self.line = reason, line


class _Parser:
    def __init__(self, source: str, tree: ast.Module) -> None:
        self.source = source
        self.tree = tree
        self.imports = _imports(tree)
        self.blocks = _known_blocks()
        self.kinds: dict[str, str] = {}  # local class name -> "decoder" | "composite"

    def qualified(self, node: ast.expr) -> str | None:
        dotted = _dotted(node)
        if dotted is None:
            return None
        root, _, rest = dotted.partition(".")
        base = self.imports.get(root, root)
        return f"{base}.{rest}" if rest else base

    def base_kind(self, node: ast.expr) -> str | None:
        name = self.qualified(node)
        if name is None:
            return None
        if name in self.kinds:
            return self.kinds[name]
        module, _, last = name.rpartition(".")
        if last in BASES and (module == BLOCKS_MODULE or module.startswith(BLOCKS_MODULE + ".")):
            return BASES[last]
        return None

    # -- argument values -------------------------------------------------------------------

    def value(self, node: ast.expr, params: set[str]) -> dict[str, Any]:
        span = _span(node)
        try:
            return {"kind": "literal", "value": ast.literal_eval(node), "span": span}
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            pass
        if isinstance(node, ast.Name) and node.id in params:
            return {"kind": "param", "name": node.id, "span": span}
        if isinstance(node, (ast.List, ast.Tuple)):
            return {"kind": "list", "items": [self.value(e, params) for e in node.elts],
                    "span": span}
        if isinstance(node, ast.Call):
            return self.call(node, params)
        return {"kind": "expr", "source": ast.get_source_segment(self.source, node) or "",
                "span": span}

    def call(self, node: ast.Call, params: set[str]) -> dict[str, Any]:
        span = _span(node)
        dotted = _dotted(node.func)
        source = ast.get_source_segment(self.source, node) or ""
        opaque = {"kind": "opaque", "call": dotted or source.split("(")[0], "source": source,
                  "span": span}
        qualified = self.qualified(node.func)
        if qualified is None:
            return opaque
        module, _, name = qualified.rpartition(".")
        local = qualified in self.kinds and self.kinds[qualified] == "composite"
        registered = (name in self.blocks and (
            module == BLOCKS_MODULE or module.startswith(BLOCKS_MODULE + ".")))
        if not (registered or local):
            return opaque
        if node.args or any(k.arg is None for k in node.keywords):
            return {**opaque, "reason": "blocks take their options by keyword"}
        out: dict[str, Any] = {"kind": "block", "block": name, "span": span,
                               "args": {k.arg: self.value(k.value, params)
                                        for k in node.keywords if k.arg}}
        if registered:
            info = self.blocks[name]
            out.update(family=info.family, tier=info.tier)
        else:
            out["local"] = True
        return out

    # -- classes ---------------------------------------------------------------------------

    def classes(self) -> list[dict[str, Any]]:
        out = []
        for node in self.tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            kinds = {k for b in node.bases if (k := self.base_kind(b))}
            if not kinds:
                continue
            kind = sorted(kinds)[0]
            self.kinds[node.name] = kind
            entry: dict[str, Any] = {
                "name": node.name, "kind": kind, "line": node.lineno,
                "end_line": node.end_lineno or node.lineno, "span": _span(node),
                "bases": [_dotted(b) or ast.unparse(b) for b in node.bases],
            }
            try:
                entry.update(self.decoder(node) if kind == "decoder" else self.composite(node))
                entry.update(representable=True, reason=None, reason_line=None)
            except _CodeOnly as why:
                entry.update(representable=False, reason=why.reason, reason_line=why.line)
            out.append(entry)
        return out

    def _body(self, node: ast.ClassDef) -> list[ast.stmt]:
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]  # the docstring
        return body

    def decoder(self, node: ast.ClassDef) -> dict[str, Any]:
        init: ast.FunctionDef | None = None
        for stmt in self._body(node):
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__" and init is None:
                init = stmt
            elif isinstance(stmt, ast.Pass):
                continue
            else:
                raise _CodeOnly(f"the class body has {_describe(stmt)}; a Decoder subclass "
                                "holds only __init__", stmt.lineno)
        if init is None:
            return {"params": [], "args": {}, "init_span": None}
        params = _signature(init)
        names = {p["name"] for p in params}
        call = self._super_call(init)
        positional = _decoder_params()
        args: dict[str, Any] = {}
        for arg, name in zip(call.args, positional, strict=False):
            args[name] = self.value(arg, names)
        for kw in call.keywords:
            assert kw.arg is not None
            args[kw.arg] = self.value(kw.value, names)
        return {"params": params, "args": args, "init_span": _span(init),
                "call_span": _span(call)}

    def _super_call(self, init: ast.FunctionDef) -> ast.Call:
        body = init.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        if not body:
            raise _CodeOnly("__init__ does not call super().__init__", init.lineno)
        calls: list[ast.Call] = []
        for stmt in body:
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and _is_super_init(stmt.value)):
                raise _CodeOnly(f"__init__ has {_describe(stmt)}; only super().__init__(...) "
                                "can be shown as a graph", stmt.lineno)
            calls.append(stmt.value)
        if len(calls) > 1:
            raise _CodeOnly("__init__ calls super().__init__ more than once", calls[1].lineno)
        call = calls[0]
        if len(call.args) > len(_decoder_params()) or any(
                isinstance(a, ast.Starred) for a in call.args) or any(
                k.arg is None for k in call.keywords):
            raise _CodeOnly("super().__init__ takes *args or **kwargs", call.lineno)
        return call

    def composite(self, node: ast.ClassDef) -> dict[str, Any]:
        slots: list[str] | None = None
        forward = None
        for stmt in self._body(node):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _target(stmt) == "SLOTS":
                try:
                    value = ast.literal_eval(stmt.value) if stmt.value else None
                except ValueError:
                    value = None
                if not isinstance(value, (tuple, list)) or not all(
                        isinstance(v, str) for v in value):
                    raise _CodeOnly("SLOTS must be a tuple of slot names", stmt.lineno)
                slots = list(value)
            elif isinstance(stmt, ast.FunctionDef):
                if stmt.name == "__init__":
                    raise _CodeOnly("a Composite template has slots, not an __init__",
                                    stmt.lineno)
                if stmt.name == "forward":
                    forward = _span(stmt)
            elif not isinstance(stmt, ast.Pass):
                raise _CodeOnly(f"the class body has {_describe(stmt)}", stmt.lineno)
        if slots is None:
            # a Composite that adds only code inherits its base's slots
            slots = []
        return {"slots": slots, "forward_span": forward}


def _target(stmt: ast.Assign | ast.AnnAssign) -> str | None:
    target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
    return target.id if isinstance(target, ast.Name) else None


def _is_super_init(call: ast.Call) -> bool:
    func = call.func
    return (isinstance(func, ast.Attribute) and func.attr == "__init__"
            and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super" and not func.value.args)


def _describe(stmt: ast.stmt) -> str:
    return {
        ast.Assign: "an assignment", ast.AnnAssign: "an assignment", ast.AugAssign: "an assignment",
        ast.For: "a loop", ast.While: "a loop", ast.If: "an if statement",
        ast.With: "a with statement", ast.Return: "a return", ast.Expr: "another call",
        ast.FunctionDef: f"the method {getattr(stmt, 'name', '')}",
    }.get(type(stmt), "a statement outside the subset")


def _signature(init: ast.FunctionDef) -> list[dict[str, Any]]:
    args = init.args
    positional = [*args.posonlyargs, *args.args][1:]  # drop self
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    out = []
    for arg, default in zip(positional, defaults, strict=True):
        out.append(_param(arg, default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        out.append(_param(arg, default))
    return out


def _param(arg: ast.arg, default: ast.expr | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": arg.arg,
        "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
        "has_default": default is not None,
    }
    if default is not None:
        try:
            entry["default"] = ast.literal_eval(default)
        except (ValueError, TypeError, SyntaxError):
            entry["default_source"] = ast.unparse(default)
    return entry


def parse(path: str | Path) -> dict[str, Any]:
    """The graph of every Decoder/Composite subclass in `path`. Never imports the file."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"{path}:{exc.lineno}: {exc.msg}") from exc
    return {"schema": 1, "nanoscope": __version__, "path": str(path),
            "classes": _Parser(source, tree).classes()}
