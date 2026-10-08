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

A class that fills a lesson template (`class OneHead(AttentionTemplate)` whose `__init__` only
calls `super().__init__(d_model, context_length, q=Linear(), ...)`) has kind "filled": its slots are
the arguments, edited like a Decoder's.

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

import nanoscope.blocks.registry as registry
from nanoscope import __version__

BLOCKS_MODULE = "nanoscope.blocks"
BASES = {"Decoder": "decoder", "Composite": "composite"}
TEMPLATE_PARAMS = ["d_model", "context_length"]  # what a Composite takes before its slots


def _known_blocks() -> dict[str, registry.BlockInfo]:
    blocks = importlib.import_module(BLOCKS_MODULE)
    blocks.load_all()  # registers every block, whatever is locked
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
        info = self.blocks.get(last)
        if info and info.family == "template" and (
                module == BLOCKS_MODULE or module.startswith(BLOCKS_MODULE + ".")):
            return "filled"
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
                if kind == "decoder":
                    entry.update(self.decoder(node))
                elif kind == "filled":
                    entry.update(self.filled(node))
                else:
                    entry.update(self.composite(node))
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
        return self._init_args(node, _decoder_params())

    def filled(self, node: ast.ClassDef) -> dict[str, Any]:
        """A class that fills a template: its slots are keyword arguments of super().__init__."""
        out = self._init_args(node, TEMPLATE_PARAMS)
        template = ""
        for base in node.bases:
            last = (_dotted(base) or "").rsplit(".", 1)[-1]
            if last in self.blocks and self.blocks[last].family == "template":
                template = last
        info = self.blocks[template]
        out["template"] = template
        out["slots"] = list(getattr(importlib.import_module(info.module), template).SLOTS)
        return out

    def _init_args(self, node: ast.ClassDef, positional: list[str]) -> dict[str, Any]:
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
        call = self._super_call(init, positional)
        args: dict[str, Any] = {}
        for arg, name in zip(call.args, positional, strict=False):
            args[name] = self.value(arg, names)
        for kw in call.keywords:
            assert kw.arg is not None
            args[kw.arg] = self.value(kw.value, names)
        return {"params": params, "args": args, "init_span": _span(init),
                "call_span": _span(call)}

    def _super_call(self, init: ast.FunctionDef, positional: list[str]) -> ast.Call:
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
        if len(call.args) > len(positional) or any(
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


# -- emit ----------------------------------------------------------------------------------
#
# `emit(graph, source)` turns an edited graph back into source. It compares the graph with
# what `parse(source)` says today, turns each difference into an edit, and applies the edits
# with libcst, which keeps every comment and every untouched character. Edits address a node
# by a path from the class's `super().__init__` call: an argument name selects that argument
# of the current call, an integer selects an item of the current list.
#
#   {"op": "set_arg", "class": "MyLM", "path": ["block", "attn"], "arg": "n_heads", "value": node}
#   {"op": "remove_arg", "class": "MyLM", "path": ["block", "attn"], "arg": "window"}
#   {"op": "replace_block", "class": "MyLM", "path": ["block", "attn", "pos"], "node": node}
#
# Structural edits (the model page's stack and slot edits):
#
#   {"op": "add_layer", "class": "MyLM"}                      n_layers + 1 (a literal)
#   {"op": "add_layer", "class": "MyLM", "path": ["pattern"], "node": node, "index": 1}
#                                                             insert into a list (index: the end)
#   {"op": "remove_layer", "class": "MyLM"}                   n_layers - 1 (never below 1)
#   {"op": "remove_layer", "class": "MyLM", "path": ["pattern"], "index": 1}
#   {"op": "set_pattern", "class": "MyLM", "items": [node, ...]}   pattern=[...], drops block=
#   {"op": "fill_slot", "class": "MyLM", "path": ["block", "attn"], "slot": "mask", "node": node}
#                                                             (node None empties the slot)

DERIVED = ("span", "family", "tier", "local", "reason")


def strip_spans(node: Any) -> Any:
    """A graph or node without source positions and the fields parse derives from the
    registry, so two graphs compare equal when they mean the same thing."""
    if isinstance(node, dict):
        return {k: strip_spans(v) for k, v in node.items()
                if k not in DERIVED and k not in ("path", "nanoscope", "line", "end_line",
                                                 "init_span", "call_span", "forward_span")}
    if isinstance(node, list):
        return [strip_spans(v) for v in node]
    return node


def diff(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """The edits that turn graph `old` into graph `new` (same classes, argument changes)."""
    old_classes = {c["name"]: c for c in old["classes"]}
    edits: list[dict[str, Any]] = []
    if [c["name"] for c in new["classes"]] != [c["name"] for c in old["classes"]]:
        raise ValueError("emit changes arguments; it cannot add, remove or rename classes")
    for cls in new["classes"]:
        before = old_classes[cls["name"]]
        if strip_spans(before) == strip_spans(cls):
            continue
        if not before["representable"]:
            raise ValueError(f"{cls['name']} is code-only ({before['reason']}), so its graph "
                             "can't be edited; edit the source")
        same_args = strip_spans(before.get("args")) == strip_spans(cls.get("args"))
        if before["kind"] not in ("decoder", "filled") or same_args:
            raise ValueError(f"{cls['name']}: only the arguments of a Decoder or a filled "
                             "template can be edited")
        _diff_args(cls["name"], before["args"], cls["args"], [], edits)
    return edits


def _diff_args(cls: str, old: dict[str, Any], new: dict[str, Any], path: list[Any],
               edits: list[dict[str, Any]]) -> None:
    for name, node in new.items():
        if name not in old:
            edits.append({"op": "set_arg", "class": cls, "path": path, "arg": name,
                          "value": node})
        else:
            _diff_node(cls, old[name], node, path, name, edits)
    for name in old:
        if name not in new:
            edits.append({"op": "remove_arg", "class": cls, "path": path, "arg": name})


def _diff_node(cls: str, old: Any, new: Any, owner: list[Any], name: str,
               edits: list[dict[str, Any]]) -> None:
    if strip_spans(old) == strip_spans(new):
        return
    here = [*owner, name]
    if old["kind"] == new["kind"] == "block" and old["block"] == new["block"] \
            and bool(old.get("local")) == bool(new.get("local")):
        _diff_args(cls, old["args"], new["args"], here, edits)
    elif old["kind"] == new["kind"] == "list" and _one_item_apart(old, new) is not None:
        op, index, node = _one_item_apart(old, new)  # type: ignore[misc]
        edits.append({"op": op, "class": cls, "path": here, "index": index,
                      **({"node": node} if node is not None else {})})
    elif old["kind"] == new["kind"] == "list" and len(old["items"]) == len(new["items"]):
        for i, (a, b) in enumerate(zip(old["items"], new["items"], strict=True)):
            if strip_spans(a) == strip_spans(b):
                continue
            if a["kind"] == b["kind"] == "block" and a["block"] == b["block"]:
                _diff_args(cls, a["args"], b["args"], [*here, i], edits)
            else:
                edits.append({"op": "replace_block", "class": cls, "path": [*here, i],
                              "node": b})
    elif new["kind"] in ("block", "list") or old["kind"] in ("block", "list"):
        edits.append({"op": "replace_block", "class": cls, "path": here, "node": new})
    else:
        edits.append({"op": "set_arg", "class": cls, "path": owner, "arg": name, "value": new})


def _one_item_apart(old: dict[str, Any], new: dict[str, Any]) -> tuple[str, int, Any] | None:
    """("add_layer", i, node) or ("remove_layer", i, None) when the lists differ by one item."""
    a = [strip_spans(x) for x in old["items"]]
    b = [strip_spans(x) for x in new["items"]]
    if len(b) == len(a) + 1:
        for i in range(len(b)):
            if b[:i] + b[i + 1:] == a:
                return "add_layer", i, new["items"][i]
    elif len(b) == len(a) - 1 and b:
        for i in range(len(a)):
            if a[:i] + a[i + 1:] == b:
                return "remove_layer", i, None
    return None


def render(node: dict[str, Any], prefix: str = "") -> str:
    """Source text for a graph node. `prefix` is put before block names ("nb.")."""
    kind = node["kind"]
    if kind == "literal":
        return repr(node["value"])
    if kind == "param":
        return str(node["name"])
    if kind in ("opaque", "expr"):
        return str(node["source"])
    if kind == "list":
        return "[" + ", ".join(render(n, prefix) for n in node["items"]) + "]"
    if kind == "block":
        name = node["block"] if node.get("local") else prefix + node["block"]
        args = ", ".join(f"{k}={render(v, prefix)}" for k, v in node["args"].items())
        return f"{name}({args})"
    raise ValueError(f"unknown node kind {kind!r}")


def _block_names(node: dict[str, Any]) -> list[str]:
    if node["kind"] == "list":
        return [n for item in node["items"] for n in _block_names(item)]
    if node["kind"] != "block":
        return []
    names = [] if node.get("local") else [node["block"]]
    return names + [n for v in node["args"].values() for n in _block_names(v)]


def emit(graph: dict[str, Any], source: str) -> str:
    """`source` with the arguments `graph` changes patched in, and nothing else touched."""
    with_edits = diff(_parse_source(source), graph)
    return apply_edits(source, with_edits) if with_edits else source


def _parse_source(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    return {"schema": 1, "nanoscope": __version__, "path": "",
            "classes": _Parser(source, tree).classes()}


def apply_edits(source: str, edits: list[dict[str, Any]]) -> str:
    """Apply edits (see above) to `source` with libcst. Raises ValueError for an edit that
    names a class, argument or position that is not there."""
    import libcst as cst

    module = cst.parse_module(source)
    existing = _parse_source(source)
    classes = {c["name"]: c for c in existing["classes"]}
    for edit in edits:
        cls = classes.get(edit["class"])
        if cls is None:
            raise ValueError(f"no Decoder or Composite class {edit['class']!r} in the file")
        if not cls["representable"]:
            raise ValueError(f"{cls['name']} is code-only ({cls['reason']}); edit the source")
        if cls["kind"] == "composite":
            raise ValueError(f"{cls['name']} is a template; its slots are filled where it is used")
        if edit["op"] == "fill_slot":
            edit = _fill_slot_as_set_arg(edit, cls, existing)
        module = _apply(cst, module, edit,
                        TEMPLATE_PARAMS if cls["kind"] == "filled" else _decoder_params())
    return module.code


def _fill_slot_as_set_arg(edit: dict[str, Any], cls: dict[str, Any],
                          existing: dict[str, Any]) -> dict[str, Any]:
    """A fill_slot is a set_arg on the template's call, once the target is known to be a
    template with that slot."""
    target: Any = {"kind": "block", "args": cls["args"]}
    if cls["kind"] == "filled":  # the class itself is the template
        target.update(block=cls["template"], family="template")
    for step in edit.get("path", []):
        target = target["args"][step] if isinstance(step, str) and target["kind"] == "block" \
            and step in target["args"] else None
        if target is None:
            raise ValueError(f"no argument {step!r} at {_where(edit)}")
    if target["kind"] != "block" or not (target.get("local") or target.get("family") == "template"):
        raise ValueError(f"{_where(edit)} is not a template; fill_slot fills template slots")
    slots = _slots_of(target, existing)
    if slots is not None and edit["slot"] not in slots:
        raise ValueError(f"{target['block']} has the slots {', '.join(slots)}, "
                         f"not {edit['slot']!r}")
    node = edit.get("node")
    value = node if node is not None else {"kind": "literal", "value": None, "span": None}
    return {"op": "set_arg", "class": edit["class"], "path": edit.get("path", []),
            "arg": edit["slot"], "value": value}


def _slots_of(node: dict[str, Any], existing: dict[str, Any]) -> list[str] | None:
    if node.get("local"):
        found = next((c for c in existing["classes"] if c["name"] == node["block"]), None)
        return found.get("slots") if found else None
    info = next((i for i in registry.all_blocks() if i.name == node["block"]), None)
    if info is None or info.user:
        return None
    cls = getattr(importlib.import_module(info.module), info.name, None)
    slots = getattr(cls, "SLOTS", None)
    return list(slots) if slots else None


def _apply(cst: Any, module: Any, edit: dict[str, Any], positional: list[str]) -> Any:
    op = edit["op"]
    nodes = {"set_arg": [edit.get("value")], "replace_block": [edit.get("node")],
             "remove_arg": [], "add_layer": [edit.get("node")], "remove_layer": [],
             "set_pattern": list(edit.get("items") or [])}.get(op)
    if nodes is None:
        raise ValueError(f"unknown edit {op!r}")
    module, prefix = _ensure_imports(cst, module, [n for node in nodes if node
                                                   for n in _block_names(node)])
    def on_call(call: Any, path: list[Any], root: bool) -> Any:
        if not path:
            if op == "replace_block":
                raise ValueError("replace_block needs a path to the node it replaces")
            if op in ("add_layer", "remove_layer"):
                if "node" in edit or "index" in edit:
                    raise ValueError(f"{op} on a list needs the path of the list "
                                     f"({_where(edit)} is a call)")
                return _bump_layers(cst, call, edit, positional if root else [])
            if op == "set_pattern":
                return _set_pattern(cst, call, edit, prefix, positional if root else [])
            return _change_arg(cst, call, edit, prefix, positional if root else [])
        step, rest = path[0], path[1:]
        if not isinstance(step, str):
            raise ValueError(f"path step {step!r} selects an argument by name here")
        index = _arg_index(call, step, positional if root else [])
        if index is None:
            raise ValueError(f"no argument {step!r} at {_where(edit)}")
        arg = call.args[index]
        if not rest and op == "replace_block":
            value = cst.parse_expression(render(edit["node"], prefix))
        else:
            value = on_value(arg.value, rest)
        args = list(call.args)
        args[index] = arg.with_changes(value=value)
        return call.with_changes(args=args)

    def on_value(value: Any, path: list[Any]) -> Any:
        if not path:
            if op in ("add_layer", "remove_layer") and isinstance(value, (cst.List, cst.Tuple)):
                return _edit_list(cst, value, edit, prefix)
            if not isinstance(value, cst.Call):
                raise ValueError(f"{_where(edit)} is not a block call")
            return on_call(value, path, False)
        step, rest = path[0], path[1:]
        if isinstance(step, int):
            if not isinstance(value, (cst.List, cst.Tuple)) or step >= len(value.elements):
                raise ValueError(f"no item {step} at {_where(edit)}")
            element = value.elements[step]
            if not rest and op == "replace_block":
                new = cst.parse_expression(render(edit["node"], prefix))
            else:
                new = on_value(element.value, rest)
            elements = list(value.elements)
            elements[step] = element.with_changes(value=new)
            return value.with_changes(elements=elements)
        if not isinstance(value, cst.Call):
            raise ValueError(f"{_where(edit)} is not a block call")
        return on_call(value, path, False)

    class Patch(cst.CSTTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.done = False
            self.in_class = False

        def visit_ClassDef(self, node: Any) -> bool:
            self.in_class = node.name.value == edit["class"]
            return self.in_class

        def leave_ClassDef(self, original: Any, updated: Any) -> Any:
            self.in_class = False
            return updated

        def leave_Call(self, original: Any, updated: Any) -> Any:
            if not self.in_class or self.done or not _cst_is_super_init(cst, updated):
                return updated
            self.done = True
            return on_call(updated, list(edit.get("path", [])), True)

    patch = Patch()
    result = module.visit(patch)
    if not patch.done:
        raise ValueError(f"{edit['class']} has no super().__init__(...) call to edit")
    return result


def _where(edit: dict[str, Any]) -> str:
    return f"{edit['class']}: " + "/".join(str(p) for p in edit.get("path", [])) or edit["class"]


def _cst_is_super_init(cst: Any, call: Any) -> bool:
    func = call.func
    return (isinstance(func, cst.Attribute) and func.attr.value == "__init__"
            and isinstance(func.value, cst.Call) and isinstance(func.value.func, cst.Name)
            and func.value.func.value == "super")


def _arg_index(call: Any, name: str, positional: list[str]) -> int | None:
    for i, arg in enumerate(call.args):
        if arg.keyword is not None and arg.keyword.value == name:
            return i
        if arg.keyword is None and i < len(positional) and positional[i] == name:
            return i
    return None


def _change_arg(cst: Any, call: Any, edit: dict[str, Any], prefix: str,
                positional: list[str]) -> Any:
    name = edit["arg"]
    index = _arg_index(call, name, positional)
    if edit["op"] == "remove_arg":
        if index is None:
            raise ValueError(f"no argument {name!r} to remove at {_where(edit)}")
        if call.args[index].keyword is None and index < len(call.args) - 1:
            raise ValueError(f"{name!r} is positional; it can only be removed from the end")
        args = [a for i, a in enumerate(call.args) if i != index]
        if args and index == len(call.args) - 1:  # the new last argument takes over its tail
            args[-1] = _take_tail(cst, args[-1], call.args[index].comma)
        return call.with_changes(args=args)
    value = cst.parse_expression(render(edit["value"], prefix))
    args = list(call.args)
    if index is not None:
        args[index] = args[index].with_changes(value=value)
    else:
        new = cst.Arg(keyword=cst.Name(name), value=value,
                      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""),
                                            whitespace_after=cst.SimpleWhitespace("")))
        tail = args[-1].comma if args else cst.MaybeSentinel.DEFAULT
        if isinstance(tail, cst.Comma):  # a trailing comma (and the newline after it) moves on
            first = args[0].comma
            args[-1] = args[-1].with_changes(comma=first if isinstance(first, cst.Comma)
                                             and len(args) > 1 else cst.Comma(
                whitespace_after=cst.SimpleWhitespace(" ")))
            new = new.with_changes(comma=tail)
        args.append(new)
    return call.with_changes(args=args)


def _take_tail(cst: Any, element: Any, tail: Any) -> Any:
    """`element` (now last) ends with `tail`, the comma the removed last element had, unless its
    own comma carries a comment: that comment stays."""
    after: Any = getattr(element.comma, "whitespace_after", None)
    if isinstance(after, cst.ParenthesizedWhitespace) and after.first_line.comment is not None:
        return element
    return element.with_changes(comma=tail)


def _bump_layers(cst: Any, call: Any, edit: dict[str, Any], positional: list[str]) -> Any:
    index = _arg_index(call, "n_layers", positional)
    if index is None:
        raise ValueError(f"{edit['class']} has no n_layers argument to change")
    value = call.args[index].value
    if not isinstance(value, cst.Integer):
        raise ValueError(f"n_layers is not a number literal in {edit['class']}; edit the source")
    now = int(value.value)
    new = now + 1 if edit["op"] == "add_layer" else now - 1
    if new < 1:
        raise ValueError("a model needs at least one layer")
    args = list(call.args)
    args[index] = args[index].with_changes(value=value.with_changes(value=str(new)))
    return call.with_changes(args=args)


def _set_pattern(cst: Any, call: Any, edit: dict[str, Any], prefix: str,
                 positional: list[str]) -> Any:
    items = edit.get("items")
    if not items:
        raise ValueError("a pattern needs at least one block")
    if _arg_index(call, "block", positional) is not None:
        call = _change_arg(cst, call, {"op": "remove_arg", "class": edit["class"],
                                       "arg": "block", "path": []}, prefix, positional)
    node: dict[str, Any] = {"kind": "list", "items": items, "span": None}
    before: Any = call.whitespace_before_args
    if isinstance(before, cst.ParenthesizedWhitespace) and len(items) > 1:
        # a call written over several lines gets one block per line, indented like its arguments
        indent = before.last_line.value
        lines = "".join(f"{indent}    {render(item, prefix)},\n" for item in items)
        node = {"kind": "expr", "source": f"[\n{lines}{indent}]", "span": None}
    return _change_arg(cst, call, {"op": "set_arg", "class": edit["class"], "path": [],
                                   "arg": "pattern", "value": node}, prefix, positional)


def _edit_list(cst: Any, value: Any, edit: dict[str, Any], prefix: str) -> Any:
    elements = list(value.elements)
    index = edit.get("index")
    if edit["op"] == "add_layer":
        if edit.get("node") is None:
            raise ValueError(f"add_layer on a list needs a node ({_where(edit)})")
        at = len(elements) if index is None else index
        if not 0 <= at <= len(elements):
            raise ValueError(f"no position {index} in the list at {_where(edit)}")
        new = cst.Element(value=cst.parse_expression(render(edit["node"], prefix)))
        first = elements[0].comma
        sep = first if isinstance(first, cst.Comma) and len(elements) > 1 else cst.Comma(
            whitespace_after=cst.SimpleWhitespace(" "))
        if at == len(elements):  # the new last item takes the old tail
            tail = elements[-1].comma
            elements[-1] = elements[-1].with_changes(comma=sep)
            new = new.with_changes(comma=tail)
        else:
            new = new.with_changes(comma=sep)
        elements.insert(at, new)
        return value.with_changes(elements=elements)
    if index is None or not 0 <= index < len(elements):
        raise ValueError(f"no item {index} to remove at {_where(edit)}")
    if len(elements) == 1:
        raise ValueError(f"the list at {_where(edit)} would be empty; a pattern needs a block")
    removed = elements.pop(index)
    if index == len(elements):  # the new last item takes the old tail (trailing comma or none)
        elements[-1] = _take_tail(cst, elements[-1], removed.comma)
    if isinstance(value, cst.Tuple) and len(elements) == 1 and \
            elements[0].comma is cst.MaybeSentinel.DEFAULT:
        elements[0] = elements[0].with_changes(comma=cst.Comma())
    return value.with_changes(elements=elements)


def _ensure_imports(cst: Any, module: Any, names: list[str]) -> tuple[Any, str]:
    """Make every block name in `names` usable in the module. Returns the module and the
    prefix to put before block names ("" for `from nanoscope.blocks import X`, "nb." for an
    `import nanoscope.blocks as nb` file)."""
    imported: set[str] = set()
    alias: str | None = None
    from_import = None
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.ImportFrom) and small.module is not None and \
                    _dotted_cst(cst, small.module) == BLOCKS_MODULE and \
                    not isinstance(small.names, cst.ImportStar):
                from_import = (stmt, small)
                imported |= {a.name.value for a in small.names if isinstance(a.name, cst.Name)}
            elif isinstance(small, cst.Import):
                for a in small.names:
                    if _dotted_cst(cst, a.name) == BLOCKS_MODULE and a.asname is not None:
                        alias = a.asname.name.value  # type: ignore[attr-defined]
    missing = sorted({n for n in names if n not in imported})
    if not missing:
        return module, ""
    if from_import is None and alias is not None:
        return module, alias + "."
    if from_import is not None:
        stmt, small = from_import
        added = [cst.ImportAlias(name=cst.Name(n)) for n in missing]
        names_now = list(small.names)
        merged = sorted([*names_now, *added], key=lambda a: a.name.value) \
            if _is_sorted(names_now) else [*names_now, *added]
        new_stmt = stmt.with_changes(body=[small.with_changes(
            names=_fix_commas(cst, small, merged))])
        return module.with_changes(body=[new_stmt if s is stmt else s for s in module.body]), ""
    line = cst.parse_statement(f"from {BLOCKS_MODULE} import {', '.join(missing)}\n")
    body = list(module.body)
    last = max((i for i, s in enumerate(body) if isinstance(s, cst.SimpleStatementLine)
                and isinstance(s.body[0], (cst.Import, cst.ImportFrom))), default=-1)
    body.insert(last + 1, line)
    return module.with_changes(body=body), ""


def _is_sorted(aliases: list[Any]) -> bool:
    names = [a.name.value for a in aliases]
    return names == sorted(names)


def _fix_commas(cst: Any, small: Any, aliases: list[Any]) -> list[Any]:
    """Commas between names; a parenthesised import keeps its own trailing-comma style."""
    out = []
    for i, alias in enumerate(aliases):
        last = i == len(aliases) - 1
        if not last and alias.comma is cst.MaybeSentinel.DEFAULT:
            alias = alias.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
        out.append(alias)
    return out


def _dotted_cst(cst: Any, node: Any) -> str:
    if isinstance(node, cst.Name):
        return node.value
    return f"{_dotted_cst(cst, node.value)}.{node.attr.value}"


# -- text -----------------------------------------------------------------------------------

def format_graph(graph: dict[str, Any], only: str | None = None) -> str:
    """The graph as an indented tree, one class per paragraph."""
    paragraphs = []
    for cls in graph["classes"]:
        if only and cls["name"] != only:
            continue
        head = f"{cls['name']}({cls['kind'].capitalize()})  {graph['path']}:{cls['line']}"
        if not cls["representable"]:
            paragraphs.append(f"{head}\n  code-only, line {cls['reason_line']}: {cls['reason']}")
        elif cls["kind"] == "composite":
            paragraphs.append(f"{head}\n  slots: {', '.join(cls['slots']) or 'none'}")
        else:
            lines = [head]
            for name, node in cls["args"].items():
                lines += _tree(name, node, 1)
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def _tree(name: str, node: dict[str, Any], depth: int) -> list[str]:
    pad = "  " * depth
    kind = node["kind"]
    if kind == "literal":
        return [f"{pad}{name} = {node['value']!r}"]
    if kind == "param":
        return [f"{pad}{name} = <{node['name']}>"]
    if kind == "expr":
        return [f"{pad}{name} = {node['source']}  (expression)"]
    if kind == "opaque":
        return [f"{pad}{name} = {node['call']}  (custom block, not editable)"]
    if kind == "list":
        lines = [f"{pad}{name} = ["]
        for i, item in enumerate(node["items"]):
            lines += _tree(str(i), item, depth + 1)
        return lines
    lines = [f"{pad}{name} = {node['block']}" + ("  (template)" if node.get("local") else "")]
    for key, child in node["args"].items():
        lines += _tree(key, child, depth + 1)
    return lines
