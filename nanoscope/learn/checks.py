"""A lesson's checks: small experiments that decide whether the learner got it.

Each kind in `loader.CHECK_KINDS` has a function in `CHECKERS`:

    def check_defines(ctx: Context, check: Check) -> Result

`run_lesson_checks` runs a lesson's checks in order, tells the learner each verdict and its
reason as it goes, and writes `learn/checks/<id>.json` (`check.v1`). Every verdict carries a
readable reason, pass or fail. If a `defines` check fails, the rest are skipped: they would
only fail for the same reason.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths
from nanoscope.fsutil import write_json_atomic
from nanoscope.learn import progress, unlocks
from nanoscope.learn.loader import Check, LessonSpec
from nanoscope.log import info


@dataclass
class Result:
    id: str
    kind: str
    passed: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "passed": self.passed, "skipped": self.skipped,
                "reason": self.reason, "evidence": self.evidence,
                "seconds": round(self.seconds, 3)}


@dataclass
class Context:
    """What a check can see: the lesson, the learner's file, and whose progress this is."""

    lesson: LessonSpec
    owner: str = "local"
    variant: str = "cpu"  # which [compute.*] variant to run
    check_id: str = "adhoc"  # names the folder this run's training runs go in
    shared: dict[str, Any] = field(default_factory=dict)  # checks pass things on (the class)

    @property
    def user_file(self) -> Path:
        from nanoscope.learn.cli import workspace_lesson_dir

        return workspace_lesson_dir(self.lesson) / "starter.py"


    @property
    def runs_dir(self) -> Path:
        """Training runs made by checks: a fresh folder per check, so edits to the learner's
        file are never answered with an old finished run."""
        return paths.runs_dir() / "lessons" / self.lesson.path / self.lesson.slug / self.check_id


Checker = Callable[[Context, Check], Result]
CHECKERS: dict[str, Checker] = {}


def checker(kind: str) -> Callable[[Checker], Checker]:
    def register(fn: Checker) -> Checker:
        CHECKERS[kind] = fn
        return fn
    return register


def check_id(lesson: LessonSpec) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{lesson.path}-{lesson.slug}-{stamp}"


def checks_dir(owner: str = "local") -> Path:
    return paths.learn_dir(owner) / "checks"


def run_one(ctx: Context, check: Check) -> Result:
    started = time.perf_counter()
    fn = CHECKERS.get(check.kind)
    try:
        if fn is None:
            result = Result(check.id, check.kind, False, f"no checker for {check.kind!r} yet")
        else:
            result = fn(ctx, check)
    except ImportError as exc:
        from nanoscope.learn.gating import LockedBlockError

        if not isinstance(exc, LockedBlockError):
            frames = traceback.extract_tb(exc.__traceback__)
            where = f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
            result = Result(check.id, check.kind, False,
                            f"crashed: {type(exc).__name__}: {exc}{where}")
            result.seconds = time.perf_counter() - started
            ctx.shared.setdefault("evidence", {})[check.id] = result.evidence
            return result
        # not a crash: the file asks for something the learner has not unlocked yet
        result = Result(check.id, check.kind, False, f"your file needs a locked block. {exc}")
    except Exception as exc:  # the learner's code, or ours: say which line, never a traceback
        frames = traceback.extract_tb(exc.__traceback__)
        where = f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
        result = Result(check.id, check.kind, False,
                        f"crashed: {type(exc).__name__}: {exc}{where}")
    result.seconds = time.perf_counter() - started
    ctx.shared.setdefault("evidence", {})[check.id] = result.evidence  # later checks can score it
    return result


def run_lesson_checks(lesson: LessonSpec, owner: str = "local",
                      variant: str = "cpu") -> dict[str, Any]:
    """Run every check of `lesson`, print each verdict, record progress and the result file.
    Returns the check.v1 document."""
    cid = check_id(lesson)
    progress.mark(lesson.id, "checking", check_id=cid, owner=owner)
    ctx = Context(lesson, owner, variant, cid)
    n = len(lesson.checks)
    info(f"checking {lesson.id} ({n} check{'s' if n != 1 else ''})")
    results: list[Result] = []
    blocked = False
    for check in lesson.checks:
        if blocked:
            result = Result(check.id, check.kind, False,
                            "skipped: fix the failing 'defines' check first", skipped=True)
        else:
            result = run_one(ctx, check)
        results.append(result)
        info(f"  [{'pass' if result.passed else 'skip' if result.skipped else 'FAIL'}] "
             f"{check.id} ({check.kind}): {result.reason}")
        blocked = blocked or (check.kind == "defines" and not result.passed)
    passed = bool(results) and all(r.passed for r in results)
    doc = {
        "schema": 1, "nanoscope": __version__, "id": cid, "lesson": lesson.id,
        "at": progress.now(), "variant": variant, "passed": passed,
        "checks": [r.to_dict() for r in results],
    }
    write_json_atomic(checks_dir(owner) / f"{cid}.json", doc)
    earned: list[str] = []
    if passed and lesson.unlocks and unlocks.exists(owner):
        # only a pass earns, and the evidence is this very result file
        unlocks.earn(lesson.id, lesson.unlocks, f"learn/checks/{cid}.json", owner)
        earned = lesson.unlocks
    progress.mark(lesson.id, "passed" if passed else "failed", check_id=cid, owner=owner)
    if earned:
        info(f"unlocked: {', '.join(earned)}")
    done = sum(r.passed for r in results)
    info(f"result: {done} of {len(results)} checks passed"
         + ("" if passed else ": read the reasons above, edit your file, and check again"))
    return doc


# -- loading the learner's file --------------------------------------------------------------

DEFAULT_ARGS = {"d_model": 16, "context_length": 8, "vocab_size": 50, "n_heads": 2,
                "n_layers": 2}  # small sizes for a class's required constructor arguments


def load_user_module(ctx: Context) -> Any:
    """Import the learner's file fresh (cached for the rest of this check run)."""
    import importlib.util
    import sys

    if "module" in ctx.shared:
        return ctx.shared["module"]
    file = ctx.user_file
    if not file.exists():
        raise FileNotFoundError(
            f"{file} does not exist: run `nanoscope learn start {ctx.lesson.id}` first")
    name = f"_nanoscope_lesson_{abs(hash((str(file), file.stat().st_mtime_ns)))}"
    spec = importlib.util.spec_from_file_location(name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # compiled from the text, not imported: the bytecode cache is keyed on the file's
        # modification second and size, so an edit of the same length made within a second
        # of the last check would otherwise run the old code
        exec(compile(file.read_text(encoding="utf-8"), str(file), "exec"), module.__dict__)
    except SyntaxError as exc:
        raise ValueError(f"{file.name} line {exc.lineno}: {exc.msg}") from exc
    finally:
        sys.modules.pop(name, None)
    ctx.shared["module"] = module
    return module


def user_class(ctx: Context, check: Check) -> type:
    """The class a check is about: its `class` argument, else the one `defines` found."""
    import torch.nn as nn

    wanted = check.args.get("class") or ctx.shared.get("class_name")
    if not wanted:
        raise ValueError("no class to check: name it with class = \"...\" in the check")
    module = load_user_module(ctx)
    cls = getattr(module, wanted, None)
    if cls is None:
        known = [n for n, v in vars(module).items()
                 if isinstance(v, type) and issubclass(v, nn.Module)
                 and v.__module__ == module.__name__]
        raise AttributeError(
            f"class {wanted} is not defined in {ctx.user_file.name}"
            + (f" (it defines: {', '.join(known)})" if known else ""))
    return cls


def constructor_args(cls: type, given: dict[str, Any] | None) -> dict[str, Any]:
    """The lesson's `args` plus small defaults for any required argument it did not give."""
    import inspect

    from nanoscope.blocks.spec import BlockModule

    args = dict(given or {})
    # a registered block's __new__ takes (*args, **kwargs); its real arguments are __init__'s
    signature = inspect.signature(cls.__init__) if issubclass(cls, BlockModule) \
        else inspect.signature(cls)
    for name, param in signature.parameters.items():
        variadic = param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        if name == "self" or name in args or variadic:
            continue
        if param.default is inspect.Parameter.empty:
            if name not in DEFAULT_ARGS:
                raise TypeError(f"{cls.__name__}() needs a value for {name}; the lesson's check "
                                "should list it under args")
            args[name] = DEFAULT_ARGS[name]
    return args


@checker("defines")
def check_defines(ctx: Context, check: Check) -> Result:
    """The class exists in the learner's file and builds (on the meta device: no memory)."""
    import torch

    from nanoscope.sizing import count_params

    ctx.shared.pop("module", None)  # re-read the file: this is where a check run starts
    ctx.shared["class_name"] = check.args["class"]
    try:
        cls = user_class(ctx, check)
    except (FileNotFoundError, AttributeError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    try:
        kwargs = constructor_args(cls, check.args.get("args"))
    except TypeError as exc:
        return Result(check.id, check.kind, False, str(exc))
    try:
        with torch.device("meta"):
            built = cls(**kwargs)
    except Exception as exc:
        return Result(check.id, check.kind, False,
                      f"{cls.__name__}({_kwargs_text(kwargs)}) failed to build: "
                      f"{type(exc).__name__}: {exc}")
    total, _ = count_params(built)
    return Result(check.id, check.kind, True,
                  f"{cls.__name__}({_kwargs_text(kwargs)}) builds, {total:,} parameters",
                  {"class": cls.__name__, "parameters": total})


@checker("certified")
def check_certified(ctx: Context, check: Check) -> Result:
    """The learner registered the block with a reference of their own (the certification the
    palette shows), AND the block matches the lesson's reference. The first alone could pass with
    a wrong formula written twice; the second alone would not teach `register_block`."""
    import nanoscope.blocks.registry as registry
    from nanoscope.blocks.certify import run_check
    from nanoscope.reference import functional

    try:
        cls = user_class(ctx, check)
    except (FileNotFoundError, AttributeError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    name = cls.__name__
    own = registry.reference_for(name)
    if own is None:
        return Result(check.id, check.kind, False,
                      f"{name} is not registered with a reference: decorate it with "
                      "@register_block(reference=your_function)")
    truth = getattr(functional, check.args["against"], None)
    if truth is None:
        return Result(check.id, check.kind, False,
                      f"the lesson names reference {check.args['against']!r}, which "
                      "nanoscope.reference doesn't have: this is a bug in the lesson")
    mine = run_check(cls, own)
    if not mine["passed"]:
        return Result(check.id, check.kind, False,
                      f"{name} does not match its own reference {own.__name__}: {mine['message']}")
    theirs = run_check(cls, truth)
    if not theirs["passed"]:
        return Result(check.id, check.kind, False,
                      f"{name} matches your reference but not the lesson's "
                      f"({check.args['against']}): {theirs['message']}. Both can be wrong "
                      "the same way: check the formula, not only the code")
    return Result(check.id, check.kind, True,
                  f"{name} is registered with reference {own.__name__} and matches it and the "
                  f"lesson's: {theirs['message']}",
                  {"class": name, "reference": own.__name__,
                   "max_abs_diff": theirs["max_abs_diff"]})


@checker("preset")
def check_preset(ctx: Context, check: Check) -> Result:
    """The learner's file registers a preset `name` derived from another: the fields in `same`
    are unchanged (so the same cached tokens apply), the fields in `differs` changed, the run is
    at most `max_steps`, warmup is at most `max_warmup_fraction` of it, and evaluation happens
    at least once."""
    from nanoscope.presets import get_preset

    name, base_name = check.args["name"], check.args["derived_from"]
    try:
        load_user_module(ctx)
    except (FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    except Exception as exc:
        return Result(check.id, check.kind, False,
                      f"{ctx.user_file.name} raised {type(exc).__name__}: {exc}")
    try:
        mine, base = get_preset(name), get_preset(base_name)
    except KeyError:
        return Result(check.id, check.kind, False,
                      f"no preset named {name!r} is registered by {ctx.user_file.name}: call "
                      f"register_preset(Preset(name={name!r}, ...))")
    for field_name in check.args.get("same", []):
        if getattr(mine, field_name) != getattr(base, field_name):
            return Result(check.id, check.kind, False,
                          f"{field_name} is {getattr(mine, field_name)!r} but {base_name} has "
                          f"{getattr(base, field_name)!r}: keep it, or the cached tokens and the "
                          "comparison with that preset no longer apply")
    for field_name in check.args.get("differs", []):
        if getattr(mine, field_name) == getattr(base, field_name):
            return Result(check.id, check.kind, False,
                          f"{field_name} is the same as {base_name}'s "
                          f"({getattr(base, field_name)!r}): the lesson wants it changed")
    limit = check.args.get("max_steps")
    if limit is not None and mine.max_steps > limit:
        return Result(check.id, check.kind, False,
                      f"max_steps is {mine.max_steps}; this preset is for a quick run, "
                      f"at most {limit}")
    fraction = check.args.get("max_warmup_fraction")
    if fraction is not None and mine.warmup_steps > fraction * mine.max_steps:
        return Result(check.id, check.kind, False,
                      f"warmup_steps is {mine.warmup_steps} of {mine.max_steps} steps "
                      f"({mine.warmup_steps / mine.max_steps:.0%}): the learning rate would "
                      f"still be climbing for more than {fraction:.0%} of the run")
    if mine.eval_interval > mine.max_steps:
        return Result(check.id, check.kind, False,
                      f"eval_interval is {mine.eval_interval} but the run has {mine.max_steps} "
                      "steps, so it would never evaluate and report no loss")
    return Result(check.id, check.kind, True,
                  f"{name} is registered: {mine.max_steps} steps, warmup {mine.warmup_steps}, "
                  f"evaluating every {mine.eval_interval}, tokens as in {base_name}",
                  {"preset": name, "max_steps": mine.max_steps})


def _kwargs_text(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())


# -- equivalent ------------------------------------------------------------------------------

def _make_inputs(specs: list[Any], generator: Any) -> list[Any]:
    """Random tensors for one trial: a list of ints is a normal-distributed float tensor of
    that shape; {shape = [...], ints = N} is token ids below N."""
    import torch

    tensors = []
    for spec in specs:
        if isinstance(spec, dict):
            tensors.append(torch.randint(0, int(spec["ints"]), tuple(spec["shape"]),
                                         generator=generator))
        else:
            tensors.append(torch.randn(*spec, generator=generator))
    return tensors


def _input_sets(inputs: list[Any]) -> list[list[Any]]:
    """`inputs` is one set of specs, or a list of sets (to try several shapes)."""
    first = inputs[0] if inputs else None
    if isinstance(first, list) and first and isinstance(first[0], (list, dict)):
        return list(inputs)
    return [list(inputs)]


def resolve_param(params: dict[str, Any], key: str) -> str:
    """`q.weight` also names the weight of a template's `q` slot filled with the `Linear` block,
    which keeps it one level down (`q.linear.weight`): the template route and the code route are
    judged by the same check. An exact name always wins; otherwise one inserted level is accepted
    when it is the only match."""
    if key in params or "." not in key:
        return key
    head, _, tail = key.partition(".")
    found = [name for name in params if name.startswith(head + ".") and name.endswith("." + tail)
             and name.count(".") == key.count(".") + 1]
    return found[0] if len(found) == 1 else key


@checker("equivalent")
def check_equivalent(ctx: Context, check: Check) -> Result:
    """The learner's module matches a naive reference on random inputs. Weights are copied
    from the module into the reference call by the lesson's `call` list: "input:0" is the
    first random input, "param:weight" the module's `weight`, anything else a constant."""
    import torch

    from nanoscope.reference import functional

    name = check.args["reference"]
    reference = getattr(functional, name, None)
    if reference is None:
        return Result(check.id, check.kind, False,
                      f"the lesson names reference {name!r}, which nanoscope.reference doesn't "
                      "have: this is a bug in the lesson, not in your code")
    try:
        cls = user_class(ctx, check)
        torch.manual_seed(0)
        module = cls(**constructor_args(cls, check.args.get("args"))).eval()
    except (AttributeError, FileNotFoundError, ValueError, TypeError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    if check.args.get("randomize"):  # norm scales start at 1: make every weight matter
        scrambler = torch.Generator().manual_seed(99)
        with torch.no_grad():
            for p in module.parameters():
                noise = torch.randn(p.shape, generator=scrambler) * 0.5
                p.copy_(noise + (1 if p.ndim == 1 else 0))
    params = dict(module.named_parameters()) | dict(module.named_buffers())
    tolerance = float(check.args.get("tolerance", 1e-5))
    generator = torch.Generator().manual_seed(1234)
    worst, shapes = 0.0, []
    for specs in _input_sets(check.args["inputs"]):
        for _ in range(int(check.args.get("trials", 3))):
            inputs = _make_inputs(specs, generator)
            shape = [list(t.shape) for t in inputs]
            try:
                call = []
                for item in check.args["call"]:
                    if isinstance(item, str) and item.startswith("input:"):
                        call.append(inputs[int(item[6:])])
                    elif isinstance(item, str) and item.startswith("param:"):
                        key = resolve_param(params, item[6:])
                        if key not in params:
                            return Result(
                                check.id, check.kind, False,
                                f"{cls.__name__} has no parameter named {key!r} (it has: "
                                f"{', '.join(params) or 'none'}): the lesson expects that name")
                        call.append(params[key].detach())
                    else:
                        call.append(item)
                with torch.no_grad():
                    got = module(*inputs)
                    got = got[0] if isinstance(got, tuple) else got
                    want = reference(*call)
            except Exception as exc:
                return Result(check.id, check.kind, False,
                              f"running on inputs of shape {shape} raised "
                              f"{type(exc).__name__}: {exc}", {"input_shapes": shape})
            if got.shape != want.shape:
                return Result(check.id, check.kind, False,
                              f"output shape {list(got.shape)} but the reference gives "
                              f"{list(want.shape)} for inputs {shape}", {"input_shapes": shape})
            diff = float((got.float() - want.float()).abs().max())
            worst = max(worst, diff)
            shapes.append(shape)
            if not diff <= tolerance:
                return Result(
                    check.id, check.kind, False,
                    f"output differs from the reference: max abs diff {diff:.3g} on inputs of "
                    f"shape {shape} (tolerance {tolerance:g})",
                    {"max_abs_diff": diff, "tolerance": tolerance, "input_shapes": shape,
                     "reference": name})
    return Result(check.id, check.kind, True,
                  f"matches the reference {name}: max abs diff {worst:.3g} over {len(shapes)} "
                  f"random inputs (tolerance {tolerance:g})",
                  {"max_abs_diff": worst, "tolerance": tolerance, "trials": len(shapes),
                   "reference": name})


# -- function --------------------------------------------------------------------------------

def _shown(value: Any) -> Any:
    if isinstance(value, list):
        return [_shown(x) for x in value]
    return round(value, 4) if isinstance(value, float) else value


@checker("function")
def check_function(ctx: Context, check: Check) -> Result:
    """A plain function in the learner's file gives the same numbers as a naive reference on
    the lesson's fixed cases. Each case is the list of arguments; the result is a number or a
    tuple of numbers, and `labels` names the parts of a tuple in the failure message."""
    import copy
    import math

    from nanoscope.reference import functional

    name = check.args["name"]
    reference = getattr(functional, check.args["reference"], None)
    if reference is None:
        return Result(check.id, check.kind, False,
                      f"the lesson names reference {check.args['reference']!r}, which "
                      "nanoscope.reference doesn't have: this is a bug in the lesson")
    try:
        function = getattr(load_user_module(ctx), name, None)
    except (FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    if not callable(function):
        return Result(check.id, check.kind, False,
                      f"function {name} is not defined in {ctx.user_file.name}")
    tolerance = float(check.args.get("tolerance", 1e-6))
    labels = check.args.get("labels", [])
    worst = 0.0

    def numbers(value: Any) -> list[float]:
        return [float(x) for x in value] if isinstance(value, (tuple, list)) else [float(value)]

    for i, case in enumerate(check.args["cases"], 1):
        shown = ", ".join(str(_shown(a)) for a in case)
        want = numbers(reference(*copy.deepcopy(case)))
        try:
            got = numbers(function(*copy.deepcopy(case)))
        except Exception as exc:
            return Result(check.id, check.kind, False,
                          f"{name}({shown}) raised {type(exc).__name__}: {exc}")
        if len(got) != len(want):
            return Result(check.id, check.kind, False,
                          f"{name}({shown}) returned {len(got)} value(s); the lesson expects "
                          f"{len(want)}" + (f" ({', '.join(labels)})" if labels else ""))
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            diff = abs(g - w) if math.isfinite(g) else math.inf
            worst = max(worst, diff)
            if not diff <= tolerance:
                part = labels[k] if k < len(labels) else f"value {k + 1}"
                return Result(
                    check.id, check.kind, False,
                    f"case {i}, {name}({shown}): {part} is {g:.6g} but the reference gives "
                    f"{w:.6g} (off by {diff:.3g}; tolerance {tolerance:g})",
                    {"case": i, "part": part, "got": g, "want": w, "tolerance": tolerance})
    return Result(check.id, check.kind, True,
                  f"{name} matches the reference {check.args['reference']} on "
                  f"{len(check.args['cases'])} cases: largest difference {worst:.3g}",
                  {"max_abs_diff": worst, "cases": len(check.args["cases"])})


# -- forbid ----------------------------------------------------------------------------------

ALIASES = {"F.": "torch.nn.functional.", "nn.": "torch.nn."}


def _canonical(name: str) -> str:
    for short, full in ALIASES.items():
        if name.startswith(short):
            return full + name[len(short):]
    return name


def forbidden_uses(source: str, forbidden: list[str]) -> list[tuple[int, str]]:
    """(line, name) for every use of a forbidden name, found in the AST with imports resolved,
    so `import torch.nn.functional as F` then `F.scaled_dot_product_attention` is caught."""
    import ast

    tree = ast.parse(source)
    imports: dict[str, str] = {}
    hits: set[tuple[int, str]] = set()
    wanted = [_canonical(f) for f in forbidden]

    def flagged(qualified: str) -> str | None:
        for entry in wanted:
            if qualified == entry or qualified.startswith(entry + "."):
                return entry
            if "." not in entry and qualified.rsplit(".", 1)[-1] == entry:
                return entry
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                imports[alias.asname or alias.name] = full
                if (hit := flagged(full)):
                    hits.add((node.lineno, hit))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0])
                if (hit := flagged(alias.name)):
                    hits.add((node.lineno, hit))
    for node in ast.walk(tree):
        chain: list[str] = []
        cur: ast.expr = node if isinstance(node, (ast.Name, ast.Attribute)) else ast.Constant(0)
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and isinstance(node, (ast.Name, ast.Attribute)):
            qualified = ".".join([imports.get(cur.id, cur.id), *reversed(chain)])
            if (hit := flagged(qualified)):
                hits.add((node.lineno, hit))
    return sorted(hits)


@checker("forbid")
def check_forbid(ctx: Context, check: Check) -> Result:
    """The learner's file does not use the shortcuts the lesson forbids (`forbid` in
    lesson.toml), nor blocks they have not unlocked."""
    names = ctx.lesson.forbid
    file = ctx.user_file
    if not file.exists():
        return Result(check.id, check.kind, False,
                      f"{file} does not exist: run `nanoscope learn start {ctx.lesson.id}` first")
    try:
        source = file.read_text(encoding="utf-8")
        found = forbidden_uses(source, names)
    except SyntaxError as exc:
        return Result(check.id, check.kind, False, f"{file.name} line {exc.lineno}: {exc.msg}")
    from nanoscope.learn import gating

    locked = gating.scan(file, ctx.owner)
    if locked:
        listing = "; ".join(f"line {u.line}: {u.id} (locked until {u.lesson})" for u in locked)
        return Result(check.id, check.kind, False,
                      f"{file.name} uses blocks you have not unlocked yet: {listing}",
                      {"locked": [{"line": u.line, "id": u.id, "lesson": u.lesson}
                                  for u in locked]})
    if found:
        listing = "; ".join(f"line {line}: {name}" for line, name in found)
        return Result(check.id, check.kind, False,
                      f"{file.name} uses what this lesson asks you to build yourself: {listing}",
                      {"uses": [{"line": line, "name": name} for line, name in found]})
    return Result(check.id, check.kind, True,
                  f"{file.name} avoids the shortcuts this lesson forbids "
                  f"({', '.join(names) or 'none listed'})",
                  {"forbidden": names})


# -- trains ----------------------------------------------------------------------------------

LOWER_IS_BETTER = ("val_bpb", "val_loss")


def lesson_preset(ctx: Context) -> str:
    variant = ctx.lesson.compute.get(ctx.variant)
    if variant is None or variant.preset is None:
        raise ValueError(f"the {ctx.variant} variant of {ctx.lesson.id} has no preset "
                         "(a budget variant is run as a study, not a single run)")
    return variant.preset


@checker("trains")
def check_trains(ctx: Context, check: Check) -> Result:
    """`run()` of the learner's model on the lesson's preset reaches the metric target."""
    from nanoscope.run import RunResult, run

    metric = check.args["metric"]
    if metric not in LOWER_IS_BETTER:
        return Result(check.id, check.kind, False,
                      f"the lesson asks for metric {metric!r}; this check knows "
                      f"{', '.join(LOWER_IS_BETTER)}")
    threshold = float(check.args["threshold"])
    try:
        cls = user_class(ctx, check)
        preset = lesson_preset(ctx)
    except (AttributeError, FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    seeds = int(check.args.get("seeds", 1))
    values, refs = [], []
    for seed in range(seeds):
        result = run(cls, preset, seed=seed, device="cpu" if ctx.variant == "cpu" else None,
                     output_dir=ctx.runs_dir / f"seed-{seed}", progress=False,
                     **check.args.get("kwargs", {}))
        assert isinstance(result, RunResult)
        values.append(result.val_losses[-1][1] if metric == "val_loss" else result.val_bpb[-1][1])
        refs.append(result.ref)
    mean = sum(values) / len(values)
    steps = result.final_step
    evidence = {"metric": metric, "values": values, "mean": mean, "threshold": threshold,
                "preset": preset, "runs": refs}
    shown = f"{metric} {mean:.3f}" + (f" (mean of {seeds} seeds)" if seeds > 1 else "")
    if mean <= threshold:
        return Result(check.id, check.kind, True,
                      f"{shown} after {steps} steps on {preset}, at or below the target "
                      f"{threshold:g}", evidence)
    return Result(check.id, check.kind, False,
                  f"{shown} after {steps} steps on {preset} is above the target {threshold:g}: "
                  "the model is not learning enough yet", evidence)


# -- verdict ---------------------------------------------------------------------------------

SHIPPED = ("bigram", "gpt2", "modern")
VERDICTS = ("better", "worse", "within noise")
PAIRINGS = ("paired", "unpaired")
UNPAIRED_OFFSET = 1000


def find_model(ctx: Context, name: str) -> type:
    """A class from the learner's file, else a shipped model by name (bigram, gpt2, modern)."""
    try:
        return user_class(ctx, Check("-", "-", {"class": name}))
    except (AttributeError, FileNotFoundError):
        if name.lower() in SHIPPED:
            from nanoscope import models

            return {"bigram": models.Bigram, "gpt2": models.GPT2,
                    "modern": models.Modern}[name.lower()]
        raise


def find_optimizer(ctx: Context, name: str | None) -> Any:
    """A verdict side's optimizer factory: "adamw" (the default) is None, "muon" is the shipped
    one, anything else is a function of that name in the learner's file."""
    if name is None or name == "adamw":
        return None
    if name == "muon":
        from nanoscope.optim import muon

        return muon
    function = getattr(load_user_module(ctx), name, None)
    if not callable(function):
        raise AttributeError(f"function {name} is not defined in {ctx.user_file.name}")
    return function


@checker("verdict")
def check_verdict(ctx: Context, check: Check) -> Result:
    """Train A and B on the lesson's preset with n seeds, `compare()` them, and require the
    verdict the lesson asks for ("better", "worse" or "within noise" for A against B)."""
    from nanoscope.compare import compare
    from nanoscope.run import run

    expect = check.args["expect"]
    if expect not in VERDICTS:
        return Result(check.id, check.kind, False,
                      f"the lesson expects verdict {expect!r}; the verdicts are "
                      f"{', '.join(VERDICTS)}: this is a bug in the lesson")
    seeds = int(check.args.get("gpu_seeds", check.args.get("seeds", 3))
                if ctx.variant == "gpu" else check.args.get("seeds", 3))
    metric = check.args.get("metric", "val_bpb")
    try:
        preset = lesson_preset(ctx)
        classes = {side: find_model(ctx, check.args[side]) for side in ("a", "b")}
        optimizers = {side: find_optimizer(ctx, check.args.get(f"{side}_optimizer"))
                      for side in ("a", "b")}
    except (AttributeError, FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    pairing = check.args.get("pairing", "paired")
    if pairing not in PAIRINGS:
        return Result(check.id, check.kind, False,
                      f"the lesson asks for pairing {pairing!r}; it is one of "
                      f"{', '.join(PAIRINGS)}: this is a bug in the lesson")
    # Unpaired: B gets other seeds, so no seed of A has a partner and compare() cannot pair.
    # They go in their own folder, so a paired check in the same lesson reuses A's finished
    # runs and neither check sees the other's seeds.
    unpaired = pairing == "unpaired"
    b_folder = "b-unpaired" if unpaired else "b"
    for side, cls in classes.items():
        offset = UNPAIRED_OFFSET if side == "b" and unpaired else 0
        folder = b_folder if side == "b" else "a"
        for seed in range(seeds):
            run(cls, preset, seed=offset + seed, device="cpu" if ctx.variant == "cpu" else None,
                output_dir=ctx.runs_dir / folder / f"seed-{offset + seed}", progress=False,
                **({"optimizer": optimizers[side]} if optimizers[side] else {}),
                **check.args.get(f"{side}_kwargs", {}))
    comparison = compare(ctx.runs_dir / "a", ctx.runs_dir / b_folder, metric=metric)
    row = comparison.rows[0]
    delta = row["delta"]
    label = f"{check.args['a']} vs {check.args['b']}"
    if delta is not None and delta["ci95_low"] is None and delta["mean"] == 0 \
            and "no observed seed variation" in delta.get("status", ""):
        # identical on every seed: the difference is exactly zero, there is just no spread
        # to build an interval from. That is "within noise" in the plainest sense.
        evidence = {"verdict": "within noise", "expected": expect, "metric": metric,
                    "seeds": seeds, "delta": 0.0, "ci95": [0.0, 0.0], "degenerate": True}
        said = (f"{label}: identical on every seed (difference exactly 0, so there is no "
                "seed variation to build an interval from), which counts as within noise")
        if expect == "within noise":
            return Result(check.id, check.kind, True, f"{said}, as the lesson expects", evidence)
        return Result(check.id, check.kind, False,
                      f"{said}; the lesson expects '{expect}'. Are the two models really "
                      "different?", evidence)
    if delta is None or delta["ci95_low"] is None:
        why = ("with fewer than 3 seeds there is no confidence interval; the lesson needs 3"
               if seeds < 3 else (delta or {}).get("status", "no interval"))
        return Result(check.id, check.kind, False, f"{label}: {why}", {"seeds": seeds})
    interval = f"{delta['mean']:+.3f} [{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}]"
    evidence = {"verdict": row["verdict"], "expected": expect, "metric": metric, "seeds": seeds,
                "delta": delta["mean"], "ci95": [delta["ci95_low"], delta["ci95_high"]],
                "paired": delta["paired"]}
    how = "" if delta["paired"] else ", unpaired"
    said = f"{label}: {row['verdict']} ({metric} difference {interval}, {seeds} seeds{how})"
    if row["verdict"] == expect:
        return Result(check.id, check.kind, True, f"{said}, as the lesson expects", evidence)
    return Result(check.id, check.kind, False,
                  f"{said}; the lesson expects '{expect}'. Look at what differs between the two "
                  "models, and at the interval: is zero inside it?", evidence)


# -- matches ---------------------------------------------------------------------------------

QUANTITIES = ("params", "flops")


def _size_of(cls: type, kwargs: dict[str, Any], quantity: str, context_length: int) -> int:
    """Non-embedding parameters, or training FLOPs per token, of a class built on the meta
    device (no memory): the same numbers a Study matches on."""
    import inspect

    from nanoscope.sizing import build_on_meta, count_params, flops_per_token

    accepted = inspect.signature(cls).parameters
    sized = {"vocab_size": 4096, "context_length": context_length}
    given = {k: v for k, v in sized.items() if k in accepted} | kwargs
    model = build_on_meta(cls, **constructor_args(cls, given))
    return count_params(model)[1] if quantity == "params" else flops_per_token(
        model, context_length)


@checker("matches")
def check_matches(ctx: Context, check: Check) -> Result:
    """The learner's class is the same size as another model: non-embedding parameters or
    training FLOPs per token within `tolerance` (default 3%). Sizing a comparison fairly is
    most of an honest ablation."""
    quantity = check.args["quantity"]
    if quantity not in QUANTITIES:
        return Result(check.id, check.kind, False,
                      f"the lesson asks to match {quantity!r}; this check knows "
                      f"{', '.join(QUANTITIES)}: this is a bug in the lesson")
    tolerance = float(check.args.get("tolerance", 0.03))
    context_length = int(check.args.get("context_length", 256))
    try:
        mine = user_class(ctx, check)
        other = find_model(ctx, check.args["against"])
        a = _size_of(mine, check.args.get("kwargs", {}), quantity, context_length)
        b = _size_of(other, check.args.get("against_kwargs", {}), quantity, context_length)
    except (AttributeError, FileNotFoundError, ValueError, TypeError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    what = "non-embedding parameters" if quantity == "params" else "training FLOPs per token"
    gap = (a - b) / b
    evidence = {"quantity": quantity, "mine": a, "other": b, "gap": gap, "tolerance": tolerance}
    said = (f"{mine.__name__} has {a:,} {what}, {other.__name__} has {b:,} "
            f"({gap:+.1%}; the lesson allows {tolerance:.0%})")
    if abs(gap) <= tolerance:
        return Result(check.id, check.kind, True, f"{said}: matched", evidence)
    side = "more" if gap > 0 else "fewer"
    return Result(check.id, check.kind, False,
                  f"{said}: {mine.__name__} has {side}, so any difference in the results could "
                  "be size, not the idea. Adjust a width until they match", evidence)


# -- ablation --------------------------------------------------------------------------------

@checker("ablation")
def check_ablation(ctx: Context, check: Check) -> Result:
    """The capstone. Read the learner's own finished Study (the full model, a baseline and one
    variant per `drops` entry, each the full model with one component removed), compare the
    variants with seed-level intervals, then hold the learner's file against what the study
    found. The file must define

        GAP = (low, high)          the interval for full - baseline, in the lesson's metric
        NULL_COMPONENT = "name"    a drop whose removal changed nothing detectable (or None)

    GAP must contain the measured difference and not be far wider than the study's own
    interval (the rule for a prediction). NULL_COMPONENT must be a drop whose verdict is
    'within noise'. Naming a component that matters, or claiming none is null when one is,
    fails with the numbers. The check trains nothing: the evidence is the learner's study."""
    from nanoscope.compare import compare
    from nanoscope.statistics import score_prediction
    from nanoscope.studyfiles import finished_runs, study_dir

    seeds = int(check.args.get("gpu_seeds", check.args.get("seeds", 3))
                if ctx.variant == "gpu" else check.args.get("seeds", 3))
    metric = check.args.get("metric", "val_bpb")
    study, full, base = check.args["study"], check.args["full"], check.args["baseline"]
    drops: list[str] = check.args["drops"]
    try:
        preset = lesson_preset(ctx)
        module = load_user_module(ctx)
    except (AttributeError, FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    gap, null = getattr(module, "GAP", None), getattr(module, "NULL_COMPONENT", "<unset>")
    problems = []
    if not (isinstance(gap, (tuple, list)) and len(gap) == 2
            and all(isinstance(x, (int, float)) for x in gap) and gap[0] < gap[1]):
        problems.append("GAP is not set: write GAP = (low, high), your interval for the "
                        f"difference between {full} and {base} (negative is better for "
                        "val_bpb)")
    if null == "<unset>" or not (null is None or null in drops):
        problems.append(f"NULL_COMPONENT must be one of {', '.join(drops)}, or None if you "
                        "think every component matters")
    if problems:  # say what is missing before anyone waits on a study
        return Result(check.id, check.kind, False, "; ".join(problems))
    assert isinstance(gap, (tuple, list))
    low, high = float(gap[0]), float(gap[1])

    folder = study_dir(study)
    wanted = [full, base, *drops]
    missing = [v for v in wanted if not (folder / v).exists()]
    if missing:
        return Result(check.id, check.kind, False,
                      f"the study {study!r} has no variant {', '.join(missing)} under "
                      f"{folder}: run the study from the lesson first (it needs {len(wanted)} "
                      f"variants: {', '.join(wanted)})")
    done = {v: finished_runs(folder / v) for v in wanted}
    short = [f"{v} ({len(r)})" for v, r in done.items() if len(r) < seeds]
    if short:
        return Result(check.id, check.kind, False,
                      f"every variant needs {seeds} finished seeds; still short: "
                      f"{', '.join(short)}. Let the study finish")
    seed_sets = {tuple(sorted(r.seed for r in runs)) for runs in done.values()}
    if len(seed_sets) > 1:
        listing = "; ".join(f"{v}: {sorted(r.seed for r in runs)}" for v, runs in done.items())
        return Result(check.id, check.kind, False,
                      f"the variants were not run on the same seeds, so they cannot be paired: "
                      f"{listing}")
    other = sorted({r.preset["name"] for runs in done.values() for r in runs} - {preset})
    if other:
        return Result(check.id, check.kind, False,
                      f"the study ran on {', '.join(other)}, but this lesson is on {preset}: "
                      "use the preset from the lesson")

    against = compare(folder / full, folder / base, metric=metric).rows[0]
    delta = against["delta"]
    if delta is None or delta["ci95_low"] is None:
        return Result(check.id, check.kind, False,
                      f"there is no interval for {full} against {base}: the lesson needs "
                      "3 seeds")
    ci = (delta["ci95_low"], delta["ci95_high"])
    score = score_prediction(delta["mean"], ci, low=low, high=high)
    measured = (f"{full} vs {base}: {delta['mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {metric}")
    verdicts: dict[str, str] = {}
    table = []
    for name in drops:
        row = compare(folder / name, folder / full, metric=metric).rows[0]
        dd = row["delta"]
        identical = (dd is not None and dd["ci95_low"] is None and dd["mean"] == 0
                     and "no observed seed variation" in dd.get("status", ""))
        verdicts[name] = "within noise" if identical else row["verdict"]
        shown = ("no interval" if dd is None or dd["ci95_low"] is None else
                 f"{dd['mean']:+.3f} [{dd['ci95_low']:+.3f}, {dd['ci95_high']:+.3f}]")
        table.append(f"{name}: {verdicts[name]} ({shown})")
    evidence = {"delta": delta["mean"], "ci95": list(ci), "gap": [low, high],
                "verdict": against["verdict"], "drops": verdicts, "seeds": seeds,
                "null_component": null, "score": score}
    fails = []
    if not score["passed"]:
        if score.get("hit") is False:
            fails.append(f"your GAP ({low:+g}, {high:+g}) does not contain the measured "
                         f"difference")
        if score.get("sharp") is False:
            fails.append("your GAP is far wider than the study's own interval, so it says "
                         "almost nothing")
    nulls = [n for n, v in verdicts.items() if v == "within noise"]
    if null is None and nulls:
        fails.append(f"you said every component matters, but removing {', '.join(nulls)} was "
                     "within noise")
    elif null is not None and verdicts[null] != "within noise":
        fails.append(f"{null} is not null: removing it was {verdicts[null]}")
    detail = "; ".join(table)
    if fails:
        return Result(check.id, check.kind, False,
                      f"{'; '.join(fails)}. Measured {measured}. Removing each component, "
                      f"against {full}: {detail}", evidence)
    return Result(check.id, check.kind, True,
                  f"your gap holds ({measured}) and {null or 'no component'} is null as you "
                  f"said. Removing each component, against {full}: {detail}", evidence)


# -- ladder ----------------------------------------------------------------------------------

LADDER_X = {"params": ("n_non_embedding_params", "non-embedding parameters"),
            "flops": ("flops_per_token", "training FLOPs per token")}


@checker("ladder")
def check_ladder(ctx: Context, check: Check) -> Result:
    """A scaling ladder from the learner's own finished Study: fit a power law to the `rungs`
    (the metric against the model's size), predict the held-out larger model, and hold the
    learner's file against what happened. The file must define

        PREDICTION = (low, high)    the metric you expect for the held-out model

    written before the held-out model was trained. It must contain the measured mean, and it
    must not be far wider than the fit's own 95% prediction interval. The message always reports
    the fitted exponent with its interval and where the held-out model really landed."""
    from nanoscope.scaling import fit_power_law
    from nanoscope.studyfiles import finished_runs, study_dir

    seeds = int(check.args.get("gpu_seeds", check.args.get("seeds", 3))
                if ctx.variant == "gpu" else check.args.get("seeds", 3))
    metric = check.args.get("metric", "val_bpb")
    key, what = LADDER_X[check.args.get("x", "params")]
    study, rungs, holdout = check.args["study"], list(check.args["rungs"]), check.args["holdout"]
    try:
        preset = lesson_preset(ctx)
        module = load_user_module(ctx)
    except (AttributeError, FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    guess = getattr(module, "PREDICTION", None)
    if not (isinstance(guess, (tuple, list)) and len(guess) == 2
            and all(isinstance(v, (int, float)) for v in guess) and guess[0] < guess[1]):
        return Result(check.id, check.kind, False,
                      f"PREDICTION is not set: write PREDICTION = (low, high), the {metric} you "
                      f"expect for {holdout}, from a fit to {', '.join(rungs)} and before you "
                      f"train {holdout}")
    low, high = float(guess[0]), float(guess[1])
    folder = study_dir(study)
    missing = [v for v in [*rungs, holdout] if not (folder / v).exists()]
    if missing:
        return Result(check.id, check.kind, False,
                      f"the study {study!r} has no variant {', '.join(missing)} under {folder}: "
                      "run the study from the lesson first")
    done = {v: finished_runs(folder / v) for v in [*rungs, holdout]}
    short = [f"{v} ({len(r)})" for v, r in done.items() if len(r) < seeds]
    if short:
        return Result(check.id, check.kind, False,
                      f"every variant needs {seeds} finished seeds; still short: "
                      f"{', '.join(short)}. Let the study finish")
    other = sorted({r.preset["name"] for runs in done.values() for r in runs} - {preset})
    if other:
        return Result(check.id, check.kind, False,
                      f"the study ran on {', '.join(other)}, but this lesson is on {preset}")
    points = [(r.config["stats"][key], r.final(metric)) for v in rungs for r in done[v]]
    fit = fit_power_law(points)
    held = done[holdout]
    x_held = held[0].config["stats"][key]
    got = sum(r.final(metric) for r in held) / len(held)
    mid, plo, phi = fit.predict(x_held)
    ci = fit.exponent_ci95
    exponent = (f"exponent {fit.exponent:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                + f" over {len(rungs)} sizes ({fit.n} runs, {what})")
    own = (f"; the fit alone predicts {mid:.3f}" + (f" [{plo:.3f}, {phi:.3f}]"
                                                    if plo is not None else ""))
    landed = f"{holdout} measured {got:.3f} {metric}"
    evidence = {"exponent": fit.exponent, "exponent_ci95": list(ci) if ci else None,
                "predicted": mid, "prediction_interval": [plo, phi], "measured": got,
                "prediction": [low, high], "holdout": holdout, "x_holdout": x_held}
    problems = []
    if not low <= got <= high:
        problems.append(f"your PREDICTION ({low:.3f}, {high:.3f}) does not contain it")
    if plo is not None and phi is not None and (high - low) > 10 * (phi - plo):
        problems.append("your PREDICTION is far wider than the fit's own prediction interval, "
                        "so it says almost nothing")
    if problems:
        return Result(check.id, check.kind, False,
                      f"{landed}: {'; '.join(problems)}. The fit: {exponent}{own}", evidence)
    inside = plo is not None and phi is not None and plo <= got <= phi
    verdict = ("inside the fit's own interval: the power law extrapolated one step" if inside
               else "outside the fit's own interval: the trend did not simply continue, "
                    "which is worth understanding")
    return Result(check.id, check.kind, True,
                  f"{landed}, inside your PREDICTION ({low:.3f}, {high:.3f}). The fit: "
                  f"{exponent}{own}. It landed {verdict}", evidence)


# -- predicted -------------------------------------------------------------------------------

PREDICTION_FILE = "prediction.toml"
PREDICTION_FIELDS = {"quantity", "verdict", "low", "high", "note"}


def prediction_path(lesson: LessonSpec) -> Path:
    from nanoscope.learn.cli import workspace_lesson_dir

    return workspace_lesson_dir(lesson) / PREDICTION_FILE


def read_prediction(file: Path) -> dict[str, Any]:
    """The learner's prediction.toml, checked for the fields a prediction may have."""
    from nanoscope.learn.loader import tomllib

    doc = tomllib.loads(file.read_text(encoding="utf-8"))
    unknown = set(doc) - PREDICTION_FIELDS
    if unknown:
        raise ValueError(f"{file.name}: unknown field {', '.join(sorted(unknown))} "
                         f"(allowed: {', '.join(sorted(PREDICTION_FIELDS))})")
    low, high = doc.get("low"), doc.get("high")
    if (low is None) != (high is None):
        raise ValueError(f"{file.name}: give both low and high, or neither")
    if low is not None and not low <= high:
        raise ValueError(f"{file.name}: low must not be above high")
    if "verdict" in doc and doc["verdict"] not in VERDICTS:
        raise ValueError(f"{file.name}: verdict must be one of {', '.join(VERDICTS)}")
    if low is None and "verdict" not in doc:
        raise ValueError(f"{file.name}: predict a verdict, an interval (low and high), or both")
    return doc


class PredictionTooLate(ValueError):
    """The lesson was already checked: a prediction only counts if it comes first."""


class PredictionMissing(ValueError):
    """Nothing was said and there is no prediction.toml to commit."""


def commit_prediction(lesson: LessonSpec, given: dict[str, Any], owner: str = "local"
                      ) -> tuple[Path, str]:
    """Write `prediction.toml` from `given` (or commit the file already there when `given` is
    empty), validate it, and record the time. Raises `PredictionTooLate` once the lesson has
    been checked, `PredictionMissing` for nothing to commit, `ValueError` for a bad prediction.
    Returns (file, timestamp)."""
    import hashlib

    import tomli_w

    first = progress.entry(lesson.id, owner)["first_checked_at"]
    if first is not None:
        raise PredictionTooLate(f"{lesson.id} was first checked at {first}: a prediction only "
                                "counts if it is recorded before the experiment runs, so it is "
                                "too late for this lesson")
    file = prediction_path(lesson)
    if given:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(tomli_w.dumps(given), encoding="utf-8")
    elif not file.exists():
        raise PredictionMissing("say what you expect, e.g. --verdict better --low -0.2 "
                                f"--high -0.05\n(or write {file} yourself)")
    read_prediction(file)
    stamp = progress.record_prediction(lesson.id, file, hashlib.sha256(
        file.read_bytes()).hexdigest(), owner)
    return file, stamp


@checker("predicted")
def check_predicted(ctx: Context, check: Check) -> Result:
    """The learner committed to a prediction *before* the first check, and it matches what the
    quantity's check found. `quantity` is the id of an earlier verdict check."""
    import hashlib

    from nanoscope.statistics import score_prediction

    file = prediction_path(ctx.lesson)
    recorded = progress.prediction(ctx.lesson.id, ctx.owner)
    if recorded is None or not file.exists():
        return Result(check.id, check.kind, False,
                      "no prediction was recorded: write it with `nanoscope learn predict "
                      f"{ctx.lesson.id}` before you run the experiment")
    first = progress.entry(ctx.lesson.id, ctx.owner)["first_checked_at"]
    if first is not None and recorded["at"] >= first:
        return Result(check.id, check.kind, False,
                      f"the prediction was recorded at {recorded['at']}, after the experiment "
                      f"first ran ({first}): a prediction only counts if it comes first",
                      {"predicted_at": recorded["at"], "first_checked_at": first})
    if hashlib.sha256(file.read_bytes()).hexdigest() != recorded["sha256"]:
        return Result(check.id, check.kind, False,
                      f"{file.name} was edited after it was recorded at {recorded['at']}: "
                      "record it again with `nanoscope learn predict` (only before the "
                      "experiment runs)")
    quantity = check.args["quantity"]
    evidence = ctx.shared.get("evidence", {}).get(quantity)
    if not evidence or "delta" not in evidence:
        return Result(check.id, check.kind, False,
                      f"there is nothing to score against: the {quantity!r} check must run and "
                      "produce a difference first")
    try:
        prediction = read_prediction(file)
    except ValueError as exc:
        return Result(check.id, check.kind, False, str(exc))
    score = score_prediction(
        evidence["delta"], tuple(evidence["ci95"]), low=prediction.get("low"),
        high=prediction.get("high"), verdict=prediction.get("verdict"),
        actual_verdict=evidence.get("verdict"))
    found = (f"the run found {evidence['delta']:+.3f} "
             f"[{evidence['ci95'][0]:+.3f}, {evidence['ci95'][1]:+.3f}], verdict "
             f"'{evidence.get('verdict')}'")
    if score["passed"]:
        return Result(check.id, check.kind, True,
                      f"your prediction was recorded first and holds: {found}", score)
    if not check.args.get("must_hit", True) and score.get("sharp") is not False:
        # a practice prediction: it must come first and say something, but a miss is reported,
        # not graded, because a wrong guess cannot be taken back and redone
        missed = "" if score["passed"] else "; it did not hold, and that is the interesting part"
        return Result(check.id, check.kind, True,
                      f"your prediction was recorded first{missed}: {found}", score)
    misses = []
    if score.get("hit") is False:
        misses.append(f"the observed difference is outside your interval "
                      f"[{prediction['low']:+g}, {prediction['high']:+g}]")
    if score.get("sharp") is False:
        misses.append("your interval is far wider than the run's own uncertainty, so it "
                      "predicts too little")
    if score.get("verdict_right") is False:
        misses.append(f"you predicted '{prediction['verdict']}'")
    return Result(check.id, check.kind, False,
                  f"{found}; {'; '.join(misses)}. Being wrong is information: say why before "
                  "you try again", score)


# -- reproduces ------------------------------------------------------------------------------

@checker("reproduces")
def check_reproduces(ctx: Context, check: Check) -> Result:
    """The learner's model trained on the lesson's preset lands where a shipped baseline's
    seeds do: inside the 95% interval for a new result, built from the baseline's seed
    spread (a single run is not expected to hit the baseline mean exactly)."""
    from nanoscope.compare import _resolve
    from nanoscope.run import RunResult, run
    from nanoscope.statistics import reproduction_interval

    metric = check.args.get("metric", "val_bpb")
    try:
        cls = user_class(ctx, check) if check.args.get("class") or ctx.shared.get(
            "class_name") else find_model(ctx, check.args["baseline"])
        preset = lesson_preset(ctx)
        base = _resolve(check.args["baseline"], preset)
    except (AttributeError, FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    if base.config["preset"]["name"] != preset:
        return Result(check.id, check.kind, False,
                      f"the baseline {check.args['baseline']!r} was trained on "
                      f"{base.config['preset']['name']!r}, not {preset!r}: this is a bug in the "
                      "lesson")
    values = [r.final(metric) for r in base.runs]
    interval = reproduction_interval(values)
    if interval is None:
        return Result(check.id, check.kind, False,
                      f"the baseline has only {len(values)} seeds, too few for an interval: "
                      "this is a bug in the lesson")
    result = run(cls, preset, seed=0, device="cpu" if ctx.variant == "cpu" else None,
                 output_dir=ctx.runs_dir / "seed-0", progress=False,
                 **check.args.get("kwargs", {}))
    assert isinstance(result, RunResult)
    got = result.val_losses[-1][1] if metric == "val_loss" else result.val_bpb[-1][1]
    low, high = interval
    evidence = {"metric": metric, "value": got, "baseline": check.args["baseline"],
                "interval": [low, high], "baseline_values": values}
    if low <= got <= high:
        return Result(check.id, check.kind, True,
                      f"{metric} {got:.3f} is inside the range {low:.3f} to {high:.3f} that "
                      f"the {check.args['baseline']} baseline's {len(values)} seeds give: "
                      "you reproduced it", evidence)
    side = "better than" if got < low else "worse than"
    return Result(check.id, check.kind, False,
                  f"{metric} {got:.3f} is outside {low:.3f} to {high:.3f}, {side} the "
                  f"{check.args['baseline']} baseline's seeds: check the model and the settings "
                  "against the lesson", evidence)
