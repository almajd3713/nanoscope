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
from nanoscope.learn import progress
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
    except Exception as exc:  # the learner's code, or ours: say which line, never a traceback
        frames = traceback.extract_tb(exc.__traceback__)
        where = f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
        result = Result(check.id, check.kind, False,
                        f"crashed: {type(exc).__name__}: {exc}{where}")
    result.seconds = time.perf_counter() - started
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
    progress.mark(lesson.id, "passed" if passed else "failed", check_id=cid, owner=owner)
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
        spec.loader.exec_module(module)
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

    args = dict(given or {})
    for name, param in inspect.signature(cls).parameters.items():
        if name in args or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
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
                        key = item[6:]
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
    if found:
        listing = "; ".join(f"line {line}: {name}" for line, name in found)
        return Result(check.id, check.kind, False,
                      f"{file.name} uses what this lesson asks you to build yourself: {listing}",
                      {"uses": [{"line": line, "name": name} for line, name in found]})
    return Result(check.id, check.kind, True,
                  f"{file.name} avoids {', '.join(names) or 'nothing in particular'}",
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
    seeds = int(check.args.get("seeds", 3))
    metric = check.args.get("metric", "val_bpb")
    try:
        preset = lesson_preset(ctx)
        classes = {side: find_model(ctx, check.args[side]) for side in ("a", "b")}
    except (AttributeError, FileNotFoundError, ValueError) as exc:
        return Result(check.id, check.kind, False, str(exc))
    for side, cls in classes.items():
        for seed in range(seeds):
            run(cls, preset, seed=seed, device="cpu" if ctx.variant == "cpu" else None,
                output_dir=ctx.runs_dir / side / f"seed-{seed}", progress=False,
                **check.args.get(f"{side}_kwargs", {}))
    comparison = compare(ctx.runs_dir / "a", ctx.runs_dir / "b", metric=metric)
    row = comparison.rows[0]
    delta = row["delta"]
    label = f"{check.args['a']} vs {check.args['b']}"
    if delta is None or delta["ci95_low"] is None:
        why = ("with fewer than 3 seeds there is no confidence interval; the lesson needs 3"
               if seeds < 3 else (delta or {}).get("status", "no interval"))
        return Result(check.id, check.kind, False, f"{label}: {why}", {"seeds": seeds})
    interval = f"{delta['mean']:+.3f} [{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}]"
    evidence = {"verdict": row["verdict"], "expected": expect, "metric": metric, "seeds": seeds,
                "delta": delta["mean"], "ci95": [delta["ci95_low"], delta["ci95_high"]]}
    said = f"{label}: {row['verdict']} ({metric} difference {interval}, {seeds} seeds)"
    if row["verdict"] == expect:
        return Result(check.id, check.kind, True, f"{said}, as the lesson expects", evidence)
    return Result(check.id, check.kind, False,
                  f"{said}; the lesson expects '{expect}'. Look at what differs between the two "
                  "models, and at the interval: is zero inside it?", evidence)
