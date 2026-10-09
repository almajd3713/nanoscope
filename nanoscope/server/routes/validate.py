from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope.learn import gating
from nanoscope.presets import Preset, get_preset
from nanoscope.server.models import ProblemItem
from nanoscope.server.routes.models import find_model
from nanoscope.specs import ModelSpec, ParamSpec, Problem, validate_run_spec
from nanoscope.studyspec import StudySpec

router = APIRouter(prefix="/api/validate", tags=["validate"])


class Validation(BaseModel):
    ok: bool
    problems: list[ProblemItem]
    # where the runs will land, one per seed, when the request is valid (the same names POST
    # /api/runs gives them: a run that is already done is not trained again)
    refs: list[str] = []


class StudyValidationResult(Validation):
    # how long the study would take on `devices` (nanoscope.estimate.estimate_study)
    estimate: dict[str, Any] | None = None
    toml: str | None = None  # the file Save would write (valid specs only)


class RunValidation(BaseModel):
    model: str
    preset: str = "tinystories-5min"
    kwargs: dict[str, Any] = {}
    seeds: Any = None


class StudyValidation(BaseModel):
    toml: str | None = None
    spec: dict[str, Any] | None = None
    devices: list[str] = []  # where it would run, for the estimate (default: cpu)
    workers_per_device: int = 1  # runs sharing a device, for the estimate


def model_spec(ref: str) -> tuple[ModelSpec, list[Problem]]:
    """A ModelSpec from the API's own listing (workspace models are read, never imported) and
    the locked-use problems of the file that defines it."""
    doc, file = find_model(ref)
    spec = ModelSpec(doc.name, doc.ref, doc.doc, [ParamSpec(
        p.name, p.annotation, p.default, p.required, p.from_data) for p in doc.params])
    locked: list[Problem] = []
    if file is not None:
        locked = [Problem("locked", None, f"line {u.line}: {u.message}",
                          f"{u.id} unlocks in the lesson {u.lesson}")
                  for u in _scan(file)]
    return spec, locked


def _scan(file: Any) -> list[gating.LockedUse]:
    try:
        return gating.scan(file)
    except (OSError, SyntaxError, ValueError):
        return []


def _result(problems: list[Problem], refs: list[str] | None = None) -> Validation:
    return Validation(ok=not problems, problems=[ProblemItem(**p.to_dict()) for p in problems],
                      refs=refs or [])


def planned_ref(spec: ModelSpec, preset: str, kwargs: dict[str, Any], seed: int) -> str:
    """Where one seed of this request will land (a valid request: see validate_run_spec)."""
    from nanoscope.runref import REQUIRED, run_ref

    tunable = {p.name for p in spec.tunable()}
    preset_names = {f.name for f in dataclass_fields(Preset)}
    model_kwargs = {k: v for k, v in kwargs.items() if k in tunable}
    overrides = {k: v for k, v in kwargs.items() if k not in tunable and k in preset_names}
    given = get_preset(preset)
    defaults = {p.name: REQUIRED if p.required else p.default for p in spec.params
                if not p.from_data}
    return run_ref(spec.name, defaults, model_kwargs, given, given.override(**overrides), seed)


@router.post("/run")
def validate_run(body: RunValidation) -> Validation:
    """Every problem with a run request at once, locked blocks included: what `run()` would
    refuse, without running anything or importing the model."""
    try:
        spec, locked = model_spec(body.model)
    except KeyError as exc:
        return _result([Problem("unknown_model", "model", str(exc.args[0]))])
    problems = validate_run_spec(spec, body.preset, body.kwargs, body.seeds, locked)
    if problems:
        return _result(problems)
    seeds = body.seeds
    chosen = [0] if seeds is None else list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    return _result(problems, [planned_ref(spec, body.preset, body.kwargs, n) for n in chosen])


@router.post("/study", response_model=StudyValidationResult)
def validate_study(body: StudyValidation) -> Validation:
    """Every problem with a study spec: its preset, seeds, baseline and each variant's model
    and keywords (named by their place in the spec, e.g. `variants[1].kwargs.n_kv_heads`)."""
    if (body.toml is None) == (body.spec is None):
        return _result([Problem("invalid_spec", None, "send exactly one of toml or spec")])
    try:
        spec = StudySpec.from_toml(body.toml) if body.toml is not None else StudySpec.from_dict(
            body.spec or {})
    except Exception as exc:  # TOML syntax errors and unknown keys: the library's own words
        return _result([Problem("invalid_spec", None, str(exc))])
    problems: list[Problem] = []
    preset: Preset | str = spec.preset
    if spec.custom_preset:
        try:
            preset = Preset(**spec.custom_preset)
        except TypeError as exc:
            problems.append(Problem("invalid_spec", "custom_preset", str(exc)))
            preset = spec.preset
    names = [v.name for v in spec.variants]
    if not spec.variants:
        problems.append(Problem("invalid_spec", "variants", "a study needs at least one variant"))
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(Problem("invalid_spec", "variants", f"variant name {name!r} is used twice"))
    if spec.baseline and spec.baseline not in names:
        problems.append(Problem("invalid_spec", "baseline",
                                f"baseline {spec.baseline!r} is not a variant "
                                f"(variants: {', '.join(names) or 'none'})"))
    if spec.mode not in ("explore", "record"):
        problems.append(Problem("invalid_spec", "mode", "mode must be explore or record"))
    if not isinstance(preset, Preset):
        try:
            get_preset(preset)
        except KeyError as exc:
            problems.append(Problem("unknown_preset", "preset", str(exc.args[0])))
    preset_fields = {f.name for f in dataclass_fields(Preset)}
    for key in spec.overrides:
        if key not in preset_fields:
            problems.append(Problem("unknown_keyword", f"overrides.{key}",
                                    f"{key!r} is not a preset field"))
    if not (isinstance(spec.seeds, list) and spec.seeds and all(
            isinstance(s, int) and not isinstance(s, bool) for s in spec.seeds)):
        problems.append(Problem("bad_seeds", "seeds", f"seeds must be a list of integers, got "
                                f"{spec.seeds!r}", "for example seeds = [0, 1, 2]"))
    for i, variant in enumerate(spec.variants):
        where = f"variants[{i}]"
        try:
            vspec, locked = model_spec(variant.model)
        except KeyError as exc:
            problems.append(Problem("unknown_model", f"{where}.model", str(exc.args[0])))
            continue
        known = {p.name for p in vspec.tunable()}
        for problem in validate_run_spec(vspec, preset, variant.kwargs, None, locked):
            if problem.code == "unknown_preset":
                continue  # reported once, above
            field = f"{where}.kwargs.{problem.field}" if problem.field else where
            problems.append(Problem(problem.code, field, problem.message, problem.hint))
        if spec.match_knob and spec.match_knob not in known and variant.name != spec.match_to:
            problems.append(Problem(
                "invalid_spec", "match_knob",
                f"{variant.name} has no parameter {spec.match_knob!r} to match on",
                f"its parameters: {', '.join(sorted(known)) or 'none'}"))
    if problems:
        return _result(problems)
    from nanoscope.estimate import estimate_study, spec_runs

    return StudyValidationResult(
        ok=True, problems=[], estimate=estimate_study(spec_runs(spec), body.devices or None,
                                body.workers_per_device),
        toml=spec.to_toml())
