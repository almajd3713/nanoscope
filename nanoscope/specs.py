"""What a model class or a preset accepts, as data; and checking a request against it.

The API and the GUI build their forms from these specs and show `Problem`s next to the field
at fault, so the library is the only place that knows what is valid.
"""

from __future__ import annotations

import inspect
import numbers
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

from nanoscope.modelref import model_ref
from nanoscope.presets import Preset, get_preset, list_presets

FROM_DATA = ("vocab_size", "context_length")  # filled in from the data and preset, never asked for


def _jsonable(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return repr(value)


@dataclass
class ParamSpec:
    name: str
    annotation: str | None
    default: Any
    required: bool
    from_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "annotation": self.annotation, "default": self.default,
                "required": self.required, "from_data": self.from_data}


@dataclass
class ModelSpec:
    name: str
    ref: str
    doc: str
    params: list[ParamSpec] = field(default_factory=list)

    @classmethod
    def from_class(cls, model_cls: type) -> ModelSpec:
        params = []
        for name, p in inspect.signature(model_cls).parameters.items():
            if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                continue
            annotation = None if p.annotation is p.empty else (
                p.annotation if isinstance(p.annotation, str) else getattr(
                    p.annotation, "__name__", str(p.annotation)))
            params.append(ParamSpec(
                name, annotation, _jsonable(p.default), p.default is p.empty,
                from_data=name in FROM_DATA))
        return cls(model_cls.__name__, model_ref(model_cls)[0],
                   inspect.getdoc(model_cls) or "", params)

    def tunable(self) -> list[ParamSpec]:
        """The parameters a user can set (not the ones taken from the data)."""
        return [p for p in self.params if not p.from_data]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ref": self.ref, "doc": self.doc,
                "params": [p.to_dict() for p in self.params]}


@dataclass
class FieldSpec:
    name: str
    type: str
    default: Any
    required: bool
    help: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "default": self.default,
                "required": self.required, "help": self.help}


@dataclass
class PresetSpec:
    name: str
    fields: list[FieldSpec]

    @classmethod
    def from_preset(cls, preset: Preset) -> PresetSpec:
        specs = []
        for f in fields(preset):
            required = f.default is MISSING and f.default_factory is MISSING
            specs.append(FieldSpec(f.name, str(f.type), getattr(preset, f.name), required,
                                   f.metadata.get("help", "")))
        return cls(preset.name, specs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "fields": [f.to_dict() for f in self.fields]}


@dataclass
class Problem:
    """One thing wrong with a request."""

    code: str
    field: str | None
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": self.message,
                "hint": self.hint}


_SIMPLE_TYPES = ("int", "float", "bool", "str")


def _type_problem(value: Any, annotation: str | None) -> str | None:
    """A message when value plainly doesn't fit a simple annotation like `int | None`."""
    if not annotation:
        return None
    options = [o.strip() for o in annotation.split("|")]
    if any(o not in _SIMPLE_TYPES + ("None",) for o in options):
        return None  # a type we don't check (tuples, dtypes, ...)
    if value is None:
        return None if "None" in options else f"must be {annotation}, got None"
    ok = (
        ("bool" in options and isinstance(value, bool))
        or ("int" in options and isinstance(value, numbers.Integral)
            and not isinstance(value, bool))
        or ("float" in options and isinstance(value, numbers.Real)
            and not isinstance(value, bool))
        or ("str" in options and isinstance(value, str))
    )
    return None if ok else f"must be {annotation}, got {type(value).__name__} {value!r}"


def validate_run_request(
    model: type, preset: str | Preset, kwargs: dict[str, Any], seeds: Any = None,
) -> list[Problem]:
    """Every problem with a run request at once (the first one is what `run()` raises)."""
    problems: list[Problem] = []
    if not isinstance(preset, Preset):
        try:
            get_preset(preset)
        except KeyError:
            problems.append(Problem(
                "unknown_preset", "preset", f"unknown preset {preset!r}",
                f"available: {', '.join(list_presets()) or 'none'}"))

    params = {p.name: p for p in ModelSpec.from_class(model).tunable()}
    preset_fields = {f.name: f for f in fields(Preset)}
    for key, value in kwargs.items():
        if key in params:
            note = _type_problem(value, params[key].annotation)
            if note:
                problems.append(Problem("wrong_type", key, f"{key} {note}",
                                        f"a parameter of {model.__name__}.__init__"))
        elif key in preset_fields:
            note = _type_problem(value, str(preset_fields[key].type))
            if note:
                problems.append(Problem("wrong_type", key, f"{key} {note}", "a preset field"))
        else:
            problems.append(Problem(
                "unknown_keyword", key,
                f"{key!r} is neither a parameter of {model.__name__}.__init__ "
                f"nor a preset field (see nanoscope.Preset)",
                f"model parameters: {', '.join(params) or 'none'}"))

    if seeds is not None:
        bad = (isinstance(seeds, bool)
               or (isinstance(seeds, int) and seeds < 1)
               or (isinstance(seeds, (list, tuple))
                   and (not seeds or not all(isinstance(s, int) and not isinstance(s, bool)
                                             for s in seeds)))
               or not isinstance(seeds, (int, list, tuple)))
        if bad:
            problems.append(Problem(
                "bad_seeds", "seeds", f"seeds must be a positive count or a list of integers, "
                f"got {seeds!r}", "for example seeds=3 or seeds=[0, 1, 2]"))
    return problems
