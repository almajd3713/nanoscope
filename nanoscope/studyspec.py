"""A study as data: a TOML file the API, the GUI and the CLI can read, write and commit.

`Study.to_spec()` / `Study.from_spec(spec)` convert between a `Study` and a `StudySpec`;
`to_toml` / `from_toml` between a spec and text. Record mode can preregister the TOML file
itself instead of a .py file.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # pyright: ignore[reportMissingImports]


def _kv(key: str, value: Any) -> str:
    return tomli_w.dumps({key: value}).strip()


def _collapse_arrays(text: str) -> str:
    """tomli_w spreads every array over several lines; keep arrays of plain values on one."""
    def one_line(match: re.Match[str]) -> str:
        return "[" + ", ".join(x.strip().rstrip(",") for x in match.group(1).splitlines()) + "]"

    return re.sub(r"\[\n((?:    [^\n\[\]{}]+,\n)+)\]", one_line, text)


@dataclass
class VariantSpec:
    name: str
    model: str  # a ref: "nanoscope.models:Modern" or "path/to/file.py:MyLM"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudySpec:
    name: str
    preset: str = "tinystories-5min"
    overrides: dict[str, Any] = field(default_factory=dict)  # preset fields changed for this study
    custom_preset: dict[str, Any] | None = None  # a preset that isn't registered, in full
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    budget: dict[str, float] | None = None  # {"tokens": n} or {"flops": n}
    match: str | None = None
    match_knob: str | None = None  # the model parameter to resize so variants match
    match_to: str | None = None  # the variant whose size the others are matched to
    range: list[int] | None = None  # [start, stop, step] of values to try for match_knob
    baseline: str | None = None
    mode: str = "explore"
    tolerance: float = 0.02
    variants: list[VariantSpec] = field(default_factory=list)
    predictions: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"schema": 1, "name": self.name, "preset": self.preset}
        for key in ("overrides", "custom_preset", "budget", "match", "match_knob", "match_to",
                    "range", "baseline"):
            value = getattr(self, key)
            if value:
                doc[key] = value
        doc.update({"seeds": self.seeds, "mode": self.mode, "tolerance": self.tolerance})
        doc["variants"] = [{"name": v.name, "model": v.model, **({"kwargs": v.kwargs}
                                                                  if v.kwargs else {})}
                           for v in self.variants]
        if self.predictions:
            doc["predictions"] = self.predictions
        return doc

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> StudySpec:
        known = {"schema", "name", "preset", "overrides", "custom_preset", "seeds", "budget",
                 "match", "match_knob", "match_to", "range", "baseline", "mode", "tolerance",
                 "variants", "predictions"}
        unknown = set(doc) - known
        if unknown:
            raise ValueError(f"unknown study spec keys: {', '.join(sorted(unknown))}")
        if "name" not in doc:
            raise ValueError("a study spec needs a name")
        body = {k: v for k, v in doc.items() if k not in ("schema", "variants")}
        spec = cls(**body)
        spec.variants = [VariantSpec(v["name"], v["model"], dict(v.get("kwargs", {})))
                         for v in doc.get("variants", [])]
        return spec

    def to_toml(self) -> str:
        """Readable TOML: one `[[variants]]` block per variant, short arrays on one line."""
        doc = self.to_dict()
        variants = doc.pop("variants")
        parts = [tomli_w.dumps(doc).rstrip()]
        for v in variants:
            block = ["[[variants]]", _kv("name", v["name"]), _kv("model", v["model"])]
            if v.get("kwargs"):
                block += ["", "[variants.kwargs]", *(_kv(k, x) for k, x in v["kwargs"].items())]
            parts.append("\n".join(block))
        return _collapse_arrays("\n\n".join(parts)) + "\n"

    @classmethod
    def from_toml(cls, text: str) -> StudySpec:
        return cls.from_dict(tomllib.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> StudySpec:
        return cls.from_toml(Path(path).read_text(encoding="utf-8"))


def check_tomlable(variant: str, kwargs: dict[str, Any]) -> None:
    """TOML has no null and no arbitrary objects: say which keyword can't be written."""
    for key, value in kwargs.items():
        items = value if isinstance(value, (list, tuple)) else [value]
        if not all(isinstance(v, (int, float, str, bool)) for v in items):
            raise ValueError(
                f"variant {variant!r}: {key}={value!r} can't be written to a study spec "
                "(only numbers, strings, booleans and lists of them)")


def resolve_variants(spec: StudySpec, preset: Any) -> list[tuple[str, type, dict[str, Any]]]:
    """(name, model class, kwargs) per variant, with `match_knob` filled in where asked.

    Declarative matching: every variant that has the knob and doesn't set it is resized so
    its non-embedding parameter count is closest to the `match_to` variant's.
    """
    import inspect

    from nanoscope.modelref import load_class
    from nanoscope.sizing import build_on_meta, count_params, match_params

    items = [(v.name, load_class(v.model), dict(v.kwargs)) for v in spec.variants]
    if not spec.match_knob:
        return items
    if not spec.range or not spec.match_to:
        raise ValueError("match_knob needs range = [start, stop, step] and match_to = a variant")
    target_item = next((i for i in items if i[0] == spec.match_to), None)
    if target_item is None:
        raise ValueError(f"match_to {spec.match_to!r} is not a variant; "
                         f"variants: {', '.join(i[0] for i in items)}")
    vocab = preset.vocab_size
    if vocab is None:
        from nanoscope.dataset import load_tokenizer
        vocab = load_tokenizer(preset).vocab_size
    from_data = {"vocab_size": vocab, "context_length": preset.context_length}

    def built_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        params = inspect.signature(cls).parameters
        return {k: v for k, v in from_data.items() if k in params} | kwargs

    _, target_cls, target_kwargs = target_item
    target = count_params(build_on_meta(target_cls, **built_kwargs(target_cls, target_kwargs)))[1]
    out = []
    for name, cls, kwargs in items:
        has_knob = spec.match_knob in inspect.signature(cls).parameters
        if name != spec.match_to and has_knob and spec.match_knob not in kwargs:
            resolved = match_params(cls, target, spec.match_knob, range(*spec.range),
                                    **built_kwargs(cls, kwargs))
            kwargs = {k: v for k, v in resolved.items() if k not in from_data or k in kwargs}
        out.append((name, cls, kwargs))
    return out
