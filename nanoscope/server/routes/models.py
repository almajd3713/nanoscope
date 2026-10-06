from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from nanoscope import paths
from nanoscope.blocks.discover import discover_models
from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.server.models import ModelSpecDoc, ParamSpecDoc
from nanoscope.specs import FROM_DATA, ModelSpec

router = APIRouter(prefix="/api", tags=["models"])

SHIPPED = {"bigram": Bigram, "gpt2": GPT2, "modern": Modern}


def shipped_models() -> list[ModelSpecDoc]:
    return [ModelSpecDoc(**ModelSpec.from_class(cls).to_dict(), shipped=True)
            for cls in SHIPPED.values()]


def workspace_models() -> list[ModelSpecDoc]:
    """Model classes in the workspace, found by reading the files (nothing is imported)."""
    root = paths.workspace_dir()
    if not root.exists():
        return []
    out = []
    for found in discover_models(root):
        file = Path(found["file"])
        relative = file.relative_to(root).as_posix() if file.is_relative_to(root) else str(file)
        out.append(ModelSpecDoc(
            name=found["name"], ref=f"{relative}:{found['name']}", doc=found["doc"],
            params=[ParamSpecDoc(**p, from_data=p["name"] in FROM_DATA) for p in found["params"]],
            source=relative))
    return out


@router.get("/models")
def models() -> list[ModelSpecDoc]:
    """The shipped models and the model classes found in the workspace."""
    return [*shipped_models(), *workspace_models()]


@router.get("/models/{ref:path}")
def model(ref: str) -> ModelSpecDoc:
    for spec in models():
        if ref in (spec.ref, spec.name, spec.name.lower()):
            return spec
    raise KeyError(f"unknown model {ref!r}; available: "
                   f"{', '.join(s.ref for s in models()) or 'none'}")
