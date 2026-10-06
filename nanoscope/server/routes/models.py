from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope import paths
from nanoscope.blocks.discover import discover_models
from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.server.jobs import submit
from nanoscope.server.models import JobDoc, ModelSpecDoc, ParamSpecDoc
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


def resolve_ref(ref: str) -> str:
    """The ref a worker can load: shipped names and module refs pass through, a workspace file
    ref (`sub/tiny.py:Tiny`) becomes an absolute path (the worker runs elsewhere)."""
    for spec in models():
        if ref in (spec.ref, spec.name, spec.name.lower()):
            if spec.shipped:
                return spec.ref
            file, _, cls = spec.ref.rpartition(":")
            return f"{paths.workspace_dir().resolve() / file}:{cls}"
    raise KeyError(f"unknown model {ref!r}; available: "
                   f"{', '.join(s.ref for s in models()) or 'none'}")


class DescribeRequest(BaseModel):
    preset: str = "tinystories-5min"
    kwargs: dict[str, Any] = {}


@router.post("/models/{ref:path}/describe", status_code=202)
def describe_model(ref: str, body: DescribeRequest | None = None) -> JobDoc:
    """Shapes, parameters, FLOPs and memory per module. It builds the model, which runs the
    learner's code, so it is a job for a worker, never done in this process."""
    body = body or DescribeRequest()
    return submit("describe", {"model": resolve_ref(ref), "preset": body.preset,
                               "kwargs": body.kwargs})


@router.get("/models/{ref:path}")
def model(ref: str) -> ModelSpecDoc:
    for spec in models():
        if ref in (spec.ref, spec.name, spec.name.lower()):
            return spec
    raise KeyError(f"unknown model {ref!r}; available: "
                   f"{', '.join(s.ref for s in models()) or 'none'}")
