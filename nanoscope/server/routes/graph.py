from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from nanoscope.blocks.graph import apply_edits, emit, parse
from nanoscope.learn import gating
from nanoscope.learn.gating import Locked, LockedBlockError
from nanoscope.server import workspace
from nanoscope.server.errors import problem
from nanoscope.server.models import GraphDoc

router = APIRouter(prefix="/api", tags=["graph"])


class GraphResponse(GraphDoc):
    etag: str


class PatchRequest(BaseModel):
    """Either the edited graph (every changed argument is patched into the source), or an
    explicit list of edits (`set_arg`, `replace_block`, `remove_arg`, and the structural
    `add_layer`, `remove_layer`, `set_pattern`, `fill_slot`; see nanoscope.blocks.graph)."""

    graph: dict[str, Any] | None = None
    edits: list[dict[str, Any]] | None = None


def _graph(target: Path) -> dict[str, Any]:
    doc = parse(target)
    doc["path"] = workspace.relative(target)
    return doc


def _locked_ids(source: str) -> dict[str, gating.LockedUse]:
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "patched.py"
        scratch.write_text(source, encoding="utf-8")
        return {u.id: u for u in gating.scan(scratch)}


@router.post("/files/{path:path}/graph")
def file_graph(path: str) -> GraphResponse:
    """The architecture graph of a model file, read with `ast` (the file is never imported)."""
    target = workspace.safe_path(path)
    return GraphResponse(**_graph(target), etag=workspace.etag_of(target))


@router.post("/files/{path:path}/graph/patch")
def patch_graph(path: str, body: PatchRequest, request: Request) -> Any:
    """Apply graph edits to the source, changing only the edited arguments. Needs `If-Match`
    like any save. Under the `guided` policy a patch that adds a block or feature the learner
    has not unlocked is refused with a 422 naming the lesson."""
    from fastapi.responses import JSONResponse

    target = workspace.safe_path(path)
    current = workspace.etag_of(target)
    sent = request.headers.get("if-match")
    if sent is None:
        return problem(428, "send If-Match with the ETag you read", request, current_etag=current)
    if sent != current:
        return problem(409, f"{path} changed since you read it; nothing was patched", request,
                       current_etag=current)
    if (body.graph is None) == (body.edits is None):
        return problem(422, "send exactly one of graph (the edited graph) or edits", request)
    source = target.read_text(encoding="utf-8")
    patched = emit(body.graph, source) if body.graph is not None else apply_edits(
        source, body.edits or [])
    if patched != source:
        before, after = _locked_ids(source), _locked_ids(patched)
        added = [use for unlock_id, use in after.items() if unlock_id not in before]
        if added:
            raise LockedBlockError([Locked(u.id, u.lesson, u.message) for u in added])
        workspace.write_atomic(target, patched)
    return JSONResponse({**_graph(target), "etag": workspace.etag_of(target)},
                        headers={"ETag": workspace.etag_of(target)})
