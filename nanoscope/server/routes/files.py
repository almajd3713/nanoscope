from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nanoscope.server import sse, workspace
from nanoscope.server.errors import problem

router = APIRouter(prefix="/api", tags=["files"])


class FileEntry(BaseModel):
    path: str
    size: int
    modified: float
    etag: str


class FileDoc(BaseModel):
    path: str
    content: str
    etag: str


@router.get("/files")
def list_files(glob: str = "**/*") -> list[FileEntry]:
    """Files in the workspace matching `glob` (default: all)."""
    out = []
    for path in workspace.listing(glob):
        stat = path.stat()
        out.append(FileEntry(path=workspace.relative(path), size=stat.st_size,
                             modified=stat.st_mtime, etag=workspace.etag_of(path)))
    return out


KINDS = {1: "added", 2: "modified", 3: "deleted"}  # watchfiles.Change values


async def watch_events(stop: asyncio.Event | None = None,
                       interval_ms: int = 15000) -> AsyncIterator[str]:
    """SSE frames for every change under the workspace (an editor, `git checkout`, the API's
    own saves), and a keepalive comment while it is quiet."""
    from watchfiles import awatch

    base = workspace.root()
    base.mkdir(parents=True, exist_ok=True)
    async for changes in awatch(base, stop_event=stop, yield_on_timeout=True,
                                rust_timeout=interval_ms, debounce=50):
        if not changes:
            yield sse.KEEPALIVE
            continue
        for change, name in sorted(changes, key=lambda c: c[1]):
            path = Path(name)
            if not path.is_relative_to(base):
                continue
            parts = path.relative_to(base).parts
            if any(p in workspace.IGNORED or p.startswith(".git") for p in parts) or (
                    path.name.endswith(".tmp")):
                continue
            etag = workspace.etag_of(path) if path.is_file() else None
            yield sse.frame("change", {"path": path.relative_to(base).as_posix(),
                                       "kind": KINDS[change.value], "etag": etag})


@router.get("/files/events")
async def file_events() -> StreamingResponse:
    """Server-sent events: `change` {path, kind, etag} when a workspace file is added, edited
    or deleted by anything, so an open editor can reload."""
    return sse.response(watch_events())


@router.get("/files/{path:path}", response_model=FileDoc)
def read_file(path: str, request: Request) -> Any:
    """A text file with its ETag. Send it back as `If-Match` when you save, so an edit made
    elsewhere is never overwritten silently."""
    target = workspace.safe_path(path)
    etag = workspace.etag_of(target)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problem(415, f"{path} is not a UTF-8 text file", request)
    from fastapi.responses import JSONResponse

    return JSONResponse({"path": path, "content": content, "etag": etag},
                        headers={"ETag": etag})


class FileWrite(BaseModel):
    content: str


@router.put("/files/{path:path}", response_model=FileDoc)
def write_file(path: str, body: FileWrite, request: Request) -> Any:
    """Save a file. For an existing file send `If-Match` with the ETag you read: if the file
    changed since, nothing is written and the answer is 409 with the diff between what is on
    disk and what you sent. A new file needs no ETag."""
    import difflib

    from fastapi.responses import JSONResponse

    target = workspace.safe_path(path, must_exist=False)
    existed = target.exists()
    if existed:
        sent = request.headers.get("if-match")
        current = workspace.etag_of(target)
        if sent is None:
            return problem(428, "this file exists: send If-Match with the ETag you read, so "
                           "an edit made elsewhere is not overwritten", request,
                           title="Precondition required", current_etag=current)
        if sent != current:
            on_disk = target.read_text(encoding="utf-8", errors="replace")
            diff = "".join(difflib.unified_diff(
                on_disk.splitlines(keepends=True), body.content.splitlines(keepends=True),
                fromfile=f"{path} (on disk)", tofile=f"{path} (yours)"))
            return problem(409, f"{path} changed since you read it; nothing was written",
                           request, current_etag=current, diff=diff)
    workspace.write_atomic(target, body.content)
    etag = workspace.etag_of(target)
    return JSONResponse({"path": path, "content": body.content, "etag": etag},
                        status_code=200 if existed else 201, headers={"ETag": etag})


class Diagnostic(BaseModel):
    code: str | None
    message: str
    line: int
    column: int
    end_line: int
    end_column: int
    fixable: bool = False


class LintRequest(BaseModel):
    content: str | None = None  # lint unsaved editor text instead of the file on disk


RUFF_ARGS = ["check", "--isolated", "--no-cache", "--select", "E,F,W,B", "--ignore", "E501",
             "--output-format", "json"]


@router.post("/files/{path:path}/lint")
def lint_file(path: str, body: LintRequest | None = None) -> list[Diagnostic]:
    """Diagnostics from ruff for a workspace file, or for unsaved text sent as `content`. The
    text goes to ruff on stdin: nothing is imported or run, and ruff reads no project config,
    so a learner's file is judged the same everywhere."""
    import json
    import shutil
    import subprocess
    import sys

    target = workspace.safe_path(path)
    text = body.content if body and body.content is not None else target.read_text(
        encoding="utf-8")
    ruff = shutil.which("ruff")
    cmd = [ruff] if ruff else [sys.executable, "-m", "ruff"]
    done = subprocess.run([*cmd, *RUFF_ARGS, "--stdin-filename", path, "-"], input=text,
                          capture_output=True, text=True, timeout=30)
    if done.returncode not in (0, 1):  # 1 means "found problems"
        raise RuntimeError(f"ruff failed: {done.stderr.strip() or done.stdout.strip()}")
    return [Diagnostic(
        code=d.get("code"), message=d["message"], line=d["location"]["row"],
        column=d["location"]["column"], end_line=d["end_location"]["row"],
        end_column=d["end_location"]["column"], fixable=bool(d.get("fix")))
        for d in json.loads(done.stdout or "[]")]
