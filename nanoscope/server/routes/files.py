from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from nanoscope.server import workspace
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
