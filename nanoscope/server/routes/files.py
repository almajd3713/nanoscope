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
