"""Errors as RFC 9457 `application/problem+json`.

The messages of the library's `ValueError`s and `TypeError`s are the product (they say what to
change), so they pass through word for word as the `detail` of a 422. A locked block becomes a
422 that names the lesson that unlocks it.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from nanoscope.learn.gating import LockedBlockError

MEDIA_TYPE = "application/problem+json"
TITLES = {400: "Bad request", 401: "Not signed in", 403: "Forbidden", 404: "Not found",
          405: "Method not allowed", 409: "Conflict", 412: "Precondition failed",
          422: "Cannot be done as asked", 500: "Server error"}


def problem(status: int, detail: str, request: Request | None = None, *, title: str | None = None,
            code: str = "about:blank", headers: dict[str, str] | None = None,
            **extra: Any) -> JSONResponse:
    body: dict[str, Any] = {"type": code, "title": title or TITLES.get(status, "Error"),
                            "status": status, "detail": detail}
    if request is not None:
        body["instance"] = request.url.path
    body.update(extra)
    return JSONResponse(body, status_code=status, media_type=MEDIA_TYPE, headers=headers)


def install(app: FastAPI) -> None:
    @app.exception_handler(LockedBlockError)
    async def locked(request: Request, exc: LockedBlockError) -> JSONResponse:
        return problem(422, str(exc), request, title="Locked until you build it", code="locked",
                       lesson=exc.locked.lesson, unlock_id=exc.locked.id,
                       locked=[{"id": x.id, "lesson": x.lesson} for x in exc.all])

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError) -> JSONResponse:
        extra = {}
        if getattr(exc, "problems", None):  # CurriculumError and other multi-problem errors
            extra["problems"] = [p.to_dict() if hasattr(p, "to_dict") else str(p)
                                 for p in exc.problems]  # type: ignore[attr-defined]
        return problem(422, str(exc), request, **extra)

    @app.exception_handler(TypeError)
    async def wrong_type(request: Request, exc: TypeError) -> JSONResponse:
        return problem(422, str(exc), request)

    @app.exception_handler(FileNotFoundError)
    async def missing(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return problem(404, str(exc), request)

    @app.exception_handler(RequestValidationError)
    async def bad_body(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                  for e in exc.errors()]
        detail = "; ".join(f"{e['field']}: {e['message']}" for e in errors)
        return problem(422, detail, request, title="The request is not valid", errors=errors)

    @app.exception_handler(StarletteHTTPException)
    async def http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(exc.status_code, str(exc.detail), request,
                       headers=dict(exc.headers) if exc.headers else None)
