"""`create_app(settings)`: the FastAPI application. Everything is under `/api`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import FileResponse

from nanoscope import __version__
from nanoscope.schemas import CURRENT
from nanoscope.server import worker
from nanoscope.server.auth import install as install_auth
from nanoscope.server.errors import install as install_errors
from nanoscope.server.errors import problem
from nanoscope.server.models import Health, ProblemDetails, Version
from nanoscope.server.routes import (
    blocks,
    compare,
    events,
    files,
    graph,
    hardware,
    hub,
    jobs,
    learn,
    meta,
    models,
    presets,
    runs,
    studies,
    validate,
)
from nanoscope.server.settings import Settings, check

# every resource module adds its `router` here
ROUTES = [presets, models, blocks, files, graph, validate, events, runs, compare, studies,
          learn, hardware, jobs, hub, meta]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the local worker (if asked) with the server and stop it with the server."""
    settings: Settings = app.state.settings
    proc = worker.start(settings.worker) if settings.worker else None
    app.state.worker = proc
    try:
        yield
    finally:
        if proc is not None:
            worker.stop(proc)


def default_static_dir() -> Path:
    """Where a built web app is shipped in the wheel (absent until the GUI is built)."""
    return Path(__file__).parent / "static"


def install_spa(app: FastAPI, folder: Path) -> None:
    """Serve a built single-page app at `/`: real files as they are, and `index.html` for any
    other path without an extension, so client-side routes survive a reload. `/api` stays the
    API: an unknown API path is a 404 problem, never the page."""
    index = folder / "index.html"
    if not index.is_file():
        return
    root = folder.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request) -> Response:
        if path == "api" or path.startswith("api/"):
            return problem(404, f"no endpoint {request.url.path}", request)
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        if "." in Path(path).name:  # a missing asset is a 404, not the app
            return problem(404, f"no file {path}", request)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    check(settings)
    app = FastAPI(
        title="nanoscope", version=__version__, docs_url="/api/docs",
        openapi_url="/api/openapi.json", redoc_url=None, lifespan=lifespan,
        responses={
            422: {"model": ProblemDetails, "description": "The request cannot be done as asked"},
            "4XX": {"model": ProblemDetails, "description": "An error (RFC 9457)"}})
    app.state.settings = settings

    api = APIRouter(prefix="/api")

    @api.get("/health", tags=["meta"])
    def health() -> Health:
        return Health(status="ok")

    @api.get("/version", tags=["meta"])
    def version() -> Version:
        return Version(nanoscope=__version__, schemas=dict(CURRENT))

    install_errors(app)
    app.include_router(api)
    for module in ROUTES:
        app.include_router(module.router)
    install_auth(app, settings)
    install_spa(app, settings.static_dir or default_static_dir())
    return app
