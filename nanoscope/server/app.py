"""`create_app(settings)`: the FastAPI application. Everything is under `/api`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from nanoscope import __version__
from nanoscope.schemas import CURRENT
from nanoscope.server import worker
from nanoscope.server.auth import install as install_auth
from nanoscope.server.errors import install as install_errors
from nanoscope.server.models import Health, Version
from nanoscope.server.routes import (
    blocks,
    files,
    graph,
    models,
    presets,
    runs,
    validate,
)
from nanoscope.server.settings import Settings, check

# every resource module adds its `router` here
ROUTES = [presets, models, blocks, files, graph, validate, runs]


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    check(settings)
    app = FastAPI(
        title="nanoscope", version=__version__, docs_url="/api/docs",
        openapi_url="/api/openapi.json", redoc_url=None, lifespan=lifespan)
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
    return app
