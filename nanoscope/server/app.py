"""`create_app(settings)`: the FastAPI application. Everything is under `/api`."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from nanoscope import __version__
from nanoscope.schemas import CURRENT
from nanoscope.server.auth import install as install_auth
from nanoscope.server.errors import install as install_errors
from nanoscope.server.models import Health, Version
from nanoscope.server.settings import Settings, check


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    check(settings)
    app = FastAPI(
        title="nanoscope", version=__version__, docs_url="/api/docs",
        openapi_url="/api/openapi.json", redoc_url=None)
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
    install_auth(app, settings)
    return app
