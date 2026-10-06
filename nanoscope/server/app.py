"""`create_app(settings)`: the FastAPI application. Everything is under `/api`."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from nanoscope import __version__
from nanoscope.schemas import CURRENT
from nanoscope.server.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="nanoscope", version=__version__, docs_url="/api/docs",
        openapi_url="/api/openapi.json", redoc_url=None)
    app.state.settings = settings

    api = APIRouter(prefix="/api")

    @api.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/version", tags=["meta"])
    def version() -> dict[str, object]:
        return {"nanoscope": __version__, "schemas": dict(CURRENT)}

    app.include_router(api)
    return app
