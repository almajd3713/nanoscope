from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from nanoscope import schemas

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/schemas")
def list_schemas() -> dict[str, int]:
    """Every JSON Schema nanoscope publishes and its current version."""
    return dict(schemas.CURRENT)


@router.get("/schemas/{name}")
def get_schema(name: str, version: int | None = None) -> dict[str, Any]:
    """One JSON Schema (2020-12), the version the library writes now unless you ask for another."""
    return schemas.get(name, version)
