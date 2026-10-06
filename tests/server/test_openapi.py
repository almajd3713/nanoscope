import json
from pathlib import Path

from nanoscope.server.openapi import render

COMMITTED = Path(__file__).parent.parent.parent / "docs" / "openapi.json"


def test_openapi_document_is_current():
    """docs/openapi.json is what the web client's types are generated from: when the API
    changes it must be regenerated (`make openapi`) and committed."""
    assert COMMITTED.exists(), "run `make openapi` and commit docs/openapi.json"
    assert COMMITTED.read_text(encoding="utf-8") == render(), (
        "docs/openapi.json is stale: run `make openapi` and commit the result")


def test_the_document_is_complete():
    spec = json.loads(render())
    assert spec["openapi"].startswith("3.") and spec["info"]["title"] == "nanoscope"
    paths = spec["paths"]
    assert len(paths) >= 50 and all(p.startswith("/api/") for p in paths)
    for needed in ("/api/health", "/api/runs", "/api/compare", "/api/files/{path}",
                   "/api/events", "/api/learn/unlocks", "/api/schemas/{name}"):
        assert needed in paths
    # every operation says what it is for
    for path, methods in paths.items():
        for method, operation in methods.items():
            assert operation.get("summary") or operation.get("description"), (path, method)
    assert "ProblemDetails" in spec["components"]["schemas"]
    run = paths["/api/runs"]["post"]["responses"]
    assert run["422"]["content"]["application/json"]["schema"]["$ref"].endswith("ProblemDetails")
