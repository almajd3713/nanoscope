import pytest
from fastapi.testclient import TestClient

from nanoscope import __version__
from nanoscope.schemas import CURRENT
from nanoscope.server.app import create_app


@pytest.fixture
def client(home):
    return TestClient(create_app())


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_version(client):
    body = client.get("/api/version").json()
    assert body["nanoscope"] == __version__ and body["schemas"] == CURRENT


def test_everything_lives_under_api(client):
    assert client.get("/health").status_code == 404
    paths = client.get("/api/openapi.json").json()["paths"]
    assert paths and all(p.startswith("/api/") for p in paths)
    assert client.get("/api/docs").status_code == 200


def test_serve_command_starts_the_app_on_loopback_by_default(home, monkeypatch, capsys):
    import uvicorn

    from nanoscope.cli import main

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    main(["serve"])
    assert (started["host"], started["port"]) == ("127.0.0.1", 8000)
    assert started["app"].state.settings.loopback
    assert "nanoscope API on http://127.0.0.1:8000/api" in capsys.readouterr().out
    main(["serve", "--port", "9001"])
    assert started["port"] == 9001


def test_problem_json(home):
    """Library errors reach the client word for word, as RFC 9457 problems."""
    from fastapi import HTTPException
    from pydantic import BaseModel

    from nanoscope.learn.gating import Locked, LockedBlockError
    from nanoscope.learn.loader import CurriculumError
    from nanoscope.models import Bigram
    from nanoscope.presets import get_preset
    from nanoscope.run import _split_kwargs
    from nanoscope.specs import Problem

    app = create_app()

    class Body(BaseModel):
        steps: int

    @app.get("/api/t/value")
    def value():
        _split_kwargs(Bigram, get_preset("tinystories-5min"), {"colour": "red"})

    @app.get("/api/t/value-error")
    def value_error():
        raise ValueError("seeds must be a positive count or a list of integers, got 0")

    @app.get("/api/t/locked")
    def locked():
        raise LockedBlockError(Locked("block:Attention", "foundations/04-multi-head", "block:"
                                      "Attention is locked until you build it yourself in the "
                                      "lesson foundations/04-multi-head."))

    @app.get("/api/t/curriculum")
    def curriculum():
        raise CurriculumError([Problem("curriculum", "a/lesson.toml: title", "is required")])

    @app.get("/api/t/missing")
    def missing():
        raise FileNotFoundError("no run at ref 'nope' (looked in /x)")

    @app.get("/api/t/http")
    def http():
        raise HTTPException(409, "the file changed")

    @app.post("/api/t/body")
    def body(b: Body):
        return b

    client = TestClient(app)
    r = client.get("/api/t/value")
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"
    expected = str(_expected_split_error())
    assert r.json()["detail"] == expected and "'colour' is neither a parameter" in expected
    assert r.json()["instance"] == "/api/t/value" and r.json()["type"] == "about:blank"
    r = client.get("/api/t/value-error")
    assert r.json()["detail"] == "seeds must be a positive count or a list of integers, got 0"
    r = client.get("/api/t/locked")
    assert r.status_code == 422 and r.json()["lesson"] == "foundations/04-multi-head"
    assert r.json()["type"] == "locked" and "foundations/04-multi-head" in r.json()["detail"]
    assert r.json()["unlock_id"] == "block:Attention"
    r = client.get("/api/t/curriculum")
    assert r.json()["problems"][0]["message"] == "is required"
    r = client.get("/api/t/missing")
    assert r.status_code == 404 and "no run at ref 'nope'" in r.json()["detail"]
    r = client.get("/api/t/http")
    assert r.status_code == 409 and r.json()["title"] == "Conflict"
    assert r.json()["detail"] == "the file changed"
    r = client.post("/api/t/body", json={"steps": "many"})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "body.steps"
    assert client.get("/api/nope").json()["title"] == "Not found"


def _expected_split_error():
    from nanoscope.models import Bigram
    from nanoscope.presets import get_preset
    from nanoscope.run import _split_kwargs

    try:
        _split_kwargs(Bigram, get_preset("tinystories-5min"), {"colour": "red"})
    except TypeError as exc:
        return exc
