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
