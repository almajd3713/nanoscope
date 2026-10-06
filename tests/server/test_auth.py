import os
import stat

import pytest
from fastapi.testclient import TestClient

from nanoscope.server.app import create_app
from nanoscope.server.settings import (
    Settings,
    check,
    ensure_token,
    is_loopback,
    login_url,
    token_path,
)

TOKEN = "s3cret-token"


def remote(**kw):
    return Settings(host="0.0.0.0", token=TOKEN, **kw)


def test_is_loopback():
    assert all(is_loopback(h) for h in ("127.0.0.1", "127.1.2.3", "::1", "localhost"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "192.168.1.5", "::", "example.com"))


def test_refuse_without_token():
    with pytest.raises(ValueError, match="refusing to listen on 0.0.0.0 without a token"):
        create_app(Settings(host="0.0.0.0"))
    with pytest.raises(ValueError):
        check(Settings(host="192.168.1.5", token=""))
    create_app(Settings(host="127.0.0.1"))  # loopback needs none
    create_app(remote())  # a token is enough


def test_token_is_made_once_readable_only_by_you(home):
    assert not token_path().exists()
    token = ensure_token({})
    assert len(token) >= 40 and token_path().read_text().strip() == token
    assert stat.S_IMODE(os.stat(token_path()).st_mode) == 0o600
    assert ensure_token({}) == token  # the same one on the next start
    assert ensure_token({"NANOSCOPE_TOKEN": "from-env"}) == "from-env"
    assert login_url(remote(port=9000)) == f"http://localhost:9000/login?token={TOKEN}"


def test_serve_beyond_loopback_prints_the_login_url(home, monkeypatch, capsys):
    import uvicorn

    from nanoscope.cli import main

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    main(["serve", "--host", "0.0.0.0", "--port", "9100"])
    out = capsys.readouterr().out
    token = token_path().read_text().strip()
    assert f"sign in at http://localhost:9100/login?token={token}" in out
    assert started["host"] == "0.0.0.0"
    main(["serve"])  # loopback: no token printed
    assert "login" not in capsys.readouterr().out


def test_no_auth_on_loopback(home):
    client = TestClient(create_app(Settings()))
    assert client.get("/api/version").status_code == 200


def test_bearer_header_or_login_cookie_beyond_loopback(home):
    client = TestClient(create_app(remote()), follow_redirects=False)
    assert client.get("/api/health").status_code == 200  # open for probes
    denied = client.get("/api/version")
    assert denied.status_code == 401 and denied.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/version", headers={"Authorization": "Bearer nope"}).status_code == 401
    good = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/api/version", headers=good).status_code == 200
    assert client.get("/login?token=nope").status_code == 401
    signed_in = client.get(f"/login?token={TOKEN}")
    assert signed_in.status_code == 303 and signed_in.headers["location"] == "/"
    cookie = signed_in.headers["set-cookie"]
    assert "nanoscope_token=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert client.get("/api/version").status_code == 200  # the cookie is kept by the client
    other = TestClient(create_app(remote()))
    assert other.get("/api/version").status_code == 401  # a fresh client has none
