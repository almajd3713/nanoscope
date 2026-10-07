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


def test_login_url_uses_the_public_port_when_set():
    url = login_url(remote(port=8000), {"NANOSCOPE_PUBLIC_PORT": "8765"})
    assert url == f"http://localhost:8765/login?token={TOKEN}"


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
    assert "WARNING: listening beyond this machine" in out
    assert "plain HTTP" in out and "run them as your user" in out and "SSH tunnel" in out
    main(["serve"])  # loopback: no token printed, no warning
    quiet = capsys.readouterr().out
    assert "login" not in quiet and "WARNING" not in quiet


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


def test_no_secrets(home, tmp_path, monkeypatch):
    """No response carries the value of a secret in the server's environment."""
    secrets = {"HF_TOKEN": "hf_SECRET_value_123", "HUGGING_FACE_HUB_TOKEN": "hf_SECOND_value_456",
               "WANDB_API_KEY": "wandb_SECRET_value_789", "NANOSCOPE_TOKEN": TOKEN}
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "m.py").write_text("import torch.nn as nn\nclass M(nn.Module):\n    pass\n")
    client = TestClient(create_app(remote()))
    auth = {"Authorization": f"Bearer {TOKEN}"}
    spec = client.get("/api/openapi.json", headers=auth).json()
    # work that moves secrets around: a run that pushes to the Hub and logs to W&B, a sync
    posted = [
        client.post("/api/runs", headers=auth, json={
            "model": "bigram", "push_to_hub": "me/runs", "wandb": True}),
        client.post("/api/sync/hub", headers=auth, json={"repo": "me/runs", "ref": "p/m/seed-0"}),
        client.post("/api/data/tinystories-5min/prepare", headers=auth),
        client.post("/api/models/bigram/describe", headers=auth),
        client.post("/api/bench", headers=auth, json={"model": "bigram"}),
    ]
    assert [r.status_code for r in posted] == [202] * 5
    urls = []
    for path, methods in spec["paths"].items():
        if "get" in methods and "{" not in path and path not in (
                "/api/events", "/api/files/events", "/api/learn/events"):  # endless streams
            urls.append(path)
    urls += ["/api/models/bigram", "/api/jobs/1", "/api/jobs/2", "/api/presets/tinystories-5min",
             "/api/files/m.py", "/api/curricula/foundations/01-bigram", "/api/schemas/job",
             "/api/runs/nope/seed-0", "/api/nope"]
    seen = 0
    for url in urls:
        response = client.get(url, headers=auth)
        everything = response.text + " ".join(f"{k}: {v}" for k, v in response.headers.items())
        for name, value in secrets.items():
            assert value not in everything, f"{name} leaked in GET {url}"
        seen += 1
    assert seen > 25
    for response in posted:  # the queued jobs' payloads name the repo, never the token
        assert "hf_SECRET" not in response.text and "wandb_SECRET" not in response.text
    # the one place the token is meant to appear: the cookie `GET /login` sets, never a body
    login = client.get(f"/login?token={TOKEN}", follow_redirects=False)
    assert TOKEN not in login.text and "nanoscope_token=" in login.headers["set-cookie"]
    denied = TestClient(create_app(remote())).get("/api/version")
    assert TOKEN not in denied.text + str(denied.headers)
