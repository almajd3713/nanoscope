import os

import pytest
from fastapi.testclient import TestClient

from nanoscope.server.app import create_app


@pytest.fixture
def ws(tmp_path, monkeypatch, home):
    root = tmp_path / "ws"
    (root / "models").mkdir(parents=True)
    (root / "models" / "my_lm.py").write_text("print('hi')\n")
    (root / "notes.md").write_text("# notes\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("x")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (root / "blob.bin").write_bytes(b"\xff\xfe\x00")
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(root))
    return root


@pytest.fixture
def client(ws):
    return TestClient(create_app())


def test_read(client, ws):
    listing = client.get("/api/files").json()
    assert [f["path"] for f in listing] == ["blob.bin", "models/my_lm.py", "notes.md"]
    assert [f["path"] for f in client.get("/api/files?glob=**/*.py").json()] == ["models/my_lm.py"]
    response = client.get("/api/files/models/my_lm.py")
    body = response.json()
    assert response.status_code == 200 and body["content"] == "print('hi')\n"
    assert body["path"] == "models/my_lm.py"
    assert response.headers["etag"] == body["etag"] == listing[1]["etag"]
    assert client.get("/api/files/models/my_lm.py",
                      headers={"If-None-Match": body["etag"]}).status_code == 304
    missing = client.get("/api/files/nope.py")
    assert missing.status_code == 404 and "no file 'nope.py'" in missing.json()["detail"]
    binary = client.get("/api/files/blob.bin")
    assert binary.status_code == 415 and "not a UTF-8 text file" in binary.json()["detail"]
    # the ETag changes with the content
    (ws / "models" / "my_lm.py").write_text("print('changed')\n")
    assert client.get("/api/files/models/my_lm.py").json()["etag"] != body["etag"]


def test_traversal(client, ws, tmp_path):
    (tmp_path / "secret.txt").write_text("outside")
    for attempt in ("../secret.txt", "models/../../secret.txt", "%2e%2e/secret.txt",
                    "/etc/passwd", "C:/Windows/win.ini"):
        response = client.get(f"/api/files/{attempt}")
        assert response.status_code in (400, 404), attempt
        assert "outside" not in response.text
    # a symlink out of the workspace is refused as well
    os.symlink(tmp_path / "secret.txt", ws / "link.txt")
    link = client.get("/api/files/link.txt")
    assert link.status_code == 400 and "points outside the workspace" in link.json()["detail"]
    assert "link.txt" not in [f["path"] for f in client.get("/api/files").json()]
