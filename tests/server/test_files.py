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


def test_write_and_conflict_409(client, ws):
    path = "/api/files/models/my_lm.py"
    first = client.get(path).json()
    # a new file needs no ETag; its parent folders are made
    created = client.put("/api/files/new/dir/x.py", json={"content": "x = 1\n"})
    assert created.status_code == 201 and (ws / "new" / "dir" / "x.py").read_text() == "x = 1\n"
    assert created.json()["etag"] == created.headers["etag"]
    # an existing file must say what it read
    bare = client.put(path, json={"content": "print('mine')\n"})
    assert bare.status_code == 428 and bare.json()["current_etag"] == first["etag"]
    assert (ws / "models" / "my_lm.py").read_text() == "print('hi')\n"  # untouched
    saved = client.put(path, json={"content": "print('mine')\n"},
                       headers={"If-Match": first["etag"]})
    assert saved.status_code == 200 and saved.json()["etag"] != first["etag"]
    assert (ws / "models" / "my_lm.py").read_text() == "print('mine')\n"
    # someone else (an editor, git checkout) changed it meanwhile: 409 with a diff, no write
    (ws / "models" / "my_lm.py").write_text("print('theirs')\n")
    conflict = client.put(path, json={"content": "print('mine, again')\n"},
                          headers={"If-Match": saved.json()["etag"]})
    assert conflict.status_code == 409
    problem = conflict.json()
    assert problem["current_etag"] == client.get(path).json()["etag"]
    assert "--- models/my_lm.py (on disk)" in problem["diff"]
    assert "-print('theirs')" in problem["diff"] and "+print('mine, again')" in problem["diff"]
    assert (ws / "models" / "my_lm.py").read_text() == "print('theirs')\n"
    # no temp files are left behind, and the traversal guard covers writes too
    assert not [p for p in (ws / "models").iterdir() if p.name.endswith(".tmp")]
    assert client.put("/api/files/../escape.py", json={"content": "x"}).status_code in (400, 404)
    assert client.put("/api/files/%2e%2e/escape.py", json={"content": "x"}).status_code == 400


def test_external_edit_event(ws):
    """Whatever changes a workspace file (an editor, git, another process) is announced."""
    import asyncio
    import json

    from nanoscope.server.routes.files import watch_events

    def parse(frame):
        event, data = frame.strip().split("\n")
        return event.removeprefix("event: "), json.loads(data.removeprefix("data: "))

    async def scenario():
        stop = asyncio.Event()
        stream = watch_events(stop, interval_ms=500)

        async def touch():
            await asyncio.sleep(0.4)
            (ws / "models" / "my_lm.py").write_text("print('edited by hand')\n")
            (ws / "fresh.py").write_text("x = 1\n")
            (ws / "__pycache__" / "junk.pyc").write_bytes(b"1")  # ignored
            (ws / "notes.md").unlink()

        task = asyncio.create_task(touch())
        seen = {}

        async def collect():
            async for frame in stream:
                if frame.startswith(":"):
                    continue
                event, data = parse(frame)
                assert event == "change"
                seen[data["path"]] = data
                if {"models/my_lm.py", "fresh.py", "notes.md"} <= set(seen):
                    break

        await asyncio.wait_for(collect(), 20)  # asyncio.timeout needs Python 3.11
        stop.set()
        await task
        return seen

    seen = asyncio.run(scenario())
    assert seen["models/my_lm.py"]["kind"] == "modified" and seen["models/my_lm.py"]["etag"]
    assert seen["fresh.py"]["kind"] == "added"
    assert seen["notes.md"]["kind"] == "deleted" and seen["notes.md"]["etag"] is None
    assert "__pycache__/junk.pyc" not in seen


def test_the_events_route_is_not_shadowed_by_the_file_route(client):
    spec = client.get("/api/openapi.json").json()["paths"]
    assert "/api/files/events" in spec and "/api/files/{path}" in spec
    assert list(spec).index("/api/files/events") < list(spec).index("/api/files/{path}")
