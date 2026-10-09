import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import assert_valid

from nanoscope import paths, queue
from nanoscope.cli import main
from nanoscope.server.app import create_app

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_learn import AUTHOR_TOML, author_folder  # noqa: E402


@pytest.fixture
def client(home):
    return TestClient(create_app())


def lesson_in_workspace(**kw):
    root = paths.workspace_dir() / "curricula"
    root.mkdir(parents=True, exist_ok=True)
    folder = author_folder(root, **kw)
    return folder.relative_to(paths.workspace_dir()).as_posix()


def test_lists_lesson_folders(client):
    assert client.get("/api/authoring").json() == []
    rel = lesson_in_workspace()
    [found] = client.get("/api/authoring").json()
    assert found["folder"] == rel and found["id"] == "my-course/01-thing"
    assert "solution.py" in found["files"]


def test_reads_a_folder_that_loads(client):
    rel = lesson_in_workspace()
    doc = client.get(f"/api/authoring/{rel}").json()
    assert doc["loads"] is True and doc["problems"] == [] and doc["result"] is None
    assert doc["lesson"]["title"] == "My layer" and doc["lesson"]["text"]["surface"]
    assert doc["files"]["solution.py"] is True and doc["files"]["notebook.py"] is False


def test_a_folder_that_does_not_load_lists_every_problem(client):
    rel = lesson_in_workspace(toml=AUTHOR_TOML.replace("[compute.cpu]", "[compute.gpu]")
                              .replace('kind = "defines"', 'kind = "x"'))
    doc = client.get(f"/api/authoring/{rel}").json()
    assert doc["loads"] is False and doc["lesson"] is None
    assert any(p["where"].endswith("checks[0].kind") for p in doc["problems"])


def test_check_is_a_job_and_its_result_is_read_back(client):
    rel = lesson_in_workspace()
    sent = client.post(f"/api/authoring/{rel}/check", json={})
    assert sent.status_code == 202
    job = sent.json()
    assert job["kind"] == "author-check" and job["payload"]["variant"] == "cpu"
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as done:
        main(["run-job", str(job["id"])])
    assert done.value.code == 0
    result = json.loads(queue.get(job["id"])["result"])
    assert_valid("author-check", result)
    read = client.get(f"/api/authoring/{rel}").json()["result"]
    assert read["state"] == "ready" and read["fresh"] is True
    (paths.workspace_dir() / rel / "starter.py").write_text("x = 1\n")
    assert client.get(f"/api/authoring/{rel}").json()["result"]["fresh"] is False


def test_paths_stay_inside_the_workspace(client):
    assert client.get("/api/authoring/../etc").status_code in (400, 404)
    assert client.get("/api/authoring/curricula/none/01-x").status_code == 404
