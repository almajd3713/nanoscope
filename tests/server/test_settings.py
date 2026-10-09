import json

import pytest
from fastapi.testclient import TestClient

from nanoscope import __version__, paths
from nanoscope.jobs.worker import Worker
from nanoscope.server.app import create_app
from nanoscope.server.settings import Settings

SECRET = "hf_this_value_must_never_appear_in_a_response"


@pytest.fixture
def client(home):
    return TestClient(create_app(Settings(host="127.0.0.1", port=8123)))


def test_settings_says_where_things_are_and_how_it_listens(client, home, monkeypatch):
    monkeypatch.delenv("NANOSCOPE_JOBS_OFFLINE", raising=False)
    body = client.get("/api/settings").json()
    assert body["version"] == __version__
    assert body["home"] == str(home)
    assert body["workspace"] == str(paths.workspace_dir())
    assert body["data"] == str(paths.data_dir()) and body["runs"] == str(paths.runs_dir())
    assert body["listening"] == {"host": "127.0.0.1", "port": 8123, "loopback": True,
                                 "token_required": False}
    assert body["jobs_offline"] is False and body["workers"] == []
    assert body["lsp_port"] == 8767


def test_offline_mode_and_a_non_loopback_address_show(home, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_JOBS_OFFLINE", "1")
    app = create_app(Settings(host="0.0.0.0", port=9000, token="t"))
    body = TestClient(app, headers={"Authorization": "Bearer t"}).get("/api/settings").json()
    assert body["jobs_offline"] is True
    assert body["listening"]["loopback"] is False and body["listening"]["token_required"] is True


def test_workers_report_which_secrets_they_have_never_the_values(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("WANDB_API_KEY", SECRET + "_wandb")
    worker = Worker("cpu", slots=1)
    worker._write_file()
    text = client.get("/api/settings").text
    body = json.loads(text)
    assert [w["device"] for w in body["workers"]] == ["cpu"]
    assert body["workers"][0]["secrets"] == {"HF_TOKEN": True, "WANDB_API_KEY": True}
    assert SECRET not in text
    # without them the answer is no, and still only a boolean
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("WANDB_API_KEY")
    worker._write_file()
    again = client.get("/api/settings").json()["workers"][0]["secrets"]
    assert again == {"HF_TOKEN": False, "WANDB_API_KEY": False}


def test_no_environment_value_leaks(client, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "NANOSCOPE_TOKEN"):
        monkeypatch.setenv(name, SECRET)
    Worker("cpu", slots=1)._write_file()
    assert SECRET not in client.get("/api/settings").text
    assert SECRET not in client.get("/api/hardware").text
