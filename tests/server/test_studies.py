import pytest
from fakes import tiny
from fastapi.testclient import TestClient
from learn_helpers import set_preset

from nanoscope import queue
from nanoscope.cli import main
from nanoscope.server.app import create_app

pytestmark = pytest.mark.usefixtures("fake_data")

SPEC = '''\
name = "toy"
preset = "test-tiny"
seeds = [0, 1]
baseline = "small"

[[variants]]
name = "small"
model = "bigram"
[variants.kwargs]
d_model = 8

[[variants]]
name = "wide"
model = "bigram"
[variants.kwargs]
d_model = 32
'''


@pytest.fixture
def client(home, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    set_preset(monkeypatch, tiny())
    return TestClient(create_app())


def work_off(job_id):
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    assert stopped.value.code == 0


def test_studies_save_list_and_refuse(client, tmp_path):
    assert client.get("/api/studies").json() == []
    saved = client.post("/api/studies", json={"toml": SPEC})
    assert saved.status_code == 201, saved.text
    entry = saved.json()
    assert (entry["name"], entry["spec"], entry["preset"], entry["mode"]) == (
        "toy", "studies/toy.toml", "test-tiny", "explore")
    assert entry["seeds"] == [0, 1] and entry["runs_total"] == 4
    assert [v["name"] for v in entry["variants"]] == ["small", "wide"]
    on_disk = (tmp_path / "ws" / "studies" / "toy.toml").read_text()
    assert 'name = "toy"' in on_disk and "[[variants]]" in on_disk  # the committable file
    assert 'model = "nanoscope.models.bigram:Bigram"' in on_disk  # short names made portable
    assert [s["name"] for s in client.get("/api/studies").json()] == ["toy"]
    again = client.post("/api/studies", json={"toml": SPEC})
    assert again.status_code == 409 and "overwrite=true" in again.json()["detail"]
    assert client.post("/api/studies", json={"toml": SPEC, "overwrite": True}).status_code == 201
    ghost = SPEC.replace('model = "bigram"\n[var', 'model = "ghost"\n[var', 1)
    bad = client.post("/api/studies", json={"toml": ghost})
    assert bad.status_code == 422 and bad.json()["problems"][0]["code"] == "unknown_model"
    assert client.post("/api/studies", json={"toml": 'name = "../evil"\n'}).status_code == 422
    assert not (tmp_path / "evil.toml").exists()


def test_studies_run_report_stop(client, tmp_path):
    client.post("/api/studies", json={"toml": SPEC})
    started = client.post("/api/studies/toy/run")
    assert started.status_code == 202
    job = started.json()["job"]
    assert (job["kind"], job["lane"], job["ref"]) == ("study", "interactive", "studies/toy")
    assert job["payload"]["spec"] == str((tmp_path / "ws" / "studies" / "toy.toml").resolve())
    # a worker loads the spec (importing its models) and queues the runs
    work_off(job["id"])
    import json
    result = json.loads(queue.get(job["id"])["result"])
    assert result["study"] == "toy" and result["runs"] == 4 and len(result["jobs"]) == 4
    refs = sorted(j["ref"] for j in queue.list_jobs() if j["kind"] == "run")
    assert refs == ["studies/toy/small/seed-0", "studies/toy/small/seed-1",
                    "studies/toy/wide/seed-0", "studies/toy/wide/seed-1"]
    assert {j["lane"] for j in queue.list_jobs() if j["kind"] == "run"} == {"batch"}
    listing = client.get("/api/studies").json()[0]
    assert listing["runs_total"] == 4
    # the report needs finished runs
    early = client.get("/api/studies/toy/report")
    assert early.status_code == 422
    assert "needs finished runs of at least 2 variants" in early.json()["detail"]
    # stop cancels what has not started
    stopped = client.post("/api/studies/toy/stop")
    assert stopped.status_code == 200
    assert {j["state"] for j in queue.list_jobs() if j["kind"] == "run"} == {"cancelled"}
    # run them for real (a second submission queues them again), then report
    client.post("/api/studies/toy/run")
    run_study_job = [j for j in queue.list_jobs() if j["kind"] == "study"][-1]
    work_off(run_study_job["id"])
    queued = [j for j in queue.list_jobs() if j["kind"] == "run" and j["state"] == "queued"]
    assert len(queued) == 4
    for _ in queued:
        queue.claim("w", "cpu")
    for j in queued:
        with pytest.raises(SystemExit):
            main(["run-job", str(j["id"])])
    report = client.get("/api/studies/toy/report")
    assert report.status_code == 200, report.text
    body = report.json()
    assert (body["study"], body["mode"], body["schema"]) == ("toy", "explore", 1)
    assert [r["label"] for r in body["rows"]] == ["small", "wide"]
    assert body["rows"][0]["verdict"] == "baseline" and body["rows"][1]["delta"] is not None
    assert body["comparison"]["curves"] and body["comparison"]["precision_plan"]["n_seeds"] == 2
    assert client.get("/api/studies/toy").status_code in (404, 405)
    final = client.get("/api/studies").json()[0]
    assert final["runs_by_state"] == {"done": 4}
    assert client.post("/api/studies/nope/run").status_code == 404
    assert client.post("/api/studies/nope/stop").status_code == 404
    assert client.get("/api/studies/nope/report").status_code == 404


def test_record_mode_studies_are_not_run_here(client):
    record = SPEC.replace("seeds = [0, 1]", 'seeds = [0, 1]\nmode = "record"')
    client.post("/api/studies", json={"toml": record})
    refused = client.post("/api/studies/toy/run")
    assert refused.status_code == 422
    assert "preregistered with a commit" in refused.json()["detail"]
