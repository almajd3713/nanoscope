import json

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
    # a run that is not done is not a result: its partial curve never moves the report
    status = tmp_path / "home" / "runs" / "studies" / "toy" / "wide" / "seed-1" / "status.json"
    doc = json.loads(status.read_text())
    status.write_text(json.dumps({**doc, "state": "running"}))
    partial = client.get("/api/studies/toy/report").json()
    assert [len(r["seeds"]) for r in partial["rows"]] == [2, 1]
    status.write_text(json.dumps(doc))
    markdown = client.get("/api/studies/toy/report.md")
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert markdown.text.startswith("# Study: toy") and "## Results" in markdown.text
    assert client.get("/api/studies/toy").status_code in (404, 405)
    final = client.get("/api/studies").json()[0]
    assert final["runs_by_state"] == {"done": 4}
    assert client.post("/api/studies/nope/run").status_code == 404
    assert client.post("/api/studies/nope/stop").status_code == 404
    assert client.get("/api/studies/nope/report").status_code == 404


def test_record_mode_runs_only_from_a_committed_spec_in_a_clean_tree(client, tmp_path):
    ws = tmp_path / "ws"
    record = SPEC.replace("seeds = [0, 1]", 'seeds = [0, 1]\nmode = "record"')
    client.post("/api/studies", json={"toml": record})
    outside = client.post("/api/studies/toy/run")
    assert outside.status_code == 422 and "git repository" in outside.json()["detail"]

    git(ws, "init", "-q", "-b", "main")
    git(ws, "config", "user.email", "t@example.com")
    git(ws, "config", "user.name", "t")
    uncommitted = client.post("/api/studies/toy/run")
    assert uncommitted.status_code == 422
    assert "commit studies/toy.toml first" in uncommitted.json()["detail"]

    git(ws, "add", "."), git(ws, "commit", "-qm", "spec")
    (ws / "notes.txt").write_text("x")
    dirty = client.post("/api/studies/toy/run")
    assert dirty.status_code == 422 and "clean git tree" in dirty.json()["detail"]
    assert dirty.json()["changed"] == ["notes.txt"]  # the files, for the page to list

    (ws / "notes.txt").unlink()
    queued = client.post("/api/studies/toy/run")
    assert queued.status_code == 202, queued.text
    work_off(queued.json()["job"]["id"])
    result = json.loads(queue.get(queued.json()["job"]["id"])["result"])
    assert result["study"] == "toy" and result["runs"] == 4
    folder = tmp_path / "home" / "runs" / "studies" / "toy"
    manifest = json.loads((folder / "study.json").read_text())
    assert manifest["study_file"] == "studies/toy.toml" and len(manifest["commit"]) == 40


def run_cli_study(tmp_path):
    """What `nanoscope study` trains, without its report: that plots with matplotlib, and a
    figure left for the garbage collector crashed the interpreter in a later test."""
    from nanoscope.study import load_study

    load_study(tmp_path / "ws" / "studies" / "toy.toml").run(devices=["cpu"])


def test_bundle_is_a_zip_of_the_evidence(client, tmp_path):
    import io
    import json
    import zipfile

    client.post("/api/studies", json={"toml": SPEC})
    assert client.get("/api/studies/toy/bundle.zip").status_code == 404
    run_cli_study(tmp_path)

    response = client.get("/api/studies/toy/bundle.zip")
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    assert 'filename="toy-bundle.zip"' in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = set(zf.namelist())
        assert {"report.md", "results.json", "spec.toml", "plan.json", "finals.json"} <= names
        finals = json.loads(zf.read("finals.json"))["finals"]
        assert sorted(finals) == ["small", "wide"] and sorted(finals["small"]) == ["0", "1"]
        assert "# Study: toy" in zf.read("report.md").decode()
        assert json.loads(zf.read("results.json"))["rows"][0]["label"] == "small"
        assert 'name = "toy"' in zf.read("spec.toml").decode()


def git(path, *args):
    import subprocess

    return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True,
                          text=True).stdout


def test_git_status_is_read_only_and_names_the_file(client, tmp_path):
    ws = tmp_path / "ws"
    assert client.get("/api/git/status").json() == {
        "repo": False, "root": None, "branch": None, "head": None, "clean": True,
        "changed": [], "identity": False, "path": None, "path_committed": None}

    git(ws, "init", "-q", "-b", "main")
    git(ws, "config", "user.email", "t@example.com")
    git(ws, "config", "user.name", "t")
    client.post("/api/studies", json={"toml": SPEC})
    status = client.get("/api/git/status", params={"path": "studies/toy.toml"}).json()
    assert status["repo"] and status["branch"] == "main" and status["head"] is None
    assert status["identity"] and not status["clean"] and status["changed"] == ["studies/toy.toml"]
    assert status["path_committed"] is False

    git(ws, "add", "."), git(ws, "commit", "-qm", "spec")
    status = client.get("/api/git/status", params={"path": "studies/toy.toml"}).json()
    assert status["clean"] and status["path_committed"] is True and len(status["head"]) == 40
    (ws / "notes.txt").write_text("x")
    dirty = client.get("/api/git/status", params={"path": "studies/toy.toml"}).json()
    assert not dirty["clean"] and dirty["changed"] == ["notes.txt"]
    assert dirty["path_committed"] is True  # the spec itself is unchanged
    assert client.get("/api/git/status", params={"path": "../x"}).status_code == 400


def test_preregister_card_and_push_go_through_jobs(client, tmp_path, monkeypatch):
    import json

    ws = tmp_path / "ws"
    git(ws, "init", "-q", "-b", "main")
    git(ws, "config", "user.email", "t@example.com")
    git(ws, "config", "user.name", "t")
    record = SPEC.replace("seeds = [0, 1]", 'seeds = [0, 1, 2]\nmode = "record"')
    client.post("/api/studies", json={"toml": record})

    preview_job = client.post("/api/studies/toy/preregister/preview")
    assert preview_job.status_code == 202, preview_job.text
    work_off(preview_job.json()["job"]["id"])
    done = queue.get(preview_job.json()["job"]["id"])
    preview = json.loads(done["result"])
    assert preview["files"] == ["studies/toy.toml"] and preview["unrelated"] == []
    assert "Preregister study toy" in preview["message"]
    assert "+++ b/studies/toy.toml" in preview["diff"]

    stale = client.post("/api/studies/toy/preregister/commit", json={"preview_hash": "0" * 16})
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit):
        main(["run-job", str(stale.json()["job"]["id"])])
    assert "preview changed" in queue.get(stale.json()["job"]["id"])["error"]

    commit_job = client.post("/api/studies/toy/preregister/commit",
                             json={"preview_hash": preview["hash"]})
    work_off(commit_job.json()["job"]["id"])
    sha = json.loads(queue.get(commit_job.json()["job"]["id"])["result"])["commit"]
    assert git(ws, "rev-parse", "HEAD").strip() == sha

    assert client.get("/api/studies/toy/card").status_code == 422  # not run yet
    run_cli_study(tmp_path)
    card = client.get("/api/studies/toy/card")
    assert card.status_code == 200, card.text
    body = card.json()
    assert body["mode"] == "record" and body["provenance"]["preregistration_commit"] == sha
    assert sorted(body["finals"]["wide"]) == ["0", "1", "2"]

    uploads = []

    class FakeApi:
        def upload_file(self, **kw):
            uploads.append(kw)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    bad = client.post("/api/studies/toy/card/push", json={"repo": "nonsense"})
    assert bad.status_code == 422 and uploads == []
    plan = client.get("/api/studies/toy/card/upload", params={"repo": "me/cards"}).json()
    assert plan["path_in_repo"] == "cards/toy.json"
    assert plan["commit_message"].startswith("nanoscope:")
    assert client.get("/api/studies/toy/card/upload", params={"repo": "bad"}).status_code == 422
    assert uploads == []  # a plan sends nothing
    pushed = client.post("/api/studies/toy/card/push",
                         json={"repo": "me/cards", "exported_at": plan["exported_at"]})
    assert pushed.status_code == 202, pushed.text
    assert pushed.json()["path_in_repo"] == "cards/toy.json" and uploads == []  # queued only
    work_off(pushed.json()["job"]["id"])
    assert len(uploads) == 1 and uploads[0]["repo_id"] == "me/cards"
    # what uploads is the text that was shown, byte for byte
    assert uploads[0]["path_or_fileobj"].decode() == plan["content"]


def test_explore_studies_have_no_card_and_unknown_studies_404(client):
    client.post("/api/studies", json={"toml": SPEC})
    refused = client.get("/api/studies/toy/card")
    assert refused.status_code == 422 and "record-mode studies only" in refused.json()["detail"]
    assert client.get("/api/studies/nope/card").status_code == 404
    assert client.post("/api/studies/nope/preregister/preview").status_code == 404
    assert client.post("/api/studies/toy/card/push", json={"repo": "me/c"}).status_code == 422


def test_sizes_job_resolves_the_match_and_flags_outliers(client):
    spec = SPEC.replace('baseline = "small"', 'baseline = "small"\ntolerance = 0.05')
    queued = client.post("/api/studies/sizes", json={"toml": spec})
    assert queued.status_code == 202, queued.text
    work_off(queued.json()["id"])
    table = json.loads(queue.get(queued.json()["id"])["result"])
    assert table["reference"] == "small" and table["tolerance"] == 0.05
    small, wide = table["variants"]
    assert small["delta"] == 0 and small["within"]
    assert wide["non_embedding_params"] > small["non_embedding_params"]
    assert wide["delta"] > 0.05 and not wide["within"]


def test_sizes_refuses_an_invalid_spec(client):
    broken = SPEC.replace('baseline = "small"', 'baseline = "nope"')
    bad = client.post("/api/studies/sizes", json={"toml": broken})
    assert bad.status_code == 422


def test_validate_returns_the_toml_and_saved_specs_read_back(client):
    verdict = client.post("/api/validate/study", json={"toml": SPEC, "workers_per_device": 2})
    assert verdict.status_code == 200 and verdict.json()["ok"]
    assert 'name = "toy"' in verdict.json()["toml"]
    client.post("/api/studies", json={"toml": SPEC})
    doc = client.get("/api/studies/toy/spec").json()
    assert doc["path"] == "studies/toy.toml" and doc["spec"]["seeds"] == [0, 1]
    assert doc["toml"].startswith("schema = 1")
    assert client.get("/api/studies/nope/spec").status_code == 404


PY_STUDY = '''\
from nanoscope.study import Study
from nanoscope.models import Bigram

study = Study("from-py", preset="test-tiny", seeds=[0, 1], baseline="small")
study.add("small", Bigram, d_model=8)
study.add("wide", Bigram, d_model=32)
'''


def test_a_python_study_opens_as_a_spec_through_a_worker(client, tmp_path):
    (tmp_path / "ws" / "mine.py").write_text(PY_STUDY)
    queued = client.post("/api/studies/from-file", json={"file": "mine.py"})
    assert queued.status_code == 202, queued.text
    work_off(queued.json()["id"])
    result = json.loads(queue.get(queued.json()["id"])["result"])
    assert result["spec"]["name"] == "from-py"
    assert [v["name"] for v in result["spec"]["variants"]] == ["small", "wide"]
    assert 'name = "from-py"' in result["toml"]
    assert client.post("/api/studies/from-file", json={"file": "x.toml"}).status_code == 422
    assert client.post("/api/studies/from-file", json={"file": "../x.py"}).status_code in (400, 404)
