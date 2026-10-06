import pytest
from fakes import tiny
from fastapi.testclient import TestClient

from nanoscope import run
from nanoscope.models import Bigram
from nanoscope.server.app import create_app

pytestmark = pytest.mark.usefixtures("fake_data")


@pytest.fixture
def client(home):
    return TestClient(create_app())


@pytest.fixture
def finished(home):
    """Two seeds of a tiny Bigram, trained for real (and a third run left half-way)."""
    return run(Bigram, tiny(), device="cpu", seeds=2, progress=False)


def test_list(client, finished):
    runs = client.get("/api/runs").json()
    assert [r["ref"] for r in runs] == [r.ref for r in finished]
    first = runs[0]
    assert (first["state"], first["step"], first["max_steps"]) == ("done", 20, 20)
    assert first["val_bpb"] > 0 and first["stale"] is False and first["error"] is None
    assert client.get("/api/runs?state=running").json() == []
    assert len(client.get("/api/runs?state=done").json()) == 2
    assert len(client.get(f"/api/runs?prefix={finished.ref}").json()) == 2
    assert client.get("/api/runs?prefix=elsewhere").json() == []
    detail = client.get(f"/api/runs/{runs[0]['ref']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["ref"] == runs[0]["ref"]
    assert body["config"]["schema"] == 1 and body["config"]["seed"] == 0
    assert body["config"]["model"]["class"] == "Bigram"
    assert body["status"]["state"] == "done" and body["status"]["schema"] == 1
    summary = body["summary"]
    assert summary["final_step"] == 20 and summary["n_evals"] == 2
    assert summary["final_val_bpb"] == pytest.approx(first["val_bpb"])
    assert summary["final_val_loss"] > 0 and summary["final_train_loss"] > 0
    missing = client.get("/api/runs/nope/seed-0")
    assert missing.status_code == 404 and "no run at ref 'nope/seed-0'" in missing.json()["detail"]
    assert client.get("/api/runs/../escape").status_code in (400, 404, 422)


def test_run_files(client, home):
    import json

    group = run(Bigram, tiny(), device="cpu", seeds=1, progress=False, checkpoint_steps=[10])
    ref = group[0].ref
    page = client.get(f"/api/runs/{ref}/metrics").json()
    assert [r["step"] for r in page["rows"]] == list(range(1, 21)) and page["last_step"] == 20
    assert page["rows"][9]["val_loss"] > 0 and "loss" in page["rows"][0]
    later = client.get(f"/api/runs/{ref}/metrics?since_step=15").json()
    assert [r["step"] for r in later["rows"]] == [16, 17, 18, 19, 20]
    assert client.get(f"/api/runs/{ref}/metrics?since_step=20").json() == {
        "rows": [], "last_step": 20}
    samples = client.get(f"/api/runs/{ref}/samples").json()
    assert [s["step"] for s in samples] == [10, 20] and isinstance(samples[0]["text"], str)
    assert client.get(f"/api/runs/{ref}/blockstats").json() == []  # not asked for
    checkpoints = client.get(f"/api/runs/{ref}/checkpoints").json()
    names = {c["name"]: c for c in checkpoints}
    assert set(names) == {"step_00000010.pt", "step_00000020.pt", "archive/step_00000010.pt"}
    assert names["step_00000020.pt"]["latest"] is True and names["step_00000010.pt"][
        "latest"] is False
    assert names["archive/step_00000010.pt"]["archived"] is True and names[
        "step_00000020.pt"]["bytes"] > 0 and names["step_00000020.pt"]["step"] == 20
    # block stats appear once a run records them
    watched = run(Bigram, tiny(), device="cpu", progress=False, block_stats=True, seed=1)
    stats = client.get(f"/api/runs/{watched.ref}/blockstats").json()
    assert [row["step"] for row in stats] == [10, 20] and stats[0]["blocks"][0]["name"] == "model"
    # the run itself is still reachable (the suffix routes do not shadow it)
    assert client.get(f"/api/runs/{ref}").status_code == 200
    json.dumps(page)
    for suffix in ("metrics", "samples", "blockstats", "checkpoints"):
        assert client.get(f"/api/runs/nope/seed-0/{suffix}").status_code == 404


def test_submit(client, monkeypatch, tmp_path):
    from learn_helpers import set_preset

    from nanoscope import paths, queue
    from nanoscope.cli import main
    from nanoscope.learn import gating, unlocks

    set_preset(monkeypatch, tiny())

    def submit(**body):
        return client.post("/api/runs", json={"preset": "test-tiny", **body})

    first = submit(model="bigram")
    assert first.status_code == 202
    body = first.json()
    assert body["ref"] == "test-tiny/bigram/seed-0" and body["state"] == "queued"
    job = body["job"]
    assert (job["kind"], job["lane"]) == ("run", "interactive") and job["ref"] == body["ref"]
    assert job["payload"] == {"model": "nanoscope.models.bigram:Bigram", "preset": "test-tiny",
                              "seed": 0, "kwargs": {}, "compile": False, "wandb": False}
    again = submit(model="bigram")  # the same run, still queued: the same job
    assert again.status_code == 202 and again.json()["job"]["id"] == job["id"]
    # a worker trains it, and the ref the API promised is where the library put it
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job["id"])])
    assert stopped.value.code == 0
    assert (paths.runs_dir() / "test-tiny" / "bigram" / "seed-0" / "status.json").exists()
    done = submit(model="bigram")
    assert done.status_code == 200
    assert done.json()["state"] == "done" and done.json()["job"] is None
    assert done.json()["run"]["status"]["state"] == "done"
    assert done.json()["run"]["summary"]["final_step"] == 20

    # keywords split into model parameters and preset overrides, and name a different run
    other = submit(model="bigram", seed=3, kwargs={"d_model": 8, "max_steps": 6}).json()
    assert other["ref"].startswith("test-tiny/bigram-") and other["ref"].endswith("/seed-3")
    assert other["job"]["payload"]["kwargs"] == {"d_model": 8}
    assert other["job"]["payload"]["overrides"] == {"max_steps": 6}
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit):
        main(["run-job", str(other["job"]["id"])])
    assert (paths.runs_dir() / other["ref"] / "status.json").exists()  # exactly where it said

    # every problem at once, with the library's wording
    bad = submit(model="bigram", kwargs={"colour": "red", "d_model": "wide"})
    assert bad.status_code == 422
    problems = bad.json()["problems"]
    assert {p["code"] for p in problems} == {"unknown_keyword", "wrong_type"}
    assert "'colour' is neither a parameter of Bigram.__init__" in bad.json()["detail"] or any(
        "neither a parameter" in p["message"] for p in problems)
    assert submit(model="nope").status_code == 404
    assert submit(model="bigram", kwargs={"max_steps": 0}).status_code in (202, 422)

    # a learner's model that uses something locked is refused, naming the lesson
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "mylm.py").write_text(
        "from nanoscope.blocks.attention import Attention\n"
        "from nanoscope.blocks.mlp import GELUMLP\nfrom nanoscope.blocks.norm import LayerNorm\n"
        "from nanoscope.blocks.structure import Block, Decoder\n\n\n"
        "class MyLM(Decoder):\n    def __init__(self, vocab_size: int):\n"
        "        super().__init__(vocab_size, 8, d_model=16, n_layers=1, block=Block(\n"
        "            norm=LayerNorm(), attn=Attention(n_heads=4, n_kv_heads=2), mlp=GELUMLP()))\n")
    gating.reload()
    assert submit(model="mylm.py:MyLM").status_code == 202  # open: nothing is locked
    unlocks.set_policy("guided")
    refused = submit(model="mylm.py:MyLM", seed=9)
    assert refused.status_code == 422 and refused.json()["type"] == "locked"
    assert refused.json()["lesson"] == "foundations/05-block"  # the first locked use
    assert set(refused.json()["lessons"]) == {
        "foundations/04-multi-head", "foundations/05-block", "modern-block/04-gqa"}
    assert "foundations/05-block" in refused.json()["detail"]
    assert not [j for j in queue.list_jobs() if j["ref"] and j["ref"].endswith("seed-9")]


def test_stop_resume(client, monkeypatch):
    import json

    from learn_helpers import set_preset

    from nanoscope import paths, queue
    from nanoscope.cli import main
    from nanoscope.status import STOP_FILE

    set_preset(monkeypatch, tiny(max_steps=40, eval_interval=10, checkpoint_interval=10))

    def submit(**body):
        return client.post("/api/runs", json={"preset": "test-tiny", "model": "bigram", **body})

    # a queued job that has not started is cancelled by stop
    queued = submit().json()
    stopped = client.post(f"/api/runs/{queued['ref']}/stop")
    assert stopped.status_code == 200
    assert stopped.json() == {"ref": queued["ref"], "stopping": [],
                              "cancelled_jobs": [queued["job"]["id"]]}
    assert queue.get(queued["job"]["id"])["state"] == "cancelled"

    # a run that stops half-way (a STOP file at step 20) can be resumed to the end
    ref = "test-tiny/bigram/seed-1"
    job = submit(seed=1).json()["job"]
    queue.cancel(job["id"])  # (the run is trained by hand below, as if started elsewhere)
    run_dir = paths.runs_dir() / ref
    run_dir.mkdir(parents=True, exist_ok=True)
    seen = []

    def stop_at_20(step, row):
        if step == 20 and not seen:
            seen.append(step)
            (run_dir / STOP_FILE).write_text("")
    from nanoscope import run as train
    from nanoscope.models import Bigram

    train(Bigram, tiny(max_steps=40, eval_interval=10, checkpoint_interval=10), seed=1,
          device="cpu", progress=False, output_dir=run_dir, on_step=stop_at_20)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "cancelled" and status["step"] == 20
    resumed = client.post(f"/api/runs/{ref}/resume")
    assert resumed.status_code == 202
    new_job = resumed.json()["job"]
    assert new_job["kind"] == "run" and new_job["ref"] == ref and new_job["state"] == "queued"
    assert new_job["payload"]["seed"] == 1 and new_job["payload"]["model"] == (
        "nanoscope.models.bigram:Bigram")
    assert new_job["payload"]["custom_preset"]["max_steps"] == 40
    again = client.post(f"/api/runs/{ref}/resume").json()["job"]
    assert again["id"] == new_job["id"]  # one job per run
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped_job:
        main(["run-job", str(new_job["id"])])
    assert stopped_job.value.code == 0
    detail = client.get(f"/api/runs/{ref}").json()
    assert detail["status"]["state"] == "done" and detail["summary"]["final_step"] == 40
    finished = client.post(f"/api/runs/{ref}/resume")
    assert finished.status_code == 409 and "nothing to resume" in finished.json()["detail"]
    assert client.post("/api/runs/nope/seed-0/stop").status_code == 404
    assert client.post("/api/runs/nope/seed-0/resume").status_code == 404
