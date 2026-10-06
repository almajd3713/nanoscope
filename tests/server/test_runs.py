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
