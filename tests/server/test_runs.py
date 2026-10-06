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
