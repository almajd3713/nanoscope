import json

import pytest
from fastapi.testclient import TestClient

from nanoscope import paths, queue
from nanoscope.cli import main
from nanoscope.learn import gating, unlocks
from nanoscope.server.app import create_app


@pytest.fixture
def client(home, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    gating.reload()
    return TestClient(create_app())


def test_lessons(client):
    paths_doc = client.get("/api/curricula").json()
    assert [p["id"] for p in paths_doc] == ["foundations", "modern-block"]
    foundations = paths_doc[0]
    assert foundations["level"] == 0 and foundations["compute"]["cpu"]["estimate_minutes"] == 14
    first = foundations["lessons"][0]
    assert first["id"] == "foundations/01-bigram" and first["state"] == "not-started"
    assert first["compute"]["cpu"] == {"preset": "tinystories-5min", "budget": None,
                                       "estimate_minutes": 3.0}
    second = foundations["lessons"][1]
    assert second["locked_by"] == ["foundations/01-bigram"]
    assert [lesson["unlocks"] for lesson in foundations["lessons"]][3:6] == [
        ["block:Attention"], ["block:Block"], ["block:Decoder"]]
    detail = client.get("/api/curricula/foundations/04-multi-head").json()
    assert detail["title"] == "Multi-head attention" and detail["unlocks"] == ["block:Attention"]
    assert detail["files"] == ["starter.py"] and detail["forbid"] == [
        "torch.nn.MultiheadAttention", "F.scaled_dot_product_attention"]
    assert detail["text"]["surface"].startswith("With `d_model = 16`")
    assert detail["text"]["reading"].startswith("Tiers: **B** build it")
    assert detail["text"]["deep"] and detail["text"]["intro"]
    assert [c["kind"] for c in detail["checks"]] == ["defines", "equivalent", "forbid"]
    assert detail["checks"][1]["reference"] == "naive_multi_head_attention"
    assert detail["workspace"] == "lessons/foundations/04-multi-head"
    assert client.get("/api/curricula/foundations/99-nope").status_code == 422


def test_start_never_overwrites_and_turns_gating_on(client, tmp_path):
    assert not unlocks.exists()
    started = client.post("/api/curricula/foundations/01-bigram/start")
    assert started.status_code == 200
    body = started.json()
    assert body["first_start"] is True and body["policy"] == "guided"
    assert body["copied"] == ["lessons/foundations/01-bigram/starter.py"] and body["kept"] == []
    assert body["state"] == "started" and unlocks.policy() == "guided"
    starter = tmp_path / "ws" / "lessons" / "foundations" / "01-bigram" / "starter.py"
    assert "class MyBigram" in starter.read_text()
    starter.write_text("# my work\n")
    again = client.post("/api/curricula/foundations/01-bigram/start").json()
    assert again["first_start"] is False and again["copied"] == [] and again["kept"] == [
        "lessons/foundations/01-bigram/starter.py"]
    assert starter.read_text() == "# my work\n"
    # the first start can ask for gating off
    unlocks_file = paths.learn_dir() / "unlocks.json"
    unlocks_file.unlink()
    opened = client.post("/api/curricula/foundations/02-mlp/start", json={"gating": "open"}).json()
    assert opened["policy"] == "open"
    unlocks_file.unlink()
    assert client.post("/api/curricula/foundations/02-mlp/start",
                       json={"gating": "wild"}).status_code == 422
    progress = client.get("/api/learn/progress").json()
    assert progress["schema"] == 1 and progress["lessons"]["foundations/01-bigram"][
        "state"] == "started"


def test_check_is_a_job(client, tmp_path, monkeypatch):
    from nanoscope.learn import checks
    from nanoscope.learn.checks import Result

    client.post("/api/curricula/foundations/04-multi-head/start")
    response = client.post("/api/curricula/foundations/04-multi-head/check")
    assert response.status_code == 202
    job = response.json()
    assert (job["kind"], job["lane"]) == ("check", "interactive")
    assert job["payload"] == {"lesson": "foundations/04-multi-head", "variant": "cpu"}
    # a worker runs it: the file the learner saved is judged, not the API's opinion of it
    (tmp_path / "ws" / "lessons" / "foundations" / "04-multi-head" / "starter.py").write_text(
        "import torch.nn as nn\n\nclass MultiHead(nn.Module):\n    pass\n")
    monkeypatch.setattr(checks, "CHECKERS", {
        k: (lambda ctx, c: Result(c.id, c.kind, True, "fine")) for k in checks.CHECKERS})
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job["id"])])
    assert stopped.value.code == 0
    result = json.loads(queue.get(job["id"])["result"])
    assert result["passed"] is True and result["lesson"] == "foundations/04-multi-head"
    progress = client.get("/api/learn/progress").json()["lessons"]["foundations/04-multi-head"]
    assert progress["state"] == "passed" and progress["attempts"] == 1
    assert client.post("/api/curricula/foundations/04-multi-head/check",
                       json={"variant": "gpu"}).status_code == 422


def test_unlocks(client):
    view = client.get("/api/learn/unlocks").json()
    assert view["policy"] == "open" and view["unlocks"] == {}
    assert view["first_run"] is True  # nobody has chosen yet
    assert view["lockable"]["block:Attention"] == {
        "lesson": "foundations/04-multi-head", "state": "open", "reason": None}
    assert len(view["lockable"]) == 9
    assert client.post("/api/learn/policy", json={"policy": "guided"}).json()["policy"] == "guided"
    view = client.get("/api/learn/unlocks").json()
    assert view["first_run"] is False
    assert {v["state"] for v in view["lockable"].values()} == {"locked"}
    # skipping one lesson needs a reason, and names the lock
    assert client.post("/api/learn/unlock", json={"id": "Attention"}).status_code == 422
    nope = client.post("/api/learn/unlock", json={"id": "Nope", "reason": "x"})
    assert nope.status_code == 422 and "'Nope' is not locked by any lesson" in nope.json()["detail"]
    skipped = client.post("/api/learn/unlock", json={"id": "Attention", "reason": "I know this"})
    state = skipped.json()["lockable"]["block:Attention"]
    assert skipped.status_code == 200
    assert state == {"lesson": "foundations/04-multi-head", "state": "skipped",
                     "reason": "I know this"}
    # earned unlocks stay earned through open and back to guided; "open" ones come and go
    unlocks.earn("modern-block/02-rope", ["block:RoPE"], "learn/checks/c.json")
    opened = client.post("/api/learn/policy", json={"policy": "open"}).json()
    assert opened["policy"] == "open"
    assert opened["lockable"]["block:RoPE"]["state"] == "earned"
    assert opened["lockable"]["feature:gqa"]["state"] == "open"
    back = client.post("/api/learn/policy", json={"policy": "guided"}).json()
    states = {k: v["state"] for k, v in back["lockable"].items()}
    assert states["block:RoPE"] == "earned" and states["block:Attention"] == "skipped"
    assert states["feature:gqa"] == "locked"
    assert client.post("/api/learn/policy", json={"policy": "wild"}).status_code == 422
    everything = client.post("/api/learn/unlock", json={"all": True}).json()
    assert everything["policy"] == "open"
    assert client.post("/api/learn/unlock", json={}).status_code == 422


def test_learn_events(home):
    import asyncio

    from nanoscope.server.routes.learn import learn_changes

    async def scenario():
        stop = asyncio.Event()
        stream = learn_changes(stop, interval_ms=500)

        async def change():
            for i in range(40):
                await asyncio.sleep(0.25)
                unlocks.set_policy("guided" if i % 2 else "open")
                from nanoscope.learn import progress
                progress.mark("foundations/01-bigram", "started")

        task = asyncio.create_task(change())
        seen = set()

        async def collect():
            async for frame in stream:
                if frame.startswith(":"):
                    continue
                event, data = frame.strip().split("\n")
                seen.add(event.removeprefix("event: "))
                assert json.loads(data.removeprefix("data: "))
                if seen == {"unlocks", "progress"}:
                    break

        await asyncio.wait_for(collect(), 20)
        stop.set()
        task.cancel()
        return seen

    assert asyncio.run(scenario()) == {"unlocks", "progress"}
