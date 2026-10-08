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


def test_inference_keeps_an_lru_of_loaded_models(home, monkeypatch):
    """The inference process loads a run once, reuses it, reloads after more training, and
    drops the least recently used model when it holds too many."""
    import multiprocessing
    import threading

    from nanoscope.jobs import inference

    loads = []

    class Loaded:
        def __init__(self, ref):
            self.ref = ref

        def generate(self, prompt, n, temperature, seed):
            return f"{self.ref}:{prompt}:{n}:{temperature}:{seed}"

    def load(ref):
        loads.append(ref)
        return Loaded(ref)

    stamps = {"a": 1, "b": 1, "c": 1}
    monkeypatch.setattr(inference, "_checkpoint_stamp", lambda ref: stamps[ref])
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(target=inference.serve, args=(child, 2, load), daemon=True)
    thread.start()

    def ask(ref, **kw):
        parent.send({"ref": ref, "prompt": "hi", "max_new_tokens": 3, **kw})
        return parent.recv()

    assert ask("a")["text"] == "a:hi:3:0.8:42" and loads == ["a"]
    assert ask("a", temperature=0.5, seed=7)["text"] == "a:hi:3:0.5:7" and loads == ["a"]  # reused
    ask("b")
    assert loads == ["a", "b"] and ask("b")["loaded"] == ["a", "b"]
    ask("c")  # a third model: the least recently used one (a) is dropped
    assert ask("c")["loaded"] == ["b", "c"] and loads == ["a", "b", "c"]
    ask("a")
    assert loads == ["a", "b", "c", "a"]  # reloaded
    stamps["a"] = 2  # more training happened: the old copy is replaced
    ask("a")
    assert loads[-1] == "a" and len(loads) == 5
    parent.send({"ref": "zzz"})  # a failure is an answer, not a crash
    assert "KeyError" in parent.recv()["error"]
    parent.send(None)
    thread.join(5)
    assert not thread.is_alive()


SLOW_MODEL = '''\
import os
import time

import torch.nn as nn


class Slow(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        if os.path.exists(os.environ["SLOW_FLAG"]):
            time.sleep(0.4)
        return self.head(self.emb(idx))
'''


@pytest.fixture
def worker_thread(home):
    """A real worker ticking in a thread: generate jobs go to its inference process."""
    import threading
    import time

    from nanoscope.jobs.worker import Worker

    worker = Worker("cpu", 1, poll_seconds=0.05)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            worker.tick()
            time.sleep(0.05)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield worker
    stop.set()
    thread.join(10)
    if worker._pool is not None:
        worker._pool.close()


def test_generate(client, worker_thread, tmp_path, monkeypatch):
    from nanoscope import queue
    from nanoscope.cli import _load_model_class

    monkeypatch.setenv("SLOW_FLAG", str(tmp_path / "slow.flag"))
    model_file = tmp_path / "slow_model.py"
    model_file.write_text(SLOW_MODEL)
    trained = run(_load_model_class(f"{model_file}:Slow"), tiny(), device="cpu", progress=False)
    ref = trained.ref
    first = client.post(f"/api/runs/{ref}/generate", json={"prompt": "Once", "max_new_tokens": 8})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["ref"] == ref and isinstance(body["text"], str) and body["job_id"] > 0
    job = queue.get(body["job_id"])
    assert (job["kind"], job["lane"], job["state"]) == ("generate", "interactive", "done")
    again = client.post(f"/api/runs/{ref}/generate", json={"prompt": "Once", "max_new_tokens": 8})
    assert again.json()["text"] == body["text"]  # same seed, same text, from the loaded model
    assert client.post(f"/api/runs/{ref}/generate",
                       json={"max_new_tokens": 0}).status_code == 422
    assert client.post("/api/runs/nope/seed-0/generate", json={}).status_code == 404
    # a generation that outruns its timeout is stopped
    (tmp_path / "slow.flag").write_text("")
    slow = client.post(f"/api/runs/{ref}/generate",
                       json={"prompt": "x", "max_new_tokens": 30, "timeout": 1.5})
    assert slow.status_code == 504, slow.text
    assert "did not finish in 1.5s" in slow.json()["detail"]
    # the worker recovers: the next call (fast again) loads the model afresh and works
    (tmp_path / "slow.flag").unlink()
    ok = client.post(f"/api/runs/{ref}/generate", json={"prompt": "Once", "max_new_tokens": 4})
    assert ok.status_code == 200


def test_generate_without_a_worker_is_a_503(client, monkeypatch):
    from nanoscope import queue
    from nanoscope.server.routes import runs as runs_route

    monkeypatch.setattr(runs_route, "QUEUE_WAIT", 0.3)
    group = run(Bigram, tiny(), device="cpu", seeds=1, progress=False)
    response = client.post(f"/api/runs/{group[0].ref}/generate", json={})
    assert response.status_code == 503 and "start one with `nanoscope worker`" in response.json()[
        "detail"]
    assert [j["state"] for j in queue.list_jobs()] == ["cancelled"]


def test_baseline_range(client, finished, home):
    import json
    import shutil

    from nanoscope import paths
    from nanoscope.compare import BASELINES_DIR
    from nanoscope.statistics import reproduction_interval

    # a run with no shipped baseline for its preset and model says so
    assert client.get(f"/api/runs/{finished[0].ref}").json()["baseline"] is None

    # a GPT-2 run on the baseline preset carries the shipped seeds' range for one new run
    source = BASELINES_DIR / "tinystories-5min" / "gpt2" / "seed-0"
    mine = paths.runs_dir() / "tinystories-5min" / "gpt2" / "seed-0"
    shutil.copytree(source, mine)
    body = client.get("/api/runs/tinystories-5min/gpt2/seed-0").json()["baseline"]
    values = [json.loads(line) for line in (mine / "metrics.jsonl").read_text().splitlines()]
    mine_final = [r["val_bpb"] for r in values if "val_bpb" in r][-1]
    shipped = [
        [json.loads(x) for x in (BASELINES_DIR / "tinystories-5min" / "gpt2" / f"seed-{i}"
                                 / "metrics.jsonl").read_text().splitlines()]
        for i in range(3)]
    finals = [[r["val_bpb"] for r in rows if "val_bpb" in r][-1] for rows in shipped]
    low, high = reproduction_interval(finals)
    assert body["ref"] == "baselines/tinystories-5min/gpt2" and body["n_seeds"] == 3
    assert body["metric"] == "val_bpb" and body["values"] == pytest.approx(finals)
    assert body["interval"] == pytest.approx([low, high])
    assert body["value"] == pytest.approx(mine_final) and body["inside"] is True

    # a result far outside the range says so
    rows = (mine / "metrics.jsonl").read_text().splitlines()
    last = max(i for i, line in enumerate(rows) if "val_bpb" in json.loads(line))
    changed = json.loads(rows[last])
    changed["val_bpb"] = high + 1.0
    rows[last] = json.dumps(changed)
    (mine / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    outside = client.get("/api/runs/tinystories-5min/gpt2/seed-0").json()["baseline"]
    assert outside["inside"] is False and outside["value"] == pytest.approx(high + 1.0)
