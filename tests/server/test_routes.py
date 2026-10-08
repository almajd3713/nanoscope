import pytest
from fakes import tiny
from fastapi.testclient import TestClient

from nanoscope.presets import get_preset, list_presets
from nanoscope.server.app import create_app


@pytest.fixture
def client(home):
    return TestClient(create_app())


def test_presets(client):
    listing = client.get("/api/presets").json()
    assert [p["name"] for p in listing] == list_presets()
    one = client.get("/api/presets/tinystories-5min")
    assert one.status_code == 200
    spec = one.json()
    fields = {f["name"]: f for f in spec["fields"]}
    assert fields["max_steps"]["default"] == get_preset("tinystories-5min").max_steps
    assert fields["precision"]["help"].startswith('"fp32"')
    assert fields["name"]["required"] is True
    missing = client.get("/api/presets/nope")
    assert missing.status_code == 404
    assert missing.json()["detail"].startswith("unknown preset 'nope'; available: ")
    assert missing.headers["content-type"] == "application/problem+json"


WORKSPACE_MODEL = '''\
import torch.nn as nn

from nanoscope.blocks import register_block


class Tiny(nn.Module):
    """A very small model."""

    def __init__(self, vocab_size: int, width: int = 8, tied=True):
        super().__init__()
        raise RuntimeError("the API must never run this")


@register_block
class AGate(nn.Module):
    def __init__(self, d_model, context_length):
        super().__init__()


class NotAModel:
    pass
'''


def test_models(client, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws" / "sub").mkdir(parents=True)
    (tmp_path / "ws" / "sub" / "tiny.py").write_text(WORKSPACE_MODEL)
    listing = client.get("/api/models").json()
    assert [m["name"] for m in listing] == ["Bigram", "GPT2", "Modern", "Tiny"]
    assert [m["shipped"] for m in listing] == [True, True, True, False]
    modern = client.get("/api/models/modern").json()
    assert modern["ref"] == "nanoscope.models.modern:Modern"
    params = {p["name"]: p for p in modern["params"]}
    assert params["vocab_size"]["from_data"] is True and params["vocab_size"]["required"]
    assert params["n_kv_heads"]["default"] == 2 and params["rope"]["annotation"] == "bool"
    assert client.get("/api/models/nanoscope.models.gpt2:GPT2").json()["name"] == "GPT2"
    tiny = client.get("/api/models/sub/tiny.py:Tiny").json()  # found by reading, never imported
    assert tiny["ref"] == "sub/tiny.py:Tiny" and tiny["source"] == "sub/tiny.py"
    assert tiny["doc"] == "A very small model." and tiny["shipped"] is False
    assert [(p["name"], p["annotation"], p["required"], p["default"], p["from_data"])
            for p in tiny["params"]] == [
        ("vocab_size", "int", True, None, True), ("width", "int", False, 8, False),
        ("tied", None, False, True, False)]
    missing = client.get("/api/models/nope")
    assert missing.status_code == 404 and "unknown model 'nope'" in missing.json()["detail"]


def test_describe_job(client, tmp_path, monkeypatch):
    import json

    from nanoscope import queue
    from nanoscope.cli import main

    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "tiny.py").write_text(
        "import torch.nn as nn\n\n\nclass Tiny(nn.Module):\n"
        "    def __init__(self, vocab_size: int, width: int = 8):\n"
        "        super().__init__()\n        self.emb = nn.Embedding(vocab_size, width)\n"
        "        self.out = nn.Linear(width, vocab_size)\n\n"
        "    def forward(self, idx):\n        return self.out(self.emb(idx))\n")
    response = client.post("/api/models/tiny.py:Tiny/describe",
                           json={"preset": "tinystories-5min", "kwargs": {"width": 4}})
    assert response.status_code == 202
    job = response.json()
    assert (job["kind"], job["lane"], job["state"]) == ("describe", "interactive", "queued")
    assert job["payload"]["model"] == f"{(tmp_path / 'ws').resolve() / 'tiny.py'}:Tiny"
    assert job["payload"]["kwargs"] == {"width": 4}
    # the API only queued it: a worker's run-job does the work
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job["id"])])
    assert stopped.value.code == 0
    result = json.loads(queue.get(job["id"])["result"])
    assert result["model"] == "Tiny" and result["kwargs"]["width"] == 4
    assert result["params"]["total"] == 4096 * 4 + 4 * 4096 + 4096
    # shipped models by name, defaults for an empty body
    shipped = client.post("/api/models/gpt2/describe")
    assert shipped.status_code == 202
    assert shipped.json()["payload"] == {
        "model": "nanoscope.models.gpt2:GPT2", "preset": "tinystories-5min", "kwargs": {}}
    assert client.post("/api/models/nope/describe").status_code == 404


def test_blocks_lock_state(client, tmp_path, monkeypatch):
    from nanoscope.learn import unlocks

    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "gate.py").write_text(
        "@register_block(reference=naive, family='mlp')\nclass Gate(nn.Module):\n"
        "    def __init__(self, d_model, context_length, hidden: int = 8):\n        pass\n")
    doc = client.get("/api/blocks").json()
    assert doc["policy"] == "open" and doc["schema"] == 1
    blocks = {b["name"]: b for b in doc["blocks"]}
    assert len(blocks) >= 20
    attn = blocks["Attention"]
    assert (attn["family"], attn["tier"], attn["certified"]) == ("attention", "composite", True)
    assert attn["lock"] == {"lockable": True, "locked": False,
                            "lesson": "foundations/04-multi-head",
                            "how": "open"}  # no unlocks.json: nothing is locked
    assert blocks["Linear"]["lock"]["lockable"] is False
    assert blocks["Linear"]["tier"] == "primitive"
    mine = blocks["Gate"]
    assert mine["user"] is True and mine["certified"] is False and mine["lock"]["locked"] is False
    # guided: composite blocks lock until earned
    unlocks.set_policy("guided")
    unlocks.earn("foundations/04-multi-head", ["block:Attention"], "learn/checks/c.json")
    doc = client.get("/api/blocks").json()
    blocks = {b["name"]: b for b in doc["blocks"]}
    assert doc["policy"] == "guided"
    assert blocks["Attention"]["lock"]["locked"] is False and blocks["Attention"]["lock"][
        "how"] == "earned"
    assert blocks["RoPE"]["lock"] == {"lockable": True, "locked": True,
                                      "lesson": "modern-block/02-rope", "how": None}
    assert blocks["RMSNorm"]["lock"]["locked"] is True
    assert blocks["GELUMLP"]["lock"]["locked"] is False  # primitives are never locked
    locked = sorted(n for n, b in blocks.items() if b["lock"]["locked"])
    assert locked == ["Block", "Decoder", "RMSNorm", "RoPE", "SwiGLU"]


def test_validate(client, tmp_path, monkeypatch):
    from nanoscope.learn import gating, unlocks
    from nanoscope.models import Modern  # noqa: F401

    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "mylm.py").write_text(
        "from nanoscope.blocks.attention import Attention\n"
        "from nanoscope.blocks.structure import Block, Decoder\n"
        "from nanoscope.blocks.norm import RMSNorm\nfrom nanoscope.blocks.mlp import SwiGLU\n\n\n"
        "class MyLM(Decoder):\n    def __init__(self, vocab_size: int, width: int = 8):\n"
        "        super().__init__(vocab_size, 8, d_model=16, n_layers=1, block=Block(\n"
        "            norm=RMSNorm(), attn=Attention(n_heads=4, n_kv_heads=2), mlp=SwiGLU()))\n")

    def run(**body):
        return client.post("/api/validate/run", json={"model": "modern", **body}).json()

    assert run() == {"ok": True, "problems": [], "refs": ["tinystories-5min/modern/seed-0"]}
    bad = run(preset="nope", kwargs={"n_kv_heads": "two", "colour": "red", "max_steps": "ten"},
              seeds=0)
    codes = {p["code"] for p in bad["problems"]}
    assert not bad["ok"] and codes == {"unknown_preset", "wrong_type", "unknown_keyword",
                                       "bad_seeds"}
    unknown = next(p for p in bad["problems"] if p["code"] == "unknown_keyword")
    assert unknown["field"] == "colour" and "neither a parameter of Modern.__init__" in unknown[
        "message"]
    assert next(p for p in bad["problems"] if p["field"] == "preset")["hint"].startswith(
        "available: ")
    assert client.post("/api/validate/run", json={"model": "nope"}).json()["problems"][0][
        "code"] == "unknown_model"
    # a workspace model is validated without being imported, locked uses included
    mine = run(model="mylm.py:MyLM", kwargs={"width": 4, "wdth": 4})
    assert [p["code"] for p in mine["problems"]] == ["unknown_keyword"]
    gating.reload()
    unlocks.set_policy("guided")
    locked = run(model="mylm.py:MyLM", kwargs={"width": 4})
    assert not locked["ok"] and {p["code"] for p in locked["problems"]} == {"locked"}
    assert any("feature:gqa" in p["message"] for p in locked["problems"])
    assert all("unlocks in the lesson" in p["hint"] for p in locked["problems"])

    def study(toml):
        return client.post("/api/validate/study", json={"toml": toml}).json()

    ok = study('name = "s"\nbaseline = "a"\n[[variants]]\nname = "a"\nmodel = "gpt2"\n'
               '[[variants]]\nname = "b"\nmodel = "modern"\n[variants.kwargs]\nn_kv_heads = 1\n')
    assert ok == {"ok": True, "problems": [], "refs": []}
    broken = study('name = "s"\nbaseline = "c"\nseeds = [0, "x"]\nmode = "wild"\n'
                   '[overrides]\nnope = 1\n[[variants]]\nname = "a"\nmodel = "gpt2"\n'
                   '[[variants]]\nname = "a"\nmodel = "modern"\n[variants.kwargs]\ncolour = 1\n'
                   '[[variants]]\nname = "z"\nmodel = "ghost"\n')
    fields = {(p["code"], p["field"]) for p in broken["problems"]}
    assert ("invalid_spec", "baseline") in fields and ("invalid_spec", "mode") in fields
    assert ("bad_seeds", "seeds") in fields and ("unknown_keyword", "overrides.nope") in fields
    assert ("unknown_keyword", "variants[1].kwargs.colour") in fields
    assert ("unknown_model", "variants[2].model") in fields
    assert any("variant name 'a' is used twice" in p["message"] for p in broken["problems"])
    typo = study('name = "s"\ncolour = "red"\n')
    assert typo["problems"][0]["code"] == "invalid_spec"
    assert "unknown study spec keys: colour" in typo["problems"][0]["message"]
    assert study("name = [")["problems"][0]["code"] == "invalid_spec"
    assert client.post("/api/validate/study", json={}).json()["problems"][0]["message"] == (
        "send exactly one of toml or spec")
    assert study('name = "s"\n')["problems"][0]["field"] == "variants"


@pytest.mark.usefixtures("fake_data")
def test_compare_equals_library(client):
    import nanoscope
    from nanoscope.models import Bigram
    from nanoscope.run import run

    a = run(Bigram, tiny(), device="cpu", seeds=3, progress=False)
    b = run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=8)
    expected = nanoscope.compare(a.ref, b.ref).to_dict()  # the same inputs: the refs
    response = client.post("/api/compare", json={"sets": [a.ref, b.ref]})
    assert response.status_code == 200
    body = response.json()
    assert body == expected  # the same document, key for key
    assert [r["verdict"] for r in body["rows"]][-1] == "baseline"
    assert len(body["curves"]) == 2 and body["precision_plan"]["n_seeds"] == 3
    # an explicit baseline, and the other metric
    with_baseline = client.post("/api/compare", json={
        "sets": [a.ref, b.ref], "baseline": a.ref, "metric": "val_loss"}).json()
    assert with_baseline == nanoscope.compare(
        a.ref, b.ref, baseline=a.ref, metric="val_loss").to_dict()
    assert with_baseline["baseline"] == body["rows"][0]["label"]
    # the library's words for a bad request, as a 422
    bad = client.post("/api/compare", json={"sets": [a.ref, b.ref], "metric": "accuracy"})
    assert bad.status_code == 422 and bad.json()["detail"] == (
        "metric must be one of val_bpb, val_loss")
    one = client.post("/api/compare", json={"sets": [a.ref]})
    assert one.status_code == 422 and "needs at least two sets" in one.json()["detail"]
    assert client.post("/api/compare", json={"sets": [a.ref, "nope/x"]}).status_code in (404, 422)


def test_data_and_hardware(client, monkeypatch):
    import json

    from nanoscope import paths
    from nanoscope.prepare import PrepareFile

    listing = client.get("/api/data").json()
    assert [d["preset"] for d in listing] == list_presets()
    assert all(d["prepare"] is None for d in listing)
    first = next(d for d in listing if d["preset"] == "tinystories-5min")
    assert first["dataset"] == "roneneldan/TinyStories" and first["vocab_size"] == 4096
    # a download in progress shows its stage
    prepare = PrepareFile("tinystories-5min")
    prepare.stage("tokenizing", "train", 40, 100)
    now = {d["preset"]: d for d in client.get("/api/data").json()}["tinystories-5min"]["prepare"]
    assert (now["stage"], now["done"], now["total"], now["label"]) == ("tokenizing", 40, 100,
                                                                      "train")
    prepare.finish()
    assert {d["preset"]: d for d in client.get("/api/data").json()}["tinystories-5min"][
        "prepare"]["stage"] == "done"
    job = client.post("/api/data/tinystories-5min/prepare")
    assert job.status_code == 202 and job.json()["kind"] == "prepare-data"
    assert job.json()["payload"] == {"preset": "tinystories-5min"}
    assert client.post("/api/data/nope/prepare").status_code == 404

    hardware = client.get("/api/hardware").json()
    assert hardware["devices"][0] == {"name": "cpu", "kind": "cpu", "memory_total": None,
                                      "memory_free": None}
    assert hardware["cpu_count"] >= 1 and hardware["torch"] and hardware["workers"] == []
    # a bench is a job, and its saved results are listed
    bench = client.post("/api/bench", json={"model": "bigram", "steps": 3, "preset": "test-tiny"})
    assert bench.status_code == 404  # test-tiny is not a registered preset here
    bench = client.post("/api/bench", json={"model": "bigram", "steps": 3})
    assert bench.status_code == 202
    assert bench.json()["payload"] == {"model": "nanoscope.models.bigram:Bigram",
                                       "preset": "tinystories-5min", "steps": 3}
    assert client.get("/api/hardware/bench").json() == []
    row = {"schema": 1, "at": "2026-10-06T10:00:00+00:00", "model": "Bigram", "device": "cpu",
           "step_ms": 12.5, "tokens_per_sec": 1000.0, "tflops": 0.01, "verdict": "CPU run",
           "preset": "tinystories-5min"}
    folder = paths.hardware_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bench.jsonl").write_text(json.dumps(row) + "\nnot json\n")
    rows = client.get("/api/hardware/bench").json()
    assert len(rows) == 1 and rows[0]["model"] == "Bigram" and rows[0]["schema"] == 1


def test_jobs(client):
    import json

    from nanoscope import paths, queue

    assert client.get("/api/jobs").json() == [] and client.get("/api/workers").json() == []
    a = client.post("/api/models/bigram/describe").json()
    b = client.post("/api/runs", json={"model": "bigram"}).json()["job"]
    c = client.post("/api/data/tinystories-5min/prepare").json()
    listing = client.get("/api/jobs").json()
    assert [j["id"] for j in listing] == [a["id"], b["id"], c["id"]]
    assert [j["id"] for j in client.get("/api/jobs?kind=run").json()] == [b["id"]]
    assert [j["id"] for j in client.get("/api/jobs?state=queued&limit=2").json()] == [
        b["id"], c["id"]]
    one = client.get(f"/api/jobs/{b['id']}").json()
    assert one["kind"] == "run" and one["state"] == "queued" and one["payload"]["model"].endswith(
        ":Bigram")
    assert one["result"] is None and one["error"] is None and one["attempts"] == 0
    cancelled = client.post(f"/api/jobs/{a['id']}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled"
    assert client.get("/api/jobs?state=cancelled").json()[0]["id"] == a["id"]
    # a running job is asked to stop; a finished one is left alone
    claimed = queue.claim("w", "cpu")
    assert claimed is not None
    running = client.post(f"/api/jobs/{claimed['id']}/cancel").json()
    assert running["state"] == "cancelling"
    queue.finish(claimed["id"], cancelled=True)
    assert client.post(f"/api/jobs/{claimed['id']}/cancel").json()["state"] == "cancelled"
    assert client.get("/api/jobs/999").status_code == 404
    assert client.post("/api/jobs/999/cancel").status_code == 404
    # workers show their heartbeat
    folder = paths.workers_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "w1.json").write_text(json.dumps({
        "schema": 1, "nanoscope": "0.2.0", "worker_id": "w1", "device": "cpu", "slots": 1,
        "jobs": [], "started_at": "2026-10-06T10:00:00+00:00",
        "heartbeat_at": "2026-10-06T10:00:05+00:00"}))
    workers = client.get("/api/workers").json()
    assert [w["worker_id"] for w in workers] == ["w1"] and "age" in workers[0]


@pytest.mark.usefixtures("fake_data")
def test_sync_hub(client, monkeypatch, tmp_path):
    import json
    import shutil

    from test_study import FakeHub

    from nanoscope import paths, queue
    from nanoscope.cli import main
    from nanoscope.models import Bigram
    from nanoscope.run import run

    hub = FakeHub(tmp_path / "hub")
    hub.root.mkdir()
    monkeypatch.setattr("huggingface_hub.HfApi", hub.api())
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub.snapshot_download)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    trained = run(Bigram, tiny(), device="cpu", push_to_hub="me/runs", progress=False)
    ref = trained.ref
    shutil.rmtree(trained.run_dir)  # "this machine" has nothing: the run lives on the Hub
    assert client.get(f"/api/runs/{ref}").status_code == 404
    response = client.post("/api/sync/hub", json={"repo": "me/runs", "ref": ref})
    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "sync-hub" and job["payload"] == {"repo": "me/runs", "ref": ref}
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job["id"])])
    assert stopped.value.code == 0
    result = json.loads(queue.get(job["id"])["result"])
    assert result == {"ref": ref, "repo": "me/runs", "pulled": True, "present": True}
    pulled = client.get(f"/api/runs/{ref}").json()
    assert pulled["summary"]["final_step"] == 20 and pulled["config"]["model"]["class"] == "Bigram"
    assert (paths.runs_dir() / ref / "latest.json").exists()
    # a second pull has nothing to do: the run is already here
    again = client.post("/api/sync/hub", json={"repo": "me/runs", "ref": ref}).json()
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit):
        main(["run-job", str(again["id"])])
    assert json.loads(queue.get(again["id"])["result"])["pulled"] is False
    # the token reaches only the jobs that need it
    from nanoscope.jobs.runner import job_env
    env = {"HF_TOKEN": "x", "WANDB_API_KEY": "y", "PATH": "/bin"}
    assert job_env("sync-hub", {}, env) == {"HF_TOKEN": "x", "PATH": "/bin"}
    assert job_env("check", {}, env) == {"PATH": "/bin"}


def test_schemas(client):
    from nanoscope import schemas

    listing = client.get("/api/schemas").json()
    assert listing == schemas.CURRENT and "status" in listing and "lesson" in listing
    one = client.get("/api/schemas/status").json()
    assert one == schemas.get("status") and one["$schema"].endswith("2020-12/schema")
    assert client.get("/api/schemas/status?version=1").json() == one
    unknown = client.get("/api/schemas/nope")
    assert unknown.status_code == 404 and "unknown schema 'nope'" in unknown.json()["detail"]
    assert client.get("/api/schemas/status?version=9").status_code == 404
    for name in listing:  # every published schema is served
        assert client.get(f"/api/schemas/{name}").status_code == 200


def test_validate_run_says_where_the_runs_land(client):
    body = {"model": "bigram", "preset": "tinystories-5min", "kwargs": {"d_model": 64}, "seeds": 2}
    ok = client.post("/api/validate/run", json=body).json()
    assert ok["ok"] is True and len(ok["refs"]) == 2
    assert ok["refs"][0].startswith("tinystories-5min/bigram-")
    assert ok["refs"][0].endswith("/seed-0")
    assert ok["refs"][1].endswith("/seed-1")
    # the names are the ones POST /api/runs gives
    queued = client.post("/api/runs",
                         json={"model": "bigram", "kwargs": {"d_model": 64}, "seed": 1})
    assert queued.json()["ref"] == ok["refs"][1]
    # defaults give the plain name; a list of seeds is taken as it is
    plain = client.post("/api/validate/run", json={"model": "bigram", "seeds": [3]}).json()
    assert plain["refs"] == ["tinystories-5min/bigram/seed-3"]
    # nothing is promised for a request with problems
    bad = client.post("/api/validate/run", json={"model": "bigram", "kwargs": {"lr": 1}}).json()
    assert bad["ok"] is False and bad["refs"] == []


def test_certify_block_lists_state_and_queues_a_job(client, tmp_path, monkeypatch):
    import json

    from nanoscope.blocks import certs

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(ws))
    source = (
        "@register_block(reference=naive, family='mlp')\nclass Gate(nn.Module):\n"
        "    def __init__(self, d_model, context_length):\n        pass\n\n\n"
        "@register_block\nclass Plain(nn.Module):\n"
        "    def __init__(self, d_model, context_length):\n        pass\n")
    (ws / "gate.py").write_text(source)

    def mine():
        blocks = {b["name"]: b for b in client.get("/api/blocks").json()["blocks"]}
        return blocks["Gate"]

    assert mine()["certification"] == {"state": "uncertified"} and mine()["certified"] is False
    job = client.post("/api/blocks/Gate/certify")
    assert job.status_code == 202
    assert job.json()["kind"] == "certify"
    assert job.json()["payload"] == {"file": str(ws / "gate.py"), "block": "Gate"}
    assert client.post("/api/blocks/Nope/certify").status_code == 404
    refused = client.post("/api/blocks/Plain/certify")
    assert refused.status_code == 422 and "no reference" in refused.json()["detail"]
    # a stored cert shows up, and goes stale when the file changes
    sha = certs.source_sha256(ws / "gate.py")
    certs.cert_path(sha).parent.mkdir(parents=True, exist_ok=True)
    certs.cert_path(sha).write_text(json.dumps({
        "schema": 1, "nanoscope": "0", "source_sha256": sha,
        "file": str((ws / "gate.py").resolve()),
        "results": {"Gate": {"passed": True, "message": "ok", "at": "t", "reference": "naive",
                             "tolerance": 1e-5, "trials": 3, "max_abs_diff": 0.0}}}))
    assert mine()["certified"] is True and mine()["certification"]["state"] == "certified"
    (ws / "gate.py").write_text(source + "# edited\n")
    assert mine()["certified"] is False and mine()["certification"]["state"] == "stale"
