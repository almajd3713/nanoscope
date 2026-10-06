import pytest
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
