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

    assert run() == {"ok": True, "problems": []}
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
    assert ok == {"ok": True, "problems": []}
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
