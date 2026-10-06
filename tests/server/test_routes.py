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
