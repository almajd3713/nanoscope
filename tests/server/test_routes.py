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
