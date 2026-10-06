"""The API end to end: real HTTP on a socket, real jobs, real files."""

import threading
import time

import httpx
import pytest
from fakes import tiny
from test_files import GRAPH_FILE
from test_tail import read_events

pytestmark = pytest.mark.usefixtures("fake_data")


def run_job_in_thread(job_id, delay=0.8):
    """What a worker does with a claimed job, after a pause so the client is connected."""
    from nanoscope import queue
    from nanoscope.cli import main

    def work():
        time.sleep(delay)
        queue.claim("w", "cpu")
        with pytest.raises(SystemExit):
            main(["run-job", str(job_id)])

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


def test_bigram_sse_compare(live_server, monkeypatch):
    from learn_helpers import set_preset

    import nanoscope

    set_preset(monkeypatch, tiny())
    url = live_server.url
    refs, threads = [], []
    for seed in (0, 1):
        sent = httpx.post(f"{url}/api/runs", json={
            "model": "bigram", "preset": "test-tiny", "seed": seed})
        assert sent.status_code == 202
        refs.append(sent.json()["ref"])
        threads.append(sent.json()["job"]["id"])
    assert refs == ["test-tiny/bigram/seed-0", "test-tiny/bigram/seed-1"]
    workers = [run_job_in_thread(job_id, delay=0.8 + 4 * i) for i, job_id in enumerate(threads)]

    def done(events):
        return any(e["event"] == "state" and e["data"]["state"] == "done" for e in events)

    with httpx.stream("GET", f"{url}/api/events", timeout=120) as r:
        events = read_events(r, lambda e: len({
            x["data"]["ref"] for x in e if x["event"] == "state"
            and x["data"]["state"] == "done"}) == 2, limit=5000)
    for worker in workers:
        worker.join(120)
    assert done(events)
    for ref in refs:  # the stream of one finished run replays its rows from `since_step`
        with httpx.stream("GET", f"{url}/api/runs/{ref}/events?since_step=0", timeout=30) as r:
            replay = read_events(r, lambda e: any(x["event"] == "eval" and x["data"]["step"] == 20
                                                  for x in e))
        assert [e["data"]["step"] for e in replay if e["event"] == "eval"] == [10, 20]
    # the comparison over HTTP is the library's, key for key
    body = httpx.post(f"{url}/api/compare", json={"sets": [refs[0], refs[1]]}).json()
    assert body == nanoscope.compare(refs[0], refs[1]).to_dict()
    assert len(body["rows"]) == 2


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(root))
    return root


def test_graph_roundtrip(live_server, workspace):
    url = live_server.url
    source = GRAPH_FILE.replace('        raise RuntimeError("the API must never run this")\n', "")
    saved = httpx.put(f"{url}/api/files/mylm.py", json={"content": source},
                      headers={"If-None-Match": "*"})
    assert saved.status_code in (200, 201), saved.text
    graph = httpx.post(f"{url}/api/files/mylm.py/graph").json()
    assert graph["classes"][0]["name"] == "MyLM"
    graph["classes"][0]["args"]["d_model"]["value"] = 64
    patched = httpx.post(f"{url}/api/files/mylm.py/graph/patch", json={"graph": graph},
                         headers={"If-Match": graph["etag"]})
    assert patched.status_code == 200, patched.text
    text = httpx.get(f"{url}/api/files/mylm.py").json()["content"]
    # only the edited argument changed, and the comment survived
    assert text == source.replace("d_model=32", "d_model=64")
    assert "# keep this comment" in text
    again = httpx.post(f"{url}/api/files/mylm.py/graph").json()
    assert again["classes"][0]["args"]["d_model"]["value"] == 64


def test_cli_run_in_events(live_server):
    from nanoscope import paths, run
    from nanoscope.models import Bigram

    def start():  # `nanoscope run` writes files and tells the server nothing
        time.sleep(1.0)
        run(Bigram, tiny(), device="cpu", progress=False,
            output_dir=paths.runs_dir() / "cli" / "seed-0")

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    with httpx.stream("GET", f"{live_server.url}/api/events", timeout=60) as r:
        events = read_events(r, lambda e: any(
            x["data"].get("ref") == "cli/seed-0" and x["event"] == "state"
            and x["data"]["state"] == "done" for x in e), limit=3000)
    thread.join(60)
    assert {e["data"]["ref"] for e in events} == {"cli/seed-0"}
    with httpx.stream("GET", f"{live_server.url}/api/runs/cli/seed-0/events?since_step=0",
                      timeout=30) as r:
        replay = read_events(r, lambda e: any(x["event"] == "eval" and x["data"]["step"] == 20
                                              for x in e))
    assert [e["data"]["step"] for e in replay if e["event"] == "eval"] == [10, 20]


LOCKED_MODEL = (
    "from nanoscope.blocks.attention import Attention\n"
    "from nanoscope.blocks.mlp import GELUMLP\nfrom nanoscope.blocks.norm import LayerNorm\n"
    "from nanoscope.blocks.structure import Block, Decoder\n\n\n"
    "class MyLM(Decoder):\n    def __init__(self, vocab_size: int):\n"
    "        super().__init__(vocab_size, 8, d_model=16, n_layers=1, block=Block(\n"
    "            norm=LayerNorm(), attn=Attention(n_heads=4), mlp=GELUMLP()))\n")


def test_locked_422(live_server, workspace, monkeypatch):
    from learn_helpers import set_preset

    from nanoscope import queue
    from nanoscope.learn import gating, unlocks

    set_preset(monkeypatch, tiny())
    url = live_server.url
    (workspace / "mylm.py").write_text(LOCKED_MODEL)
    gating.reload()
    unlocks.set_policy("guided")
    refused = httpx.post(f"{url}/api/runs", json={"model": "mylm.py:MyLM",
                                                  "preset": "test-tiny"})
    assert refused.status_code == 422 and refused.json()["type"] == "locked"
    assert refused.json()["lesson"] in refused.json()["detail"]
    assert queue.list_jobs() == []
    # a patch that adds a locked feature is refused the same way, and writes nothing
    unlocks.earn("foundations/04-multi-head", ["block:Attention"], "e")
    unlocks.earn("foundations/05-block", ["block:Block"], "e")
    unlocks.earn("foundations/06-gpt2", ["block:Decoder"], "e")
    graph = httpx.post(f"{url}/api/files/mylm.py/graph").json()
    gqa = {"op": "set_arg", "class": "MyLM", "path": ["block", "attn"], "arg": "n_kv_heads",
           "value": {"kind": "literal", "value": 1, "span": None}}
    patch = httpx.post(f"{url}/api/files/mylm.py/graph/patch", json={"edits": [gqa]},
                       headers={"If-Match": graph["etag"]})
    assert patch.status_code == 422 and patch.json()["type"] == "locked"
    assert patch.json()["lesson"] == "modern-block/04-gqa"
    assert (workspace / "mylm.py").read_text() == LOCKED_MODEL
