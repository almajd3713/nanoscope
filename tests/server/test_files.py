import os

import pytest
from fastapi.testclient import TestClient

from nanoscope.server.app import create_app


@pytest.fixture
def ws(tmp_path, monkeypatch, home):
    root = tmp_path / "ws"
    (root / "models").mkdir(parents=True)
    (root / "models" / "my_lm.py").write_text("print('hi')\n")
    (root / "notes.md").write_text("# notes\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("x")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (root / "blob.bin").write_bytes(b"\xff\xfe\x00")
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(root))
    return root


@pytest.fixture
def client(ws):
    return TestClient(create_app())


def test_read(client, ws):
    listing = client.get("/api/files").json()
    assert [f["path"] for f in listing] == ["blob.bin", "models/my_lm.py", "notes.md"]
    assert [f["path"] for f in client.get("/api/files?glob=**/*.py").json()] == ["models/my_lm.py"]
    response = client.get("/api/files/models/my_lm.py")
    body = response.json()
    assert response.status_code == 200 and body["content"] == "print('hi')\n"
    assert body["path"] == "models/my_lm.py"
    assert response.headers["etag"] == body["etag"] == listing[1]["etag"]
    assert client.get("/api/files/models/my_lm.py",
                      headers={"If-None-Match": body["etag"]}).status_code == 304
    missing = client.get("/api/files/nope.py")
    assert missing.status_code == 404 and "no file 'nope.py'" in missing.json()["detail"]
    binary = client.get("/api/files/blob.bin")
    assert binary.status_code == 415 and "not a UTF-8 text file" in binary.json()["detail"]
    # the ETag changes with the content
    (ws / "models" / "my_lm.py").write_text("print('changed')\n")
    assert client.get("/api/files/models/my_lm.py").json()["etag"] != body["etag"]


def test_traversal(client, ws, tmp_path):
    (tmp_path / "secret.txt").write_text("outside")
    for attempt in ("../secret.txt", "models/../../secret.txt", "%2e%2e/secret.txt",
                    "/etc/passwd", "C:/Windows/win.ini"):
        response = client.get(f"/api/files/{attempt}")
        assert response.status_code in (400, 404), attempt
        assert "outside" not in response.text
    # a symlink out of the workspace is refused as well
    os.symlink(tmp_path / "secret.txt", ws / "link.txt")
    link = client.get("/api/files/link.txt")
    assert link.status_code == 400 and "points outside the workspace" in link.json()["detail"]
    assert "link.txt" not in [f["path"] for f in client.get("/api/files").json()]


def test_write_and_conflict_409(client, ws):
    path = "/api/files/models/my_lm.py"
    first = client.get(path).json()
    # a new file needs no ETag; its parent folders are made
    created = client.put("/api/files/new/dir/x.py", json={"content": "x = 1\n"})
    assert created.status_code == 201 and (ws / "new" / "dir" / "x.py").read_text() == "x = 1\n"
    assert created.json()["etag"] == created.headers["etag"]
    # an existing file must say what it read
    bare = client.put(path, json={"content": "print('mine')\n"})
    assert bare.status_code == 428 and bare.json()["current_etag"] == first["etag"]
    assert (ws / "models" / "my_lm.py").read_text() == "print('hi')\n"  # untouched
    saved = client.put(path, json={"content": "print('mine')\n"},
                       headers={"If-Match": first["etag"]})
    assert saved.status_code == 200 and saved.json()["etag"] != first["etag"]
    assert (ws / "models" / "my_lm.py").read_text() == "print('mine')\n"
    # someone else (an editor, git checkout) changed it meanwhile: 409 with a diff, no write
    (ws / "models" / "my_lm.py").write_text("print('theirs')\n")
    conflict = client.put(path, json={"content": "print('mine, again')\n"},
                          headers={"If-Match": saved.json()["etag"]})
    assert conflict.status_code == 409
    problem = conflict.json()
    assert problem["current_etag"] == client.get(path).json()["etag"]
    assert "--- models/my_lm.py (on disk)" in problem["diff"]
    assert "-print('theirs')" in problem["diff"] and "+print('mine, again')" in problem["diff"]
    assert (ws / "models" / "my_lm.py").read_text() == "print('theirs')\n"
    # no temp files are left behind, and the traversal guard covers writes too
    assert not [p for p in (ws / "models").iterdir() if p.name.endswith(".tmp")]
    assert client.put("/api/files/../escape.py", json={"content": "x"}).status_code in (400, 404)
    assert client.put("/api/files/%2e%2e/escape.py", json={"content": "x"}).status_code == 400


def test_external_edit_event(ws):
    """Whatever changes a workspace file (an editor, git, another process) is announced."""
    import asyncio
    import json

    from nanoscope.server.routes.files import watch_events

    def parse(frame):
        event, data = frame.strip().split("\n")
        return event.removeprefix("event: "), json.loads(data.removeprefix("data: "))

    async def scenario():
        stop = asyncio.Event()
        stream = watch_events(stop, interval_ms=500)

        async def touch():
            # the watcher needs a moment to start: keep changing things until it has seen them
            for i in range(40):
                await asyncio.sleep(0.25)
                (ws / "models" / "my_lm.py").write_text(f"print('edited by hand {i}')\n")
                (ws / "fresh.py").write_text(f"x = {i}\n")
                (ws / "__pycache__" / "junk.pyc").write_bytes(b"1")  # ignored
                (ws / "notes.md").write_text("# back\n")
                await asyncio.sleep(0.15)
                (ws / "notes.md").unlink()

        task = asyncio.create_task(touch())
        seen = {}

        async def collect():
            async for frame in stream:
                if frame.startswith(":"):
                    continue
                event, data = parse(frame)
                assert event == "change"
                seen[data["path"]] = data
                kinds = {d["kind"] for d in seen.values()}
                if {"models/my_lm.py", "fresh.py", "notes.md"} <= set(seen) and "deleted" in kinds:
                    break

        await asyncio.wait_for(collect(), 20)  # asyncio.timeout needs Python 3.11
        stop.set()
        task.cancel()
        return seen

    seen = asyncio.run(scenario())
    assert seen["models/my_lm.py"]["kind"] == "modified" and seen["models/my_lm.py"]["etag"]
    assert seen["fresh.py"]["kind"] in ("added", "modified")
    assert seen["notes.md"]["kind"] in ("deleted", "added", "modified")
    assert "__pycache__/junk.pyc" not in seen


def test_the_events_route_is_not_shadowed_by_the_file_route(client):
    spec = client.get("/api/openapi.json").json()["paths"]
    assert "/api/files/events" in spec and "/api/files/{path}" in spec
    assert list(spec).index("/api/files/events") < list(spec).index("/api/files/{path}")


GRAPH_FILE = '''\
from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, SwiGLU


class MyLM(Decoder):
    def __init__(self, vocab_size: int):
        raise RuntimeError("the API must never run this")
        super().__init__(
            vocab_size, 64, d_model=32, n_layers=2,  # keep this comment
            block=Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=SwiGLU()),
        )
'''


def test_graph(client, ws):
    (ws / "mylm.py").write_text(GRAPH_FILE.replace(
        '        raise RuntimeError("the API must never run this")\n', ""))
    graph = client.post("/api/files/mylm.py/graph").json()
    assert graph["path"] == "mylm.py" and graph["etag"] == client.get("/api/files/mylm.py").json()[
        "etag"]
    cls = graph["classes"][0]
    assert cls["name"] == "MyLM" and cls["args"]["d_model"]["value"] == 32
    # a file that raises on import is still readable: nothing is imported
    (ws / "boom.py").write_text("raise RuntimeError('never')\n" + GRAPH_FILE)
    assert client.post("/api/files/boom.py/graph").status_code == 200
    assert client.post("/api/files/nope.py/graph").status_code == 404

    # patch by edited graph: one argument changes, the comment survives
    edited = {**graph}
    edited["classes"][0]["args"]["d_model"]["value"] = 64
    etag = graph["etag"]
    patched = client.post("/api/files/mylm.py/graph/patch", json={"graph": edited},
                          headers={"If-Match": etag})
    assert patched.status_code == 200, patched.text
    text = (ws / "mylm.py").read_text()
    assert "d_model=64, n_layers=2,  # keep this comment" in text
    assert patched.json()["classes"][0]["args"]["d_model"]["value"] == 64
    assert patched.json()["etag"] != etag and patched.headers["etag"] == patched.json()["etag"]
    # patch by explicit edits
    swap = {"op": "set_arg", "class": "MyLM", "path": ["block", "attn"], "arg": "n_heads",
            "value": {"kind": "literal", "value": 4, "span": None}}
    again = client.post("/api/files/mylm.py/graph/patch", json={"edits": [swap]},
                        headers={"If-Match": patched.json()["etag"]})
    assert again.status_code == 200 and "Attention(n_heads=4)" in (ws / "mylm.py").read_text()
    # the same preconditions as a save
    assert client.post("/api/files/mylm.py/graph/patch", json={"edits": [swap]}).status_code == 428
    stale = client.post("/api/files/mylm.py/graph/patch", json={"edits": [swap]},
                        headers={"If-Match": etag})
    assert stale.status_code == 409 and stale.json()["current_etag"]
    both = client.post("/api/files/mylm.py/graph/patch", json={"edits": [], "graph": {}},
                       headers={"If-Match": again.json()["etag"]})
    assert both.status_code == 422 and "exactly one of graph" in both.json()["detail"]
    bad = client.post("/api/files/mylm.py/graph/patch", headers={"If-Match": again.json()["etag"]},
                      json={"edits": [{"op": "set_arg", "class": "Ghost", "path": [],
                                       "arg": "x", "value": {"kind": "literal", "value": 1}}]})
    assert bad.status_code == 422
    assert "no Decoder or Composite class 'Ghost'" in bad.json()["detail"]


def test_graph_patch_refuses_locked_blocks_under_guided(client, ws):
    from nanoscope.learn import gating, unlocks

    (ws / "mylm.py").write_text(GRAPH_FILE.replace(
        '        raise RuntimeError("the API must never run this")\n', "").replace(
        "from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, SwiGLU",
        "from nanoscope.blocks.attention import Attention\n"
        "from nanoscope.blocks.mlp import SwiGLU\nfrom nanoscope.blocks.norm import RMSNorm\n"
        "from nanoscope.blocks.structure import Block, Decoder"))
    gating.reload()
    unlocks.set_policy("guided")
    unlocks.earn("foundations/04-multi-head", ["block:Attention"], "e")
    unlocks.earn("foundations/05-block", ["block:Block"], "e")
    unlocks.earn("foundations/06-gpt2", ["block:Decoder"], "e")
    unlocks.earn("modern-block/01-rmsnorm", ["block:RMSNorm"], "e")
    unlocks.earn("modern-block/03-swiglu", ["block:SwiGLU"], "e")
    graph = client.post("/api/files/mylm.py/graph").json()
    before = (ws / "mylm.py").read_text()
    rope = {"op": "set_arg", "class": "MyLM", "path": ["block", "attn"], "arg": "pos",
            "value": {"kind": "block", "block": "RoPE", "args": {}, "span": None}}
    refused = client.post("/api/files/mylm.py/graph/patch", json={"edits": [rope]},
                          headers={"If-Match": graph["etag"]})
    assert refused.status_code == 422 and refused.json()["type"] == "locked"
    assert refused.json()["lesson"] == "modern-block/02-rope"
    assert "modern-block/02-rope" in refused.json()["detail"]
    assert (ws / "mylm.py").read_text() == before  # nothing written
    gqa = {"op": "set_arg", "class": "MyLM", "path": ["block", "attn"], "arg": "n_kv_heads",
           "value": {"kind": "literal", "value": 1, "span": None}}
    refused = client.post("/api/files/mylm.py/graph/patch", json={"edits": [gqa]},
                          headers={"If-Match": graph["etag"]})
    assert refused.status_code == 422 and refused.json()["unlock_id"] == "feature:gqa"
    # once earned, the same patch goes through
    unlocks.earn("modern-block/04-gqa", ["feature:gqa"], "e")
    ok = client.post("/api/files/mylm.py/graph/patch", json={"edits": [gqa]},
                     headers={"If-Match": graph["etag"]})
    assert ok.status_code == 200 and "n_kv_heads=1" in (ws / "mylm.py").read_text()


def test_lint(client, ws):
    (ws / "bad.py").write_text(
        "import os\nimport torch.nn as nn\n\nclass M(nn.Module):\n    def f(self):\n"
        "        x = 1\n        raise RuntimeError('never run')\n")
    found = client.post("/api/files/bad.py/lint").json()
    by_code = {d["code"]: d for d in found}
    assert set(by_code) == {"F401", "F841"}
    assert by_code["F401"]["message"] == "`os` imported but unused" and by_code["F401"][
        "line"] == 1 and by_code["F401"]["fixable"] is True
    assert by_code["F841"]["line"] == 6 and by_code["F841"]["column"] == 9
    # unsaved editor text, and a syntax error
    live = client.post("/api/files/bad.py/lint", json={"content": "def (:\n"}).json()
    assert live and {d["code"] for d in live} == {"invalid-syntax"} and live[0]["line"] == 1
    clean = client.post("/api/files/models/my_lm.py/lint")
    assert clean.status_code == 200 and clean.json() == []
    assert client.post("/api/files/nope.py/lint").status_code == 404
    # nothing was executed or written
    assert (ws / "bad.py").read_text().count("never run") == 1


TEMPLATE_FILE = '''\
from nanoscope.blocks import AttentionTemplate, BlockTemplate, CausalMask, Decoder, Linear
from nanoscope.blocks import RMSNorm, ScaledDotScores, Softmax, SwiGLU, WeightedSum


class Fill(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(
            vocab_size, 16, d_model=32, n_layers=2,
            block=BlockTemplate(
                norm1=RMSNorm(),
                attn=AttentionTemplate(
                    q=Linear(), k=Linear(), v=Linear(), scores=ScaledDotScores(), mask=None,
                    normalize=Softmax(), mix=WeightedSum(), out=Linear()),
                norm2=RMSNorm(), mlp=SwiGLU()),
        )
'''


def test_graph_patch_all_ops(client, ws):
    from nanoscope.learn import gating, unlocks

    def block(name, **args):
        return {"kind": "block", "block": name, "args": args, "span": None}

    def lit(value):
        return {"kind": "literal", "value": value, "span": None}

    layer = block("Block", norm=block("RMSNorm"), attn=block("Attention", n_heads=lit(2)),
                  mlp=block("SwiGLU"))
    (ws / "mylm.py").write_text(GRAPH_FILE.replace(
        '        raise RuntimeError("the API must never run this")\n', ""))
    etag = client.post("/api/files/mylm.py/graph").json()["etag"]

    def patch(edit, path="mylm.py"):
        nonlocal etag
        response = client.post(f"/api/files/{path}/graph/patch", json={"edits": [edit]},
                               headers={"If-Match": etag})
        if response.status_code == 200:
            etag = response.json()["etag"]
        return response

    def args():
        return client.post("/api/files/mylm.py/graph").json()["classes"][0]["args"]

    assert patch({"op": "add_layer", "class": "MyLM"}).status_code == 200
    assert args()["n_layers"]["value"] == 3
    assert patch({"op": "remove_layer", "class": "MyLM"}).status_code == 200
    assert args()["n_layers"]["value"] == 2
    done = patch({"op": "set_pattern", "class": "MyLM", "items": [layer, layer]})
    assert done.status_code == 200 and "block" not in args()
    assert len(args()["pattern"]["items"]) == 2
    assert patch({"op": "add_layer", "class": "MyLM", "path": ["pattern"], "node": layer,
                  "index": 1}).status_code == 200
    assert len(args()["pattern"]["items"]) == 3
    assert patch({"op": "remove_layer", "class": "MyLM", "path": ["pattern"],
                  "index": 0}).status_code == 200
    assert len(args()["pattern"]["items"]) == 2
    assert "# keep this comment" in (ws / "mylm.py").read_text()
    bad = patch({"op": "remove_layer", "class": "MyLM", "path": ["pattern"], "index": 9})
    assert bad.status_code == 422 and "no item 9" in bad.json()["detail"]
    assert patch({"op": "nope", "class": "MyLM"}).status_code == 422

    # fill_slot on a template, then the guided policy refuses a locked block in a slot
    (ws / "fill.py").write_text(TEMPLATE_FILE)
    etag = client.post("/api/files/fill.py/graph").json()["etag"]
    fill = {"op": "fill_slot", "class": "Fill", "path": ["block", "attn"], "slot": "mask",
            "node": block("CausalMask")}
    assert patch(fill, "fill.py").status_code == 200
    assert "mask=CausalMask()" in (ws / "fill.py").read_text()
    gating.reload()
    unlocks.set_policy("guided")
    before = (ws / "fill.py").read_text()
    refused = patch({**fill, "slot": "normalize", "node": block("RoPE")}, "fill.py")
    assert refused.status_code == 422
    assert (ws / "fill.py").read_text() == before
    etag = client.post("/api/files/mylm.py/graph").json()["etag"]
    roped = block("Block", norm=block("RMSNorm"), mlp=block("SwiGLU"),
                  attn=block("Attention", n_heads=lit(2), pos=block("RoPE")))
    refused = patch({"op": "add_layer", "class": "MyLM", "path": ["pattern"], "node": roped})
    assert refused.status_code == 422 and refused.json()["type"] == "locked"
