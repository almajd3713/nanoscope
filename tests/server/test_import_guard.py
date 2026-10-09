"""What the API process may import. It reads files and the queue and never runs user code, so it
must not reach for the parts of the library that import models or train them."""

import ast
from pathlib import Path

import nanoscope
import nanoscope.server as server_package

SERVER = Path(server_package.__file__).parent

# The documented library surface the server builds on. Anything else is a design question:
# add it here on purpose, with a reason, or call it through a job.
PUBLIC_MODULES = {
    "nanoscope.paths", "nanoscope.store", "nanoscope.queue", "nanoscope.schemas",
    "nanoscope.specs", "nanoscope.presets", "nanoscope.status", "nanoscope.progress",
    "nanoscope.prepare", "nanoscope.statistics", "nanoscope.studyspec", "nanoscope.study",
    "nanoscope.hardware", "nanoscope.estimate", "nanoscope.compare", "nanoscope.log",
    "nanoscope.fsutil", "nanoscope.learn", "nanoscope.blocks.catalog", "nanoscope.blocks.certs",
    "nanoscope.blocks.discover", "nanoscope.blocks.graph", "nanoscope.blocks.registry",
    "nanoscope.jobs.payload", "nanoscope.runref", "nanoscope.cards", "nanoscope.studyfiles",
    "nanoscope.models",  # the shipped models: our code, not a learner's
}
# These execute or import user code: the API process never touches them (plan 8.1).
FORBIDDEN = {"nanoscope.inspect", "nanoscope.modelref", "nanoscope.run", "nanoscope.train_loop",
             "nanoscope.blockstats", "nanoscope.bench", "nanoscope.jobs.execute",
             "nanoscope.jobs.runner", "nanoscope.jobs.worker", "nanoscope.blocks.certify",
             "nanoscope.prereg"}  # prereg runs git, so repository hooks run: worker only


def imported_names(path: Path):
    """(module, names) for every import of nanoscope in a file, including lazy ones."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == (
                "nanoscope"):
            yield node.module, [a.name for a in node.names], node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "nanoscope":
                    yield alias.name, [], node.lineno


def allowed(module: str) -> bool:
    return module == "nanoscope.server" or module.startswith("nanoscope.server.") or any(
        module == m or module.startswith(m + ".") for m in PUBLIC_MODULES)


def test_public_only():
    public_names = set(nanoscope.__all__) | {"__version__"}
    offenders = []
    for file in sorted(SERVER.rglob("*.py")):
        for module, names, line in imported_names(file):
            where = f"{file.relative_to(SERVER.parent)}:{line}"
            if module == "nanoscope":
                offenders += [f"{where} imports nanoscope.{n}" for n in names
                              if n not in public_names and n not in _submodules()]
            elif not allowed(module) or any(
                    module == f or module.startswith(f + ".") for f in FORBIDDEN):
                offenders.append(f"{where} imports {module}")
    assert not offenders, "the server may import only public library names:\n" + "\n".join(
        offenders)


def _submodules() -> set[str]:
    """`from nanoscope import paths` names a submodule: fine if the module is allowed."""
    return {m.split(".", 1)[1] for m in PUBLIC_MODULES if m.count(".") == 1}


def test_the_lists_do_not_overlap_and_name_real_modules():
    import importlib

    assert not {m for m in PUBLIC_MODULES for f in FORBIDDEN if m == f or m.startswith(f + ".")}
    for module in PUBLIC_MODULES | FORBIDDEN:
        importlib.import_module(module)  # a typo here would silently allow or forbid nothing


SENTINEL = '''\
import os

import torch.nn as nn

from nanoscope.blocks import Decoder, register_block
from nanoscope.blocks.attention import Attention
from nanoscope.blocks.mlp import GELUMLP
from nanoscope.blocks.norm import LayerNorm
from nanoscope.blocks.structure import Block

open(os.environ["SENTINEL_MARKER"], "w").write("imported")  # proof that something ran this file


class Sentinel(nn.Module):
    """A model whose import leaves a mark."""

    def __init__(self, vocab_size: int, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, width)

    def forward(self, idx):
        return self.emb(idx)


class SentinelLM(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size, 8, d_model=16, n_layers=1, block=Block(
            norm=LayerNorm(), attn=Attention(n_heads=2), mlp=GELUMLP()))


@register_block(family="mlp")
class SentinelBlock(nn.Module):
    def __init__(self, d_model, context_length, hidden: int = 4):
        super().__init__()
'''


def test_never_imports_workspace(home, tmp_path, monkeypatch):
    """Every endpoint that touches a workspace model, file or study, exercised: none of them
    imports the workspace (plan 8.1). The sentinel file writes a marker if anything does."""
    import sys

    from fastapi.testclient import TestClient

    from nanoscope.server.app import create_app
    from nanoscope.server.routes import runs as runs_route

    marker = tmp_path / "marker.txt"
    workspace = tmp_path / "ws"
    (workspace / "studies").mkdir(parents=True)
    (workspace / "sentinel_model.py").write_text(SENTINEL)
    monkeypatch.setenv("SENTINEL_MARKER", str(marker))
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(workspace))
    monkeypatch.setattr(runs_route, "QUEUE_WAIT", 0.2)
    client = TestClient(create_app())
    ref = "sentinel_model.py:Sentinel"
    spec = ('name = "s"\nseeds = [0, 1]\n[[variants]]\nname = "a"\nmodel = "' + ref + '"\n'
            '[[variants]]\nname = "b"\nmodel = "bigram"\n')

    def etag(path):
        return client.get(f"/api/files/{path}").json()["etag"]

    calls = [
        ("GET", "/api/models", None, {}),
        ("GET", f"/api/models/{ref}", None, {}),
        ("POST", f"/api/models/{ref}/describe", {}, {}),
        ("GET", "/api/blocks", None, {}),
        ("GET", "/api/files", None, {}),
        ("GET", "/api/files/sentinel_model.py", None, {}),
        ("PUT", "/api/files/notes/x.py", {"content": "x = 1\n"}, {}),
        ("POST", "/api/files/sentinel_model.py/graph", None, {}),
        ("POST", "/api/files/sentinel_model.py/lint", None, {}),
        ("POST", "/api/validate/run", {"model": ref, "kwargs": {"width": 4}}, {}),
        ("POST", "/api/validate/study", {"toml": spec}, {}),
        ("POST", "/api/runs", {"model": ref, "preset": "tinystories-5min"}, {}),
        ("POST", "/api/studies", {"toml": spec}, {}),
        ("POST", "/api/studies/s/run", None, {}),
        ("GET", "/api/studies", None, {}),
        ("GET", "/api/studies/s/report", None, {}),
        ("POST", "/api/studies/s/stop", None, {}),
        ("POST", "/api/bench", {"model": ref, "steps": 3}, {}),
        ("POST", "/api/runs/none/seed-0/generate", {}, {}),
        ("POST", "/api/curricula/foundations/01-bigram/start", None, {}),
        ("POST", "/api/curricula/foundations/01-bigram/check", None, {}),
        ("GET", "/api/curricula", None, {}),
        ("GET", "/api/curricula/foundations/03-attention-head", None, {}),
        ("GET", "/api/learn/progress", None, {}),
        ("GET", "/api/learn/unlocks", None, {}),
        ("POST", "/api/learn/policy", {"policy": "guided"}, {}),
        ("POST", "/api/learn/unlock", {"all": True}, {}),
        ("POST", "/api/compare", {"sets": ["a/seed-0", "b/seed-0"]}, {}),
        ("GET", "/api/runs", None, {}),
        ("GET", "/api/presets", None, {}),
        ("GET", "/api/data", None, {}),
        ("GET", "/api/hardware", None, {}),
        ("GET", "/api/hardware/bench", None, {}),
        ("GET", "/api/jobs", None, {}),
        ("GET", "/api/workers", None, {}),
        ("GET", "/api/schemas/status", None, {}),
        ("POST", "/api/sync/hub", {"repo": "me/runs", "ref": "p/m/seed-0"}, {}),
    ]
    for method, url, body, headers in calls:
        response = client.request(method, url, json=body, headers=headers)
        assert response.status_code < 500, (method, url, response.status_code, response.text)
    # a graph patch (needs the file's current ETag) and a save over the sentinel itself
    graph = client.post("/api/files/sentinel_model.py/graph").json()
    lm = next(c for c in graph["classes"] if c["name"] == "SentinelLM")
    lm["args"]["d_model"]["value"] = 32
    patched = client.post("/api/files/sentinel_model.py/graph/patch", json={"graph": graph},
                          headers={"If-Match": etag("sentinel_model.py")})
    assert patched.status_code == 200, patched.text
    assert "d_model=32" in (workspace / "sentinel_model.py").read_text()

    assert not marker.exists(), "the API process imported a workspace file"
    assert "sentinel_model" not in sys.modules
    assert not [m for m in list(sys.modules.values()) if str(
        getattr(m, "__file__", "") or "").startswith(str(workspace))]
