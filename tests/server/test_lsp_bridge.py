import json
import sys
import textwrap

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nanoscope.server import lsp_bridge

# A stand-in language server: answers every request with {"echo": method}, framed like the real
# ones, so the bridge is tested without basedpyright.
FAKE = textwrap.dedent('''
    import json, sys
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        length = None
        while True:
            line = stdin.readline()
            if not line:
                raise SystemExit
            if not line.strip():
                break
            name, _, value = line.decode().partition(":")
            if name.lower() == "content-length":
                length = int(value)
        message = json.loads(stdin.read(length))
        body = json.dumps({"jsonrpc": "2.0", "id": message.get("id"),
                           "result": {"echo": message["method"]}}).encode()
        stdout.write(b"Content-Length: %d\\r\\n\\r\\n" % len(body) + body)
        stdout.flush()
''')


@pytest.fixture
def client(tmp_path, monkeypatch):
    server = tmp_path / "fake_lsp.py"
    server.write_text(FAKE)
    monkeypatch.setenv("NANOSCOPE_LSP_COMMAND", f"{sys.executable} {server}")
    return TestClient(lsp_bridge.app)


def test_health_names_the_server_command(client):
    assert client.get("/health").json()["status"] == "ok"


def test_messages_go_to_the_server_and_back_unchanged(client):
    with client.websocket_connect("/lsp") as ws:
        for i, method in enumerate(["initialize", "textDocument/hover"], start=1):
            ws.send_text(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": {}}))
            assert json.loads(ws.receive_text()) == {
                "jsonrpc": "2.0", "id": i, "result": {"echo": method}}


def test_each_connection_gets_its_own_server(client):
    with client.websocket_connect("/lsp") as a, client.websocket_connect("/lsp") as b:
        a.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "a", "params": {}}))
        b.send_text(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "b", "params": {}}))
        assert json.loads(a.receive_text())["result"] == {"echo": "a"}
        assert json.loads(b.receive_text())["result"] == {"echo": "b"}


@pytest.mark.parametrize("origin,ok", [
    (None, True), ("http://127.0.0.1:8765", True), ("http://localhost:3000", True),
    ("http://[::1]:8765", True), ("https://evil.example", False),
    ("http://127.0.0.1.evil.example", False)])
def test_only_pages_on_this_machine_may_connect(client, origin, ok):
    headers = {"origin": origin} if origin else {}
    if ok:
        with client.websocket_connect("/lsp", headers=headers):
            pass
    else:
        with pytest.raises(WebSocketDisconnect) as refused, \
                client.websocket_connect("/lsp", headers=headers):
            pass
        assert refused.value.code == 1008


def test_an_operator_can_allow_another_origin(client, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_LSP_ORIGINS", "https://lab.example, https://other.example")
    with client.websocket_connect("/lsp", headers={"origin": "https://lab.example"}):
        pass
