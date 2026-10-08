import socket
import threading
import time

import pytest

from nanoscope.server.app import create_app


class LiveServer:
    def __init__(self, url):
        self.url = url


@pytest.fixture
def live_server(home):
    """The app on a real socket in a thread: server-sent events need a real connection (the
    in-process test client buffers a response until it ends, and these never end)."""
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "the test server did not start"
    yield LiveServer(f"http://127.0.0.1:{port}")
    server.should_exit = True
    thread.join(10)
