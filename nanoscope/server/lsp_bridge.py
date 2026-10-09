"""A language server over a WebSocket, for the code editor (the `lsp` compose service).

Browsers cannot talk to a language server's stdio, so each WebSocket connection gets its own
server process (`basedpyright-langserver --stdio` unless NANOSCOPE_LSP_COMMAND says otherwise)
and the bridge turns WebSocket text messages into Content-Length-framed stdio messages and back.
It does not look inside the messages and writes nothing itself: the workspace is mounted read
only in the service.

    uvicorn nanoscope.server.lsp_bridge:app --host 0.0.0.0 --port 3000

The editor connects to `ws://<host>:<port>/lsp`; `GET /health` says the service is up. Only pages
served from this machine may connect (`localhost`, `127.0.0.1`, `[::1]`), plus any origin listed
in NANOSCOPE_LSP_ORIGINS (comma separated); another site's page is refused, since the server can
read the workspace.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
from typing import IO
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def command() -> list[str]:
    return shlex.split(os.environ.get("NANOSCOPE_LSP_COMMAND", "basedpyright-langserver --stdio"))


def origin_allowed(origin: str | None) -> bool:
    """No Origin header (not a browser) or a page on this machine, or one the operator listed."""
    if not origin:
        return True
    extra = {o.strip().rstrip("/") for o in os.environ.get("NANOSCOPE_LSP_ORIGINS", "").split(",")}
    if origin.rstrip("/") in extra:
        return True
    return (urlparse(origin).hostname or "") in LOCAL_HOSTS


def read_message(stdout: IO[bytes]) -> bytes | None:
    """One Content-Length framed message body, or None when the server has gone."""
    length = None
    while True:
        line = stdout.readline()
        if not line:
            return None
        text = line.decode("ascii", "replace").strip()
        if not text:
            break
        name, _, value = text.partition(":")
        if name.lower() == "content-length":
            length = int(value.strip())
    if length is None:
        return None
    body = stdout.read(length)
    return body if len(body) == length else None


async def lsp(ws: WebSocket) -> None:
    if not origin_allowed(ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    # blocking pipes read in worker threads: no event-loop subprocess transports to outlive it
    proc = subprocess.Popen(
        command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=os.environ.get("NANOSCOPE_LSP_ROOT") or None)
    assert proc.stdin is not None and proc.stdout is not None
    stdin, stdout = proc.stdin, proc.stdout

    def write(body: bytes) -> None:
        stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        stdin.flush()

    async def to_server() -> None:
        try:
            while True:
                await asyncio.to_thread(write, (await ws.receive_text()).encode())
        except (WebSocketDisconnect, BrokenPipeError):
            return  # the editor went away, or the server did

    async def to_client() -> None:
        while (body := await asyncio.to_thread(read_message, stdout)) is not None:
            await ws.send_text(body.decode())

    tasks = [asyncio.create_task(to_server()), asyncio.create_task(to_client())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        proc.kill()  # also ends a read blocked in a thread
        for task in tasks:
            task.cancel()
        with contextlib.suppress(OSError):
            stdin.close()
        proc.wait()  # returns at once after a kill
        stdout.close()
    with contextlib.suppress(RuntimeError):
        await ws.close()


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "command": command()[0]})


app = Starlette(routes=[Route("/health", health), WebSocketRoute("/lsp", lsp)])
