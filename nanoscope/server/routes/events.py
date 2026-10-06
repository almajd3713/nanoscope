from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nanoscope import store
from nanoscope.server import sse
from nanoscope.server.tail import Tail, run_dir_of, stream

router = APIRouter(prefix="/api", tags=["events"])

RESCAN_SECONDS = 2.0  # how often the multiplexed stream looks for new runs


async def _until_disconnected(request: Request, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if await request.is_disconnected():
            stop.set()
            return
        await asyncio.sleep(0.25)


@router.get("/runs/{ref:path}/events")
async def run_events(ref: str, request: Request, since_step: int | None = None
                     ) -> StreamingResponse:
    """Server-sent events for one run: `state`, `step` (at most 4 a second), `eval`, `sample`,
    `checkpoint`, `blockstats`, and `reset` when the metrics file is rewritten (a resume).
    Without `since_step` the stream starts now; with it, rows after that step are replayed.
    Works for runs started from the CLI too: it reads the same files."""
    folder = run_dir_of(ref)  # a missing run is a 404 before the stream opens
    tail = Tail(folder, store.ref_of(folder), since_step=since_step)
    stop = asyncio.Event()
    asyncio.create_task(_until_disconnected(request, stop))
    return sse.response(stream(lambda: [tail], stop=stop))


@router.get("/events")
async def all_events(request: Request, prefix: str = "") -> StreamingResponse:
    """The events of every run under a ref prefix on one stream (each event names its `ref`).
    Runs that appear while it is open are followed from their first row."""
    stop = asyncio.Event()
    asyncio.create_task(_until_disconnected(request, stop))
    tails: dict[str, Tail] = {}
    state = {"scanned": 0.0, "first": True}

    def current() -> list[Tail]:
        now = time.monotonic()
        if now - state["scanned"] >= RESCAN_SECONDS:
            state["scanned"] = now
            try:
                found = store.list_runs(prefix)
            except (OSError, ValueError):  # a file mid-write: try again on the next pass
                found = []
            for run in found:
                ref = store.ref_of(run.run_dir)
                if ref not in tails:
                    # runs already there when we connect are followed live; later ones from the top
                    tails[ref] = Tail(run.run_dir, ref, since_step=None if state["first"] else 0)
            if found:
                state["first"] = False
        return list(tails.values())

    return sse.response(stream(current, stop=stop))
