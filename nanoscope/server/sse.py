"""Server-sent events: formatting and the response."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse


def frame(event: str, data: Any, event_id: str | None = None) -> str:
    """One SSE message: `event:`, an optional `id:`, and the JSON `data:`."""
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {json.dumps(data)}")
    return "\n".join(lines) + "\n\n"


KEEPALIVE = ": keepalive\n\n"  # a comment: keeps proxies and browsers from closing an idle stream


def response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
