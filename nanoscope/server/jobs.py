"""Jobs as the API shows them."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from nanoscope import queue
from nanoscope.server.models import JobDoc


def job_doc(row: sqlite3.Row) -> JobDoc:
    def loaded(text: str | None) -> Any:
        return json.loads(text) if text else None

    return JobDoc(
        id=row["id"], kind=row["kind"], lane=row["lane"], state=row["state"], ref=row["ref"],
        owner=row["owner"], device=row["device"], worker_id=row["worker_id"],
        attempts=row["attempts"], payload=loaded(row["payload"]) or {},
        result=loaded(row["result"]), error=row["error"], created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"])


def submit(kind: str, payload: dict[str, Any], *, lane: str = "interactive",
           ref: str | None = None) -> JobDoc:
    """Put a job on the queue; invalid payloads raise the queue's own error (a 422)."""
    job_id = queue.enqueue(kind, payload, lane=lane, ref=ref)
    assert job_id is not None
    return job_doc(queue.get(job_id))
