from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from nanoscope import queue
from nanoscope.progress import read_workers
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import JobDoc

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs")
def list_jobs(state: str | None = None, kind: str | None = None, limit: int = 200) -> list[JobDoc]:
    """Jobs, newest last: queued, running, done, failed or cancelled. Filter by `state` and
    `kind` (run, study, describe, check, generate, bench, prepare-data)."""
    rows = [job_doc(r) for r in queue.list_jobs(state)
            if kind is None or r["kind"] == kind]
    return rows[-limit:]


@router.get("/jobs/{job_id}")
def get_job(job_id: int) -> JobDoc:
    try:
        return job_doc(queue.get(job_id))
    except KeyError:
        raise FileNotFoundError(f"no job {job_id}") from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int) -> JobDoc:
    """Cancel a job: a queued one never starts; a running run gets a STOP (it checkpoints and
    can be resumed). A job that already finished stays as it is."""
    get_job(job_id)
    queue.cancel(job_id)
    return job_doc(queue.get(job_id))


@router.get("/workers")
def workers() -> list[dict[str, Any]]:
    """The running workers (worker.v1 files), each with the age of its last heartbeat."""
    return read_workers()
