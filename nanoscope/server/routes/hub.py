from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope import queue
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import JobDoc

router = APIRouter(prefix="/api", tags=["hub"])


class HubSync(BaseModel):
    repo: str  # "user/runs": the private Hub repo the run was pushed to
    ref: str  # the run ref to create or update here, e.g. "tinystories-5min/modern/seed-0"
    path: str | None = None  # where the run sits in the repo (default: the same as ref)


@router.post("/sync/hub", status_code=202)
def sync_hub(body: HubSync) -> JobDoc:
    """Pull a run that was trained on Kaggle or Colab (with `push_to_hub=`) into this machine.
    It is a job (it downloads checkpoints); the HF token is passed to it and to nothing else."""
    payload = {"repo": body.repo, "ref": body.ref}
    if body.path:
        payload["path"] = body.path
    job_id = queue.enqueue("sync-hub", payload, lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))
