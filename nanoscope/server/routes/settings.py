from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from nanoscope import __version__, paths
from nanoscope.progress import read_workers

router = APIRouter(prefix="/api", tags=["settings"])

WORKER_GONE_SECONDS = 60  # a worker whose heartbeat is older than this is not counted


class Listening(BaseModel):
    host: str
    port: int
    loopback: bool
    token_required: bool  # beyond loopback every request needs the token


class WorkerSecrets(BaseModel):
    worker_id: str
    device: str
    secrets: dict[str, bool]


class SettingsDoc(BaseModel):
    """Read-only facts about this server. Secrets appear only as yes or no."""

    version: str
    home: str
    workspace: str
    data: str
    runs: str
    listening: Listening
    jobs_offline: bool  # NANOSCOPE_JOBS_OFFLINE: workers refuse jobs that need the network
    workers: list[WorkerSecrets]


@router.get("/settings")
def settings(request: Request) -> SettingsDoc:
    """Where this server keeps things, how it listens, and which credentials its workers have
    (HF_TOKEN, WANDB_API_KEY: yes or no, from their heartbeat, never the values). Change a value
    where the page says: these are environment variables and flags, not settings you edit here."""
    served = request.app.state.settings
    workers: list[dict[str, Any]] = [w for w in read_workers() if w["age"] < WORKER_GONE_SECONDS]
    return SettingsDoc(
        version=__version__, home=str(paths.home() or paths._USER_HOME),
        workspace=str(paths.workspace_dir()),
        data=str(paths.data_dir()), runs=str(paths.runs_dir()),
        listening=Listening(host=served.host, port=served.port, loopback=served.loopback,
                            token_required=not served.loopback),
        jobs_offline=os.environ.get("NANOSCOPE_JOBS_OFFLINE", "") not in ("", "0"),
        workers=[WorkerSecrets(worker_id=w["worker_id"], device=w["device"],
                               secrets=w.get("secrets", {})) for w in workers])
