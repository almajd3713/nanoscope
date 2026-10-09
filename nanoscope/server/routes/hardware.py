from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from nanoscope import paths, queue
from nanoscope.prepare import read_all
from nanoscope.presets import get_preset, list_presets
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import BenchDoc, JobDoc, PrepareDoc
from nanoscope.server.routes.models import resolve_ref

router = APIRouter(prefix="/api", tags=["data and hardware"])


class DataEntry(BaseModel):
    preset: str
    dataset: str
    tokenizer: str
    vocab_size: int | None
    prepare: PrepareDoc | None  # prepare.json: the stage a download or tokenizing is in, or done


@router.get("/data")
def data() -> list[DataEntry]:
    """Every preset and the state of its data (downloading, tokenizing, done, failed), read
    from the `prepare.json` files that `prepare-data` and `run` write while they work."""
    state = {doc["preset"]: doc for doc in read_all()}
    out = []
    for name in list_presets():
        preset = get_preset(name)
        doc = state.get(name)
        out.append(DataEntry(
            preset=name, dataset=preset.dataset, tokenizer=preset.tokenizer,
            vocab_size=preset.vocab_size,
            prepare=PrepareDoc.model_validate(doc) if doc else None))
    return out


@router.post("/data/{preset}/prepare", status_code=202)
def prepare(preset: str) -> JobDoc:
    """Download and tokenize a preset's data as a job (it can take minutes)."""
    get_preset(preset)  # unknown presets are a 404 with the list of known ones
    job_id = queue.enqueue("prepare-data", {"preset": preset}, lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))


class Device(BaseModel):
    name: str
    kind: str  # "cpu" or "cuda"
    memory_total: int | None = None
    memory_free: int | None = None
    label: str | None = None  # the GPU's own name


class Hardware(BaseModel):
    devices: list[Device]
    cpu_count: int
    torch: str
    workers: list[dict[str, Any]]  # the workers that are running (their worker.v1 files)


@router.get("/hardware")
def hardware() -> Hardware:
    """What this machine can train on, and which workers are running."""
    import torch

    from nanoscope.progress import read_workers

    devices = [Device(name="cpu", kind="cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            devices.append(Device(name=f"cuda:{i}", kind="cuda", memory_total=total,
                                  memory_free=free, label=torch.cuda.get_device_name(i)))
    return Hardware(devices=devices, cpu_count=os.cpu_count() or 1, torch=torch.__version__,
                    workers=read_workers())


class BenchRequest(BaseModel):
    model: str
    preset: str = "tinystories-5min"
    steps: int = 60
    device: str | None = None  # measure on this device's worker (default: any worker)


@router.post("/bench", status_code=202)
def bench(body: BenchRequest, request: Request) -> JobDoc:
    """Measure training speed on a worker's device (a job: it trains for real). The result is
    saved to the bench history, as `nanoscope bench --save` does."""
    get_preset(body.preset)
    payload = {"model": resolve_ref(body.model), "preset": body.preset, "steps": body.steps}
    if body.device:
        payload["device"] = body.device
    job_id = queue.enqueue("bench", payload, lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))


@router.get("/hardware/bench")
def bench_results() -> list[BenchDoc]:
    """Every saved bench result (`nanoscope bench --save`), oldest first."""
    path = paths.hardware_dir() / "bench.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(BenchDoc.model_validate(json.loads(line)))
        except ValueError:
            continue
    return rows
