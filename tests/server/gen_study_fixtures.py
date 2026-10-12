# ruff: noqa: E501
"""Writes the real API responses the web tests use (web/test/fixtures/study-*.json, git-status-*,
prereg-preview, hardware-live, workers-live, jobs-live, bench-history) from a tiny record-mode study and a
live-looking queue. Not collected by default; regenerate from the repository root with

    uv run pytest tests/server/gen_study_fixtures.py

The files hold this machine's GPU name and a temporary workspace path; that is real output, not a secret.
"""
import json
from pathlib import Path

import pytest
from test_studies import SPEC, client, git, run_cli_study  # noqa: F401

from nanoscope import queue
from nanoscope.cli import main

pytestmark = pytest.mark.usefixtures("fake_data")
OUT = Path("web/test/fixtures")


def work_off(job_id):
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit):
        main(["run-job", str(job_id)])
    queue.finish(job_id)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def test_dump(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    ws = tmp_path / "ws"
    git(ws, "init", "-q", "-b", "main")
    git(ws, "config", "user.email", "t@example.com")
    git(ws, "config", "user.name", "t")
    record = SPEC.replace("seeds = [0, 1]", 'seeds = [0, 1, 2]\nmode = "record"') + "\n[predictions.wide]\nval_bpb = 1.2\n"
    client.post("/api/studies", json={"toml": record})
    dump("git-status-dirty.json", client.get("/api/git/status", params={"path": "studies/toy.toml"}).json())
    pv = client.post("/api/studies/toy/preregister/preview")
    work_off(pv.json()["job"]["id"])
    preview_job = queue.get(pv.json()["job"]["id"])
    dump("prereg-preview.json", json.loads(preview_job["result"]))
    dump("study-spec.json", client.get("/api/studies/toy/spec").json())
    cm = client.post("/api/studies/toy/preregister/commit", json={"preview_hash": json.loads(preview_job["result"])["hash"]})
    work_off(cm.json()["job"]["id"])
    dump("git-status-clean.json", client.get("/api/git/status", params={"path": "studies/toy.toml"}).json())
    run_cli_study(tmp_path)
    dump("study-list.json", client.get("/api/studies").json())
    dump("study-runs.json", client.get("/api/runs").json())
    dump("study-report.json", client.get("/api/studies/toy/report").json())
    (OUT / "study-report.md").write_text(client.get("/api/studies/toy/report.md").text)
    dump("study-card-upload.json", client.get("/api/studies/toy/card/upload", params={"repo": "RedhouaneLazib/nanoscope-ablation-cards"}).json())
    dump("study-card.json", client.get("/api/studies/toy/card").json())
    # a live queue: a study queued, two runs claimed by a GPU worker, a bench done on the CPU
    from fakes import tiny

    from nanoscope.bench import bench
    from nanoscope.jobs.worker import Worker
    from nanoscope.models import Bigram

    bench(Bigram, tiny(), steps=6, device="cpu", save=True)
    client.post("/api/studies", json={"toml": SPEC.replace('name = "toy"', 'name = "late"')})
    started = client.post("/api/studies/late/run").json()["job"]["id"]
    work_off(started)
    done = client.post("/api/bench", json={"model": "bigram", "steps": 6}).json()
    queue.claim("w", "cpu")  # the bench is the oldest queued job
    queue.finish(done["id"])
    gpu = Worker(device="cuda:0", slots=2)
    queue.claim(gpu.worker_id, "cuda:0")
    queue.claim(gpu.worker_id, "cuda:0")
    gpu.active = {j["id"]: None for j in queue.list_jobs() if j["state"] == "running"}  # type: ignore
    gpu._write_file()
    Worker(device="cpu", slots=1)._write_file()
    dump("hardware-live.json", client.get("/api/hardware").json())
    dump("workers-live.json", client.get("/api/workers").json())
    dump("jobs-live.json", client.get("/api/jobs").json())
    dump("bench-history.json", client.get("/api/hardware/bench").json())
