"""A worker: claims jobs for one device, runs each in its own process, keeps leases alive.

    nanoscope worker --device cuda:0 --slots 2

Each tick it requeues expired leases, checks its children (finishing, renewing or timing
them out), and fills free slots. On SIGTERM it asks its children to stop (a STOP file for
runs, so they checkpoint), waits, and hands their jobs back to the queue.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscope import __version__, paths, queue, store
from nanoscope.fsutil import write_json_atomic
from nanoscope.hardware import HEADROOM, free_memory, probe_memory
from nanoscope.jobs.inference import InferencePool, InferenceRunner
from nanoscope.jobs.runner import JobRunner, SubprocessRunner, job_env
from nanoscope.log import info
from nanoscope.schemas.upgrade import read_json
from nanoscope.status import HEARTBEAT_SECONDS, STOP_FILE

STOP_GRACE = 30.0  # seconds a child gets to checkpoint after STOP before it is killed


@dataclass
class _Active:
    job: sqlite3.Row
    runner: JobRunner
    started: float
    limit: float | None
    stopping_since: float | None = None
    timed_out: bool = False


def memory_key(payload: dict[str, Any], device: str) -> str:
    """What a probe's result is cached under: model, its kwargs, the preset, the device name."""
    shape = {k: payload.get(k) for k in ("model", "kwargs", "preset", "overrides", "custom_preset")}
    digest = hashlib.sha256(json.dumps(shape, sort_keys=True, default=str).encode()).hexdigest()
    return f"{payload['model']}|{digest[:16]}|{device}"


def _probe_job(job: sqlite3.Row, payload: dict[str, Any]) -> int:
    """Bytes a run job needs on its device, by training one real step in this process."""
    import inspect

    import torch

    from nanoscope.dataset import load_tokenizer
    from nanoscope.jobs.execute import model_class
    from nanoscope.jobs.payload import build_preset

    preset = build_preset(payload)
    cls = model_class(payload["model"])
    params = inspect.signature(cls).parameters
    from_data = {"vocab_size": load_tokenizer(preset).vocab_size,
                 "context_length": preset.context_length}
    kwargs = {k: v for k, v in from_data.items() if k in params} | payload.get("kwargs", {})
    return probe_memory(lambda: cls(**kwargs), preset, torch.device(job["device"]))


class Worker:
    def __init__(
        self, device: str = "cpu", slots: int = 1, *, lanes: tuple[str, ...] = queue.LANES,
        worker_id: str | None = None,
        runner_factory: Callable[[], JobRunner] = SubprocessRunner,
        timeout: float | None = None, lease_seconds: float = queue.LEASE_SECONDS,
        poll_seconds: float = 1.0, stop_grace: float = STOP_GRACE, exit_when_idle: bool = False,
        free_memory: Callable[[str], int] = free_memory,
        probe: Callable[[sqlite3.Row, dict[str, Any]], int] = _probe_job,
    ) -> None:
        self.device, self.slots, self.lanes = device, slots, lanes
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{device}"
        self.runner_factory, self.timeout = runner_factory, timeout
        self.lease_seconds, self.poll_seconds = lease_seconds, poll_seconds
        self.stop_grace, self.exit_when_idle = stop_grace, exit_when_idle
        self._free_memory, self._probe = free_memory, probe
        self.active: dict[int, _Active] = {}
        self._pool: InferencePool | None = None
        self.shutting_down = False
        self._started_at = self._now()

    # -- control ---------------------------------------------------------------------------

    def shutdown(self, *_: Any) -> None:
        """Ask the loop to wind down (safe to call from a signal handler)."""
        self.shutting_down = True

    def run(self) -> None:
        old = {s: signal.signal(s, self.shutdown) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            info(f"worker {self.worker_id}: {self.slots} slot(s) on {self.device}")
            while not self.shutting_down:
                busy = self.tick()
                if self.exit_when_idle and not busy and queue.pending(self.lanes) == 0:
                    break  # (a job another worker holds may still come back: its lease expires)
                time.sleep(self.poll_seconds)
            self._wind_down()
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            if self._pool is not None:
                self._pool.close()
            self.worker_file().unlink(missing_ok=True)

    def tick(self) -> bool:
        """One pass of the loop. Returns whether any job is running afterwards."""
        for job_id in list(self.active):
            self._check(job_id)  # renews our leases before anyone's expiry is judged
        queue.requeue_expired(skip_worker=self.worker_id)
        if not self.shutting_down:
            self._fill()
        self._write_file()
        return bool(self.active)

    # -- children --------------------------------------------------------------------------

    def _fill(self) -> None:
        while len(self.active) < self.slots:
            job = queue.claim(self.worker_id, self.device, self.lanes,
                              lease_seconds=self.lease_seconds)
            if job is None:
                return
            if not self._admit(job):
                return
            self._start(job)

    def _admit(self, job: sqlite3.Row) -> bool:
        """Does the job fit this GPU's free memory? If not it goes back, queued, not failed."""
        if job["kind"] != "run" or not self.device.startswith("cuda") or self.slots < 2:
            return True
        payload = json.loads(job["payload"])
        key = memory_key(payload, self.device)
        need = queue.cached_memory(key)
        if need is None:
            need = self._probe(job, payload)
            queue.cache_memory(key, need)
        free = self._free_memory(self.device)
        if need <= free * HEADROOM:
            return True
        if not self.active:
            queue.finish(job["id"], f"needs about {need / 2**30:.1f} GiB on {self.device} "
                         f"but only {free / 2**30:.1f} GiB is free")
        else:
            queue.release(job["id"], self.worker_id)
        return False

    def _start(self, job: sqlite3.Row) -> None:
        payload = json.loads(job["payload"])
        if job["kind"] == "generate":  # served from a long-lived process that keeps models loaded
            self._pool = self._pool or InferencePool()
            runner: JobRunner = InferenceRunner(self._pool)
        else:
            runner = self.runner_factory()
        env = job_env(job["kind"], payload)
        env["NANOSCOPE_WORKER_SLOTS"] = str(self.slots)
        log = paths.job_logs_dir() / f"{job['id']}.log"
        runner.start(job["id"], log, env)
        limit = payload.get("timeout") or self.timeout
        now = time.time()
        self.active[job["id"]] = _Active(job, runner, now, limit)
        info(f"job {job['id']} ({job['kind']}{' ' + job['ref'] if job['ref'] else ''}) "
             f"started on {self.device}, log: {log}")

    def _check(self, job_id: int) -> None:
        active = self.active[job_id]
        code = active.runner.poll()
        if code is not None:
            del self.active[job_id]
            self._finished(active, code)
            return
        now = time.time()
        if active.stopping_since is not None:
            if now - active.stopping_since >= self.stop_grace:
                active.runner.terminate(kill=True)
        elif active.limit is not None and now - active.started >= active.limit:
            active.timed_out = True
            self._ask_to_stop(active)
        elif queue.get(job_id)["state"] == "cancelling":
            self._ask_to_stop(active)  # queue.cancel already wrote STOP for runs
        if self._alive(active) and not queue.renew(job_id, self.worker_id,
                                                   lease_seconds=self.lease_seconds):
            # The queue gave the job to someone else (our lease expired): don't fight it.
            active.runner.terminate(kill=True)
            del self.active[job_id]

    def _alive(self, active: _Active) -> bool:
        """Is this child making progress? A run in state `running` must keep its heartbeat."""
        ref = active.job["ref"]
        if active.job["kind"] != "run" or not ref:
            return True
        try:
            doc = read_json(store.resolve(ref, must_exist=False) / "status.json", "status")
        except (OSError, ValueError):
            return True
        if doc.get("state") != "running" or not doc.get("heartbeat_at"):
            return True
        beat = datetime.fromisoformat(doc["heartbeat_at"]).timestamp()
        return time.time() - beat < max(self.lease_seconds, 3 * HEARTBEAT_SECONDS)

    def _ask_to_stop(self, active: _Active) -> None:
        active.stopping_since = time.time()
        ref = active.job["ref"]
        folder = store.resolve(ref, must_exist=False) if ref else None
        if active.job["kind"] == "run" and folder is not None and folder.is_dir():
            (folder / STOP_FILE).write_text("", encoding="utf-8")
        else:
            active.runner.terminate()

    def _mark_requeued(self, active: _Active) -> None:
        """A run stopped for our shutdown is not cancelled: its job is back in the queue and will
        resume. Say so in its status.json, so nothing reading the run folder sees `cancelled`."""
        ref = active.job["ref"]
        folder = store.resolve(ref, must_exist=False) if ref else None
        if active.job["kind"] != "run" or folder is None:
            return
        path = folder / "status.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if doc.get("state") in ("running", "preparing", "stopped", "cancelled"):
            doc["state"] = "queued"
            doc["updated_at"] = self._now()
            write_json_atomic(path, doc)

    def _finished(self, active: _Active, code: int) -> None:
        job_id = active.job["id"]
        if active.timed_out:
            state = queue.finish(job_id, f"timeout after {active.limit:g}s")
        elif self.shutting_down:
            state = queue.release(job_id, self.worker_id) or "released"
            if state == "queued":
                self._mark_requeued(active)
        else:
            row = queue.get(job_id)
            if code != 0:
                error = row["error"] or f"exit code {code}; see {self._log(job_id)}"
                state = queue.finish(job_id, error)
            else:
                result = json.loads(row["result"]) if row["result"] else {}
                ended = result.get("state") in ("stopped", "cancelled")
                state = queue.finish(job_id, cancelled=ended)
        info(f"job {job_id} {state}")

    def _log(self, job_id: int) -> Path:
        return paths.job_logs_dir() / f"{job_id}.log"

    def _wind_down(self) -> None:
        """SIGTERM: STOP every child, wait for them to checkpoint, give their jobs back."""
        for active in self.active.values():
            self._ask_to_stop(active)
        end = time.time() + self.stop_grace
        while self.active and time.time() < end:
            for job_id in list(self.active):
                code = self.active[job_id].runner.poll()
                if code is not None:
                    self._finished(self.active.pop(job_id), code)
            time.sleep(min(self.poll_seconds, 0.1))
        for job_id, active in list(self.active.items()):
            active.runner.terminate(kill=True)
            if queue.release(job_id, self.worker_id) == "queued":
                self._mark_requeued(active)
            del self.active[job_id]
        self._write_file()

    # -- state on disk ---------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def worker_file(self) -> Path:
        return paths.workers_dir() / f"{self.worker_id}.json"

    def _write_file(self) -> None:
        write_json_atomic(self.worker_file(), {
            "schema": 1, "nanoscope": __version__, "worker_id": self.worker_id,
            "device": self.device, "slots": self.slots, "pid": os.getpid(),
            "host": socket.gethostname(), "jobs": sorted(self.active),
            "started_at": self._started_at, "heartbeat_at": self._now(),
            "secrets": self._secrets(),
        })

    @staticmethod
    def _secrets() -> dict[str, bool]:
        """Which credentials this worker has, as yes or no. Never the values: a job only gets
        the ones it needs (jobs/runner.py job_env), and the web page reads this file."""
        hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return {"HF_TOKEN": bool(hf), "WANDB_API_KEY": bool(os.environ.get("WANDB_API_KEY"))}
