"""A worker started with the server, for native (non-compose) use: `nanoscope serve --worker cpu`.

It is an ordinary `nanoscope worker` process: the API only enqueues, so jobs run (and user code
executes) there, never in the API process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nanoscope import paths


def log_path() -> Path:
    return paths.server_dir() / "worker.log"


def start(device: str) -> subprocess.Popen[bytes]:
    log = log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "nanoscope.cli", "worker", "--device", device]
    with log.open("ab") as handle:
        return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)


def stop(proc: subprocess.Popen[bytes], wait: float = 15.0) -> None:
    """Ask the worker to finish (it requeues what it holds), then make sure it is gone."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
