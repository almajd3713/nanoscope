"""Starting a job's process. Today that is a subprocess; containers come with deployments B/C."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

# Secrets reach only the jobs that use them; describe, check and inspect never get one.
SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY")


def _die_with_parent() -> None:
    """Linux: SIGKILL this process when its parent (the worker) dies, however it dies.

    Otherwise a killed worker leaves its training process running, and the requeued job would
    start a second process on the same run folder. The run resumes from its last checkpoint."""
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


class JobRunner(Protocol):
    """One job's process: start it, ask whether it ended, stop it."""

    def start(self, job_id: int, log: Path, env: Mapping[str, str]) -> None: ...

    def poll(self) -> int | None:
        """The exit code, or None while it runs."""
        ...

    def terminate(self, kill: bool = False) -> None: ...


class SubprocessRunner:
    """`nanoscope run-job <id>` in a fresh process, its output appended to `log`."""

    def __init__(self, command: Callable[[int], list[str]] | None = None) -> None:
        self._command = command or (
            lambda job_id: [sys.executable, "-m", "nanoscope.cli", "run-job", str(job_id)])
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self, job_id: int, log: Path, env: Mapping[str, str]) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as fh:
            self._proc = subprocess.Popen(
                self._command(job_id), stdout=fh, stderr=subprocess.STDOUT, env=dict(env),
                start_new_session=True,  # a Ctrl-C at the worker is the worker's to handle
                preexec_fn=_die_with_parent if sys.platform == "linux" else None)

    def poll(self) -> int | None:
        if self._proc is None:
            raise RuntimeError("the job was never started")
        return self._proc.poll()

    def terminate(self, kill: bool = False) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGKILL if kill else signal.SIGTERM)


class ContainerRunner:
    """One container per job; part of deployments B and C, not built yet."""

    def start(self, job_id: int, log: Path, env: Mapping[str, str]) -> None:
        raise NotImplementedError(
            "container jobs belong to deployments B/C (a hosted or multi-machine setup); "
            "use SubprocessRunner for a local worker")

    def poll(self) -> int | None:
        raise NotImplementedError("container jobs belong to deployments B/C")

    def terminate(self, kill: bool = False) -> None:
        raise NotImplementedError("container jobs belong to deployments B/C")


def job_env(kind: str, payload: dict[str, Any],
            environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a job's process gets: ours, minus every secret it doesn't need."""
    env = dict(os.environ if environ is None else environ)
    wanted: set[str] = set()
    if kind in ("prepare-data", "sync-hub", "card-push"):
        wanted.update(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    elif kind == "run":
        if payload.get("push_to_hub"):
            wanted.update(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
        if payload.get("wandb"):
            wanted.add("WANDB_API_KEY")
    for name in SECRETS:
        if name not in wanted:
            env.pop(name, None)
    return env
