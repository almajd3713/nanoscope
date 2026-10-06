"""Generating text from a trained run, for the API.

A worker keeps one long-lived *inference process* with the recently used models loaded (an
LRU), so the second sample from a model does not pay for loading it again. It is a separate
process, not the API's: loading a run imports the learner's model class, which is user code.
A generation that outruns its timeout gets the process killed (the models reload on the next
call), because Python cannot interrupt a running forward pass any other way.
"""

from __future__ import annotations

import multiprocessing
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nanoscope import queue

MAX_MODELS = 4


def _checkpoint_stamp(ref: str) -> float:
    """Changes when the run's latest checkpoint does, so a model is reloaded after more training."""
    from nanoscope import store

    latest = store.resolve(ref) / "latest.json"
    return latest.stat().st_mtime_ns if latest.exists() else 0.0


def serve(conn: Any, max_models: int = MAX_MODELS,
          load: Callable[[str], Any] | None = None) -> None:
    """The inference process's loop: take a request, answer with text or an error."""
    if load is None:
        from nanoscope.store import load_run

        load = load_run
    models: OrderedDict[tuple[str, float], Any] = OrderedDict()
    while True:
        try:
            request = conn.recv()
        except EOFError:
            return
        if request is None:
            return
        try:
            key = (request["ref"], _checkpoint_stamp(request["ref"]))
            if key in models:
                models.move_to_end(key)
            else:
                for old in [k for k in models if k[0] == key[0]]:
                    del models[old]  # an older checkpoint of the same run
                models[key] = load(request["ref"])
                while len(models) > max_models:
                    models.popitem(last=False)
            text = models[key].generate(
                request.get("prompt", ""), request.get("max_new_tokens", 200),
                request.get("temperature", 0.8), request.get("seed", 42))
            conn.send({"text": text, "loaded": sorted({k[0] for k in models})})
        except Exception as exc:
            conn.send({"error": f"{type(exc).__name__}: {exc}"})


class InferencePool:
    """One inference process, started when first needed, serving one request at a time."""

    def __init__(self, max_models: int = MAX_MODELS) -> None:
        self.max_models = max_models
        self._lock = threading.Lock()
        self._proc: Any = None
        self._conn: Any = None

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        context = multiprocessing.get_context("spawn")  # a fresh interpreter: no inherited state
        parent, child = context.Pipe()
        self._proc = context.Process(target=serve, args=(child, self.max_models), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent

    def generate(self, request: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        with self._lock:
            self._ensure()
            self._conn.send(dict(request))
            if not self._conn.poll(timeout):
                self.kill()
                raise TimeoutError(f"generation did not finish in {timeout:g}s")
            try:
                return self._conn.recv()
            except EOFError:
                self.kill()
                raise RuntimeError("the inference process stopped while generating") from None

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.is_alive():
            proc.kill()
            proc.join(5)
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    close = kill


class InferenceRunner:
    """A worker "slot" for a generate job: same interface as a subprocess runner, but the work
    goes to the shared inference process from a thread."""

    def __init__(self, pool: InferencePool) -> None:
        self.pool = pool
        self._thread: threading.Thread | None = None
        self._code: int | None = None

    def start(self, job_id: int, log: Path, env: Mapping[str, str]) -> None:
        import json

        payload = json.loads(queue.get(job_id)["payload"])

        def work() -> None:
            try:
                answer = self.pool.generate(payload, float(payload.get("timeout", 60)))
                if "error" in answer:
                    queue.record(job_id, error=answer["error"])
                    self._code = 1
                else:
                    queue.record(job_id, result={"ref": payload["ref"], "text": answer["text"]})
                    self._code = 0
            except Exception as exc:
                queue.record(job_id, error=f"{type(exc).__name__}: {exc}")
                self._code = 1

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def poll(self) -> int | None:
        return self._code

    def terminate(self, kill: bool = False) -> None:
        self.pool.kill()  # the waiting thread then reports the failure
