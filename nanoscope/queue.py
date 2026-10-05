"""The job queue: one SQLite file in WAL mode, shared by the API, the CLI and the workers.

Every write goes through `connect()` so the schema exists and every connection waits for
a competing writer instead of failing at once.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match

from nanoscope import paths, schemas, store
from nanoscope.status import STOP_FILE

LANES = ("interactive", "batch")  # claim order: interactive first
LEASE_SECONDS = 60.0
MAX_ATTEMPTS = 3

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    lane TEXT NOT NULL DEFAULT 'batch',
    payload TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    ref TEXT,
    owner TEXT NOT NULL DEFAULT 'local',
    devices_required INTEGER NOT NULL DEFAULT 1,
    device TEXT,
    worker_id TEXT,
    lease_until REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs (state, lane, id);
CREATE TABLE IF NOT EXISTS memory_cache (
    key TEXT PRIMARY KEY,
    bytes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_ref ON jobs (ref);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the queue (creating it on first use). Callers close it; `closing()` works."""
    path = path or paths.queue_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: transactions are explicit (BEGIN IMMEDIATE) where atomicity matters.
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return conn


def schema_version(path: Path | None = None) -> int:
    with closing(connect(path)) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row["value"])


class InvalidJob(ValueError):
    """The payload doesn't match the job.v1 schema of its kind."""


def validate_job(kind: str, payload: dict[str, Any]) -> None:
    """Check a payload against the job.v1 branch for its kind (oneOf alone hides the cause)."""
    schema = schemas.get("job")
    branches = {
        b["properties"]["kind"]["const"]: b["properties"]["payload"] for b in schema["oneOf"]
    }
    if kind not in branches:
        raise InvalidJob(f"unknown job kind {kind!r}; use one of: {', '.join(sorted(branches))}")
    error = best_match(
        jsonschema.Draft202012Validator(branches[kind]).iter_errors(payload))
    if error is not None:
        where = "/".join(str(p) for p in error.absolute_path)
        raise InvalidJob(f"invalid {kind} job, payload {where or '<root>'}: {error.message}")


def _run_state(ref: str | None) -> str | None:
    """The state in a run folder's status.json, or None when there is no (readable) one."""
    if not ref:
        return None
    try:
        doc = json.loads((store.resolve(ref, must_exist=False) / "status.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return doc.get("state") if isinstance(doc, dict) else None


def enqueue(
    kind: str, payload: dict[str, Any], lane: str = "batch", ref: str | None = None,
    *, owner: str = "local", devices_required: int = 1, path: Path | None = None,
) -> int | None:
    """Put a job on the queue and return its id.

    A run whose folder is already done gets no job (None; load it by its ref). One that is
    already queued or running returns the id of the job that has it."""
    if lane not in LANES:
        raise InvalidJob(f"unknown lane {lane!r}; use one of: {', '.join(LANES)}")
    if devices_required != 1:
        raise InvalidJob("multi-device jobs need DDP (M2)")
    validate_job(kind, payload)
    if kind == "run" and ref:
        if _run_state(ref) == "done":
            return None
        with closing(connect(path)) as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE kind = 'run' AND ref = ? AND state IN "
                "('queued', 'running', 'cancelling') ORDER BY id LIMIT 1", (ref,)).fetchone()
        if row:
            return int(row["id"])
    with closing(connect(path)) as conn:
        cur = conn.execute(
            "INSERT INTO jobs (kind, lane, payload, ref, owner, devices_required, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, lane, json.dumps(payload), ref, owner, devices_required, time.time()),
        )
        return int(cur.lastrowid or 0)


def get(job_id: int, *, path: Path | None = None) -> sqlite3.Row:
    with closing(connect(path)) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"no job {job_id}")
    return row


def claim(
    worker_id: str, device: str, lanes: tuple[str, ...] = LANES,
    *, lease_seconds: float = LEASE_SECONDS, path: Path | None = None,
) -> sqlite3.Row | None:
    """Take the next queued job (interactive lane first, then oldest first) or None.

    Atomic: BEGIN IMMEDIATE takes the write lock before the read, so two workers never get
    the same job. A run whose folder is already done is marked done without being handed out."""
    marks = ",".join("?" for _ in lanes)
    with closing(connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            while True:
                row = conn.execute(
                    f"SELECT * FROM jobs WHERE state = 'queued' AND lane IN ({marks}) "
                    "ORDER BY CASE lane WHEN 'interactive' THEN 0 ELSE 1 END, id LIMIT 1",
                    lanes).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                now = time.time()
                if row["kind"] == "run" and _run_state(row["ref"]) == "done":
                    conn.execute("UPDATE jobs SET state = 'done', finished_at = ? WHERE id = ?",
                                 (now, row["id"]))
                    continue
                conn.execute(
                    "UPDATE jobs SET state = 'running', device = ?, worker_id = ?, "
                    "lease_until = ?, started_at = ? WHERE id = ?",
                    (device, worker_id, now + lease_seconds, now, row["id"]))
                claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
                conn.execute("COMMIT")
                return claimed
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


def renew(
    job_id: int, worker_id: str, *, lease_seconds: float = LEASE_SECONDS, path: Path | None = None,
) -> bool:
    """Extend a lease; False when the job is no longer this worker's (it was requeued)."""
    with closing(connect(path)) as conn:
        cur = conn.execute(
            "UPDATE jobs SET lease_until = ? WHERE id = ? AND worker_id = ? "
            "AND state IN ('running', 'cancelling')",
            (time.time() + lease_seconds, job_id, worker_id))
        return cur.rowcount == 1


def requeue_expired(
    *, max_attempts: int = MAX_ATTEMPTS, now: float | None = None, path: Path | None = None,
) -> list[int]:
    """Give back jobs whose lease ran out; after max_attempts they fail. Returns their ids."""
    now = time.time() if now is None else now
    with closing(connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT id, attempts, state FROM jobs WHERE state IN ('running', 'cancelling') "
            "AND lease_until < ?", (now,)).fetchall()
        for row in rows:
            attempts = row["attempts"] + 1
            if row["state"] == "cancelling":
                conn.execute("UPDATE jobs SET state = 'cancelled', finished_at = ? WHERE id = ?",
                             (now, row["id"]))
            elif attempts >= max_attempts:
                conn.execute(
                    "UPDATE jobs SET state = 'failed', attempts = ?, finished_at = ?, error = ? "
                    "WHERE id = ?",
                    (attempts, now, f"lease expired {attempts} times (the worker died)", row["id"]))
            else:
                conn.execute(
                    "UPDATE jobs SET state = 'queued', attempts = ?, worker_id = NULL, "
                    "device = NULL, lease_until = NULL WHERE id = ?", (attempts, row["id"]))
        conn.execute("COMMIT")
    return [int(r["id"]) for r in rows]


def finish(
    job_id: int, error: str | None = None, *, cancelled: bool = False, path: Path | None = None,
) -> str:
    """Record how a claimed job ended and return its final state (a cancel wins over both)."""
    with closing(connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            raise KeyError(f"no job {job_id}")
        if row["state"] == "cancelling" or cancelled:
            state = "cancelled"
        else:
            state = "failed" if error else "done"
        conn.execute("UPDATE jobs SET state = ?, error = COALESCE(?, error), finished_at = ? "
                     "WHERE id = ?", (state, error, time.time(), job_id))
        conn.execute("COMMIT")
    return state


def cancel(job_id: int, *, path: Path | None = None) -> str:
    """Cancel a job. Queued: cancelled at once. Running: STOP goes into its run folder and it
    becomes cancelled when `finish()` is called as its child exits. Returns the new state."""
    with closing(connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            raise KeyError(f"no job {job_id}")
        state = row["state"]
        if state == "queued":
            state = "cancelled"
            conn.execute("UPDATE jobs SET state = 'cancelled', finished_at = ? WHERE id = ?",
                         (time.time(), job_id))
        elif state == "running":
            state = "cancelling"
            conn.execute("UPDATE jobs SET state = 'cancelling' WHERE id = ?", (job_id,))
        conn.execute("COMMIT")
    if state == "cancelling" and row["ref"]:
        folder = store.resolve(row["ref"], must_exist=False)
        if folder.is_dir():
            (folder / STOP_FILE).write_text("", encoding="utf-8")
    return state


def record(
    job_id: int, result: dict[str, Any] | None = None, error: str | None = None,
    *, path: Path | None = None,
) -> None:
    """Store what a job produced (or why it broke) without changing its state."""
    with closing(connect(path)) as conn:
        conn.execute("UPDATE jobs SET result = COALESCE(?, result), error = COALESCE(?, error) "
                     "WHERE id = ?",
                     (None if result is None else json.dumps(result, default=str), error, job_id))


def release(job_id: int, worker_id: str, *, path: Path | None = None) -> str | None:
    """Hand a claimed job back unharmed (a worker shutting down, or a job that doesn't fit yet).

    It returns to the queue with its attempts unchanged; a cancel in progress completes."""
    with closing(connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT state FROM jobs WHERE id = ? AND worker_id = ? "
                           "AND state IN ('running', 'cancelling')", (job_id, worker_id)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return None
        if row["state"] == "cancelling":
            state = "cancelled"
            conn.execute("UPDATE jobs SET state = 'cancelled', finished_at = ? WHERE id = ?",
                         (time.time(), job_id))
        else:
            state = "queued"
            conn.execute("UPDATE jobs SET state = 'queued', worker_id = NULL, device = NULL, "
                         "lease_until = NULL, started_at = NULL WHERE id = ?", (job_id,))
        conn.execute("COMMIT")
    return state


def list_jobs(state: str | None = None, *, path: Path | None = None) -> list[sqlite3.Row]:
    with closing(connect(path)) as conn:
        if state:
            return conn.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY id", (state,)).fetchall()
        return conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()


def cached_memory(key: str, *, path: Path | None = None) -> int | None:
    """Bytes a job needed on a device, as an earlier probe measured them."""
    with closing(connect(path)) as conn:
        row = conn.execute("SELECT bytes FROM memory_cache WHERE key = ?", (key,)).fetchone()
    return int(row["bytes"]) if row else None


def cache_memory(key: str, nbytes: int, *, path: Path | None = None) -> None:
    with closing(connect(path)) as conn:
        conn.execute("INSERT OR REPLACE INTO memory_cache (key, bytes) VALUES (?, ?)",
                     (key, nbytes))


def cancel_prefix(ref: str, *, path: Path | None = None) -> list[int]:
    """Cancel every queued or running job whose ref is `ref` or lies under it (a study)."""
    under = ref.rstrip("/") + "/"
    with closing(connect(path)) as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE state IN ('queued', 'running') "
            "AND (ref = ? OR substr(ref, 1, ?) = ?)", (ref, len(under), under)).fetchall()
    ids = [int(r["id"]) for r in rows]
    for job_id in ids:
        cancel(job_id, path=path)
    return ids


def format_jobs(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "no jobs"
    lines = []
    for r in rows:
        where = f" on {r['device']}" if r["device"] else ""
        what = r["ref"] or json.loads(r["payload"]).get("preset") or ""
        note = f"  ({r['error']})" if r["error"] else ""
        lines.append(f"#{r['id']}  {r['state']:<10} {r['lane']:<11} {r['kind']:<13} "
                     f"{what}{where}{note}")
    return "\n".join(lines)
