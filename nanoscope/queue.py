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

from nanoscope import paths, schemas

LANES = ("interactive", "batch")

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
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs (state, lane, id);
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


def enqueue(
    kind: str, payload: dict[str, Any], lane: str = "batch", ref: str | None = None,
    *, path: Path | None = None,
) -> int:
    """Put a job on the queue and return its id."""
    if lane not in LANES:
        raise InvalidJob(f"unknown lane {lane!r}; use one of: {', '.join(LANES)}")
    validate_job(kind, payload)
    with closing(connect(path)) as conn:
        cur = conn.execute(
            "INSERT INTO jobs (kind, lane, payload, ref, created_at) VALUES (?, ?, ?, ?, ?)",
            (kind, lane, json.dumps(payload), ref, time.time()),
        )
        return int(cur.lastrowid or 0)
