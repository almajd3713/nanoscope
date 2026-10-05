"""The job queue: one SQLite file in WAL mode, shared by the API, the CLI and the workers.

Every write goes through `connect()` so the schema exists and every connection waits for
a competing writer instead of failing at once.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from nanoscope import paths

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
