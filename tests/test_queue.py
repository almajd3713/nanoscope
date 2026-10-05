import sqlite3
from contextlib import closing

from nanoscope import paths, queue


def test_create_makes_a_wal_database_with_the_tables(home):
    with closing(queue.connect()) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in rows}
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert mode == "wal"
    assert {"jobs", "meta"} <= tables
    assert {"id", "kind", "lane", "payload", "state", "ref", "owner", "devices_required", "device",
            "worker_id", "lease_until", "attempts", "error", "created_at"} <= columns
    assert paths.queue_db() == home / "queue.db"
    assert queue.schema_version() == queue.SCHEMA_VERSION


def test_create_is_idempotent(home):
    queue.connect().close()
    queue.connect().close()
    with closing(sqlite3.connect(paths.queue_db())) as conn:
        assert conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 1
