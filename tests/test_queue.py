import json
import sqlite3
from contextlib import closing

import pytest

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


def test_enqueue_stores_a_queued_job(home):
    payload = {"model": "nanoscope.models.bigram:Bigram", "preset": "tinystories-5min", "seed": 1}
    job_id = queue.enqueue("run", payload, ref="studies/s/v/seed-1")
    with closing(queue.connect()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert (row["kind"], row["lane"], row["state"], row["owner"]) == (
        "run", "batch", "queued", "local")
    assert row["ref"] == "studies/s/v/seed-1"
    assert json.loads(row["payload"]) == payload
    assert row["devices_required"] == 1 and row["attempts"] == 0


def test_enqueue_refuses_an_invalid_payload(home):
    with pytest.raises(queue.InvalidJob, match="model"):
        queue.enqueue("run", {"preset": "tinystories-5min"})
    with pytest.raises(queue.InvalidJob, match="seed"):
        queue.enqueue("run", {"model": "m:M", "preset": "p", "seed": "one"})
    with pytest.raises(queue.InvalidJob):
        queue.enqueue("teleport", {})
    with pytest.raises(queue.InvalidJob, match="lane"):
        queue.enqueue("prepare-data", {"preset": "p"}, lane="urgent")
    with closing(queue.connect()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
