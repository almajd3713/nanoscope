import json
import sqlite3
import threading
import time
from contextlib import closing

import pytest

from nanoscope import paths, queue


def add(*args, **kwargs) -> int:
    job_id = queue.enqueue(*args, **kwargs)
    assert job_id is not None
    return job_id


def take(*args, **kwargs) -> sqlite3.Row:
    job = queue.claim(*args, **kwargs)
    assert job is not None
    return job


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
    job_id = add("run", payload, ref="studies/s/v/seed-1")
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
        add("run", {"model": "m:M", "preset": "p", "seed": "one"})
    with pytest.raises(queue.InvalidJob):
        queue.enqueue("teleport", {})
    with pytest.raises(queue.InvalidJob, match="lane"):
        add("prepare-data", {"preset": "p"}, lane="urgent")
    with closing(queue.connect()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


RUN = {"model": "nanoscope.models.bigram:Bigram", "preset": "tinystories-5min"}


def _status(ref, state):
    folder = paths.runs_dir() / ref
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "status.json").write_text(json.dumps({"schema": 1, "state": state}))
    return folder


def test_claim_takes_the_oldest_job_and_leases_it(home):
    first = add("prepare-data", {"preset": "a"})
    add("prepare-data", {"preset": "b"})
    job = take("w1", "cpu")
    assert job["id"] == first and job["state"] == "running"
    assert (job["worker_id"], job["device"]) == ("w1", "cpu")
    assert job["lease_until"] > time.time()
    assert take("w1", "cpu")["id"] != first
    assert queue.claim("w1", "cpu") is None


def test_lanes_interactive_before_batch(home):
    batch = add("prepare-data", {"preset": "a"})
    urgent = add("prepare-data", {"preset": "b"}, lane="interactive")
    assert take("w", "cpu")["id"] == urgent
    assert take("w", "cpu")["id"] == batch
    add("prepare-data", {"preset": "c"}, lane="interactive")
    assert queue.claim("w", "cpu", lanes=("batch",)) is None


def test_claim_two_threads_never_double_claim(home):
    ids = [add("prepare-data", {"preset": str(i)}) for i in range(40)]
    got, lock = [], threading.Lock()

    def work(name):
        while (job := queue.claim(name, "cpu")) is not None:
            with lock:
                got.append(job["id"])

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(got) == sorted(ids)


def test_lease_expired_jobs_requeue_then_fail(home):
    job_id = add("prepare-data", {"preset": "a"})
    for attempt in (1, 2):
        take("w", "cpu", lease_seconds=-1)
        assert queue.requeue_expired() == [job_id]
        row = queue.get(job_id)
        assert (row["state"], row["attempts"], row["worker_id"]) == ("queued", attempt, None)
    take("w", "cpu", lease_seconds=-1)
    queue.requeue_expired()
    row = queue.get(job_id)
    assert row["state"] == "failed" and "lease expired" in row["error"]


def test_lease_renew_keeps_a_job_and_refuses_a_stranger(home):
    job_id = add("prepare-data", {"preset": "a"})
    take("w", "cpu", lease_seconds=-1)
    assert queue.renew(job_id, "other") is False
    assert queue.renew(job_id, "w") is True
    assert queue.requeue_expired() == []


def test_cancel_a_queued_job(home):
    job_id = add("prepare-data", {"preset": "a"})
    assert queue.cancel(job_id) == "cancelled"
    assert queue.claim("w", "cpu") is None


def test_cancel_a_running_job_writes_stop_and_ends_cancelled(home):
    folder = _status("r1", "running")
    job_id = add("run", RUN, ref="r1")
    take("w", "cpu")
    assert queue.cancel(job_id) == "cancelling"
    assert (folder / "STOP").exists()
    assert queue.get(job_id)["state"] == "cancelling"
    assert queue.finish(job_id) == "cancelled"


def test_finish_records_done_and_failed(home):
    a = add("prepare-data", {"preset": "a"})
    b = add("prepare-data", {"preset": "b"})
    take("w", "cpu")
    take("w", "cpu")
    assert queue.finish(a) == "done"
    assert queue.finish(b, "boom") == "failed"
    assert queue.get(b)["error"] == "boom"


def test_duplicate_submits(home):
    _status("done-run", "done")
    assert queue.enqueue("run", RUN, ref="done-run") is None
    first = add("run", RUN, ref="r2")
    assert queue.enqueue("run", RUN, ref="r2") == first
    take("w", "cpu")
    assert queue.enqueue("run", RUN, ref="r2") == first
    queue.finish(first, "boom")
    assert queue.enqueue("run", RUN, ref="r2") != first


def test_folders_win_over_a_queued_job(home):
    job_id = add("run", RUN, ref="r3")
    nxt = add("run", RUN, ref="r4")
    _status("r3", "done")
    assert take("w", "cpu")["id"] == nxt
    assert queue.get(job_id)["state"] == "done"


def test_devices_required_over_one_is_refused(home):
    with pytest.raises(queue.InvalidJob, match=r"multi-device jobs need DDP \(M2\)"):
        queue.enqueue("run", RUN, devices_required=2)
    job_id = add("run", RUN, owner="alice")
    row = queue.get(job_id)
    assert (row["owner"], row["devices_required"]) == ("alice", 1)


def test_jobs_command_lists_and_cancels(home, capsys):
    from nanoscope.cli import main

    main(["jobs"])
    assert capsys.readouterr().out.strip() == "no jobs"
    first = add("prepare-data", {"preset": "tinystories-5min"})
    second = add("run", RUN, lane="interactive", ref="r9")
    main(["jobs"])
    out = capsys.readouterr().out
    assert f"#{first}  queued" in out and "prepare-data" in out and "tinystories-5min" in out
    assert f"#{second}  queued     interactive run" in out and "r9" in out

    main(["jobs", "cancel", str(first)])
    assert capsys.readouterr().out.strip() == f"job {first}: cancelled"
    main(["jobs", "--state", "queued"])
    out = capsys.readouterr().out
    assert f"#{first}" not in out and f"#{second}" in out


def test_lease_a_live_worker_is_not_requeued_by_its_own_tick(home):
    job_id = add("prepare-data", {"preset": "a"})
    take("w", "cpu", lease_seconds=-1)
    assert queue.requeue_expired(skip_worker="w") == []
    assert queue.get(job_id)["state"] == "running"
    assert queue.requeue_expired(skip_worker="someone-else") == [job_id]
