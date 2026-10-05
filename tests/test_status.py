import json
import os
import signal

import pytest
from fakes import tiny
from helpers import assert_valid

from nanoscope import paths, run
from nanoscope.models import Bigram
from nanoscope.status import HEARTBEAT_SECONDS, StatusFile


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_write_is_atomic_valid_and_complete(tmp_path):
    status = StatusFile(tmp_path / "run", 20, device="cpu")
    status.write("preparing")
    doc = json.loads((tmp_path / "run" / "status.json").read_text())
    assert_valid("status", doc)
    assert doc["state"] == "preparing" and doc["max_steps"] == 20 and doc["device"] == "cpu"
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_job_id_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_JOB_ID", "job-7")
    assert StatusFile(tmp_path, 1).doc["job_id"] == "job-7"


def test_heartbeat_writes_at_most_every_five_seconds(tmp_path):
    clock = Clock()
    status = StatusFile(tmp_path, 100, clock=clock)
    path = tmp_path / "status.json"
    status(1, {})  # the first step flips to running immediately
    assert json.loads(path.read_text())["state"] == "running"
    stamp = path.stat().st_mtime_ns
    clock.now += HEARTBEAT_SECONDS - 1
    status(2, {})
    assert path.stat().st_mtime_ns == stamp  # too soon, nothing written
    clock.now += 2
    status(3, {})
    assert json.loads(path.read_text())["step"] == 3


def test_fail_records_the_error_and_ctrl_c_is_stopped(tmp_path):
    status = StatusFile(tmp_path, 10)
    try:
        raise ValueError("bad batch")
    except ValueError as exc:
        status.fail(exc)
    doc = json.loads((tmp_path / "status.json").read_text())
    assert doc["state"] == "failed"
    assert doc["error"]["type"] == "ValueError" and doc["error"]["message"] == "bad batch"
    assert len(doc["error"]["traceback"]) <= 20
    assert_valid("status", doc)

    status.fail(KeyboardInterrupt())
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "stopped"


def test_unknown_states_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown run state"):
        StatusFile(tmp_path, 1).write("paused")


# The lifecycle through run():


def _status(run_dir):
    return json.loads((run_dir / "status.json").read_text())


def test_lifecycle_goes_preparing_running_done(fake_data, monkeypatch):
    seen = []
    real_write = StatusFile.write

    def spy(self, state, *args, **kwargs):
        if not seen or seen[-1] != state:
            seen.append(state)
        return real_write(self, state, *args, **kwargs)

    monkeypatch.setattr(StatusFile, "write", spy)
    result = run(Bigram, tiny(), device="cpu", progress=False)
    assert seen == ["preparing", "running", "done"]
    doc = _status(result.run_dir)
    assert doc["state"] == "done" and doc["step"] == 20 and doc["max_steps"] == 20
    assert_valid("status", doc)


def test_a_crash_is_recorded_as_failed_with_its_error(fake_data):
    def boom(step, row):
        if step == 3:
            raise RuntimeError("out of memory")

    run_dir = paths.runs_dir() / "crashy"
    with pytest.raises(RuntimeError, match="out of memory"):
        run(Bigram, tiny(), device="cpu", output_dir=run_dir, on_step=boom, progress=False)
    doc = _status(run_dir)
    assert doc["state"] == "failed"
    assert doc["error"]["type"] == "RuntimeError" and doc["error"]["message"] == "out of memory"
    assert_valid("status", doc)


def test_ctrl_c_during_training_is_stopped(fake_data):
    def interrupt(step, row):
        if step == 4:
            os.kill(os.getpid(), signal.SIGINT)

    run_dir = paths.runs_dir() / "ctrlc"
    result = run(Bigram, tiny(), device="cpu", output_dir=run_dir, on_step=interrupt,
                 progress=False)
    assert result.train_result.stopped_early
    assert _status(run_dir)["state"] == "stopped" and _status(run_dir)["step"] == 4


def test_a_refused_resume_leaves_the_status_alone(fake_data):
    run_dir = paths.runs_dir() / "kept"
    run(Bigram, tiny(), device="cpu", output_dir=run_dir, progress=False)
    before = (run_dir / "status.json").read_text()
    with pytest.raises(ValueError, match="different config"):
        run(Bigram, tiny(), device="cpu", output_dir=run_dir, learning_rate=1e-3, progress=False)
    assert (run_dir / "status.json").read_text() == before
