import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import torch
from fakes import tiny
from test_study import PRESET_SRC

from nanoscope import hardware, paths, queue
from nanoscope.cli import main
from nanoscope.dataset import load_data
from nanoscope.jobs.payload import preset_fields
from nanoscope.jobs.runner import ContainerRunner, SubprocessRunner, job_env
from nanoscope.jobs.worker import Worker
from nanoscope.study import load_study

GIB = 2**30


def run_payload(**extra):
    return {"model": "bigram", "kwargs": {"d_model": 8}, **preset_fields(tiny(max_steps=6)),
            **extra}


def add(*args, **kwargs) -> int:
    job_id = queue.enqueue(*args, **kwargs)
    assert job_id is not None
    return job_id


class FakeRunner:
    """A child that ends when told to (`finish`) or when terminated."""

    started: list[int] = []

    def __init__(self):
        self.code = None

    def start(self, job_id, log, env):
        FakeRunner.started.append(job_id)

    def poll(self):
        return self.code

    def terminate(self, kill=False):
        self.code = -9 if kill else -15


@pytest.fixture(autouse=True)
def _fresh_fake():
    FakeRunner.started = []


def sleeper(job_id):
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def test_runner_subprocess_runs_and_terminates(home, tmp_path):
    runner = SubprocessRunner(lambda job_id: [sys.executable, "-c", "print('hi from job')"])
    runner.start(7, tmp_path / "7.log", {})
    deadline = time.time() + 20
    while runner.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    assert runner.poll() == 0
    assert "hi from job" in (tmp_path / "7.log").read_text()

    slow = SubprocessRunner(sleeper)
    slow.start(8, tmp_path / "8.log", {})
    assert slow.poll() is None
    slow.terminate()
    deadline = time.time() + 20
    while slow.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    assert slow.poll() == -signal.SIGTERM


def test_runner_container_stub_names_the_deployments(tmp_path):
    with pytest.raises(NotImplementedError, match="deployments B/C"):
        ContainerRunner().start(1, tmp_path / "x.log", {})


def test_env_allowlist_keeps_secrets_from_jobs_that_do_not_need_them():
    environ = {"HF_TOKEN": "h", "WANDB_API_KEY": "w", "PATH": "/bin"}
    plain = job_env("run", run_payload(), environ)
    assert plain == {"PATH": "/bin"}
    hub = job_env("run", run_payload(push_to_hub="me/runs"), environ)
    assert hub == {"HF_TOKEN": "h", "PATH": "/bin"}
    wandb = job_env("run", run_payload(wandb=True), environ)
    assert wandb == {"WANDB_API_KEY": "w", "PATH": "/bin"}
    assert job_env("prepare-data", {"preset": "p"}, environ) == {"HF_TOKEN": "h", "PATH": "/bin"}
    assert job_env("bench", {"model": "bigram"}, environ) == {"PATH": "/bin"}


@pytest.mark.usefixtures("fake_data")
def test_run_job_trains_in_a_fresh_process_and_records_the_result(home, monkeypatch):
    job_id = add("run", run_payload(), ref="jobs-test/a")
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    assert stopped.value.code == 0
    row = queue.get(job_id)
    result = json.loads(row["result"])
    assert result["state"] == "done" and result["final_step"] == 6
    assert (paths.runs_dir() / "jobs-test" / "a" / "status.json").exists()
    status = json.loads((paths.runs_dir() / "jobs-test" / "a" / "status.json").read_text())
    assert status["job_id"] == str(job_id)


def test_offline_refuses_jobs_that_need_the_network(home, monkeypatch):
    monkeypatch.setenv("NANOSCOPE_JOBS_OFFLINE", "1")
    pushing = add("run", run_payload(push_to_hub="me/runs"), ref="jobs-test/off")
    syncing = add("sync-hub", {"ref": "x/y", "repo": "me/runs"})
    for job_id in (pushing, syncing):
        queue.claim("w", "cpu")
        with pytest.raises(SystemExit) as stopped:
            main(["run-job", str(job_id)])
        assert stopped.value.code == 1
        assert "NANOSCOPE_JOBS_OFFLINE=1" in queue.get(job_id)["error"]
    assert "me/runs" in queue.get(pushing)["error"]
    assert "Hugging Face Hub" in queue.get(syncing)["error"]


@pytest.mark.usefixtures("fake_data")
def test_offline_still_trains_on_cached_data(home, monkeypatch):
    load_data(tiny(max_steps=6))   # the cache is filled while the network is allowed
    monkeypatch.setenv("NANOSCOPE_JOBS_OFFLINE", "1")
    job_id = add("run", run_payload(), ref="jobs-test/offline-ok")
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    assert stopped.value.code == 0


def test_run_job_records_an_error_and_exits_nonzero(home, capsys):
    job_id = add("run", {"model": "no_such_module:Nope", "preset": "tinystories-5min"})
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    assert stopped.value.code == 1
    assert "no_such_module" in queue.get(job_id)["error"]


def drive(worker, until, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        worker.tick()
        if until():
            return
        time.sleep(0.05)
    raise AssertionError("worker did not get there in time")


@pytest.mark.usefixtures("fake_data")
def test_worker_runs_a_queued_job_in_a_child_process(home):
    load_data(tiny())  # the child finds the tokens cached
    job_id = add("run", run_payload(), ref="jobs-test/w")
    worker = Worker("cpu", 1, poll_seconds=0.05)
    drive(worker, lambda: queue.get(job_id)["state"] == "done")
    assert (paths.runs_dir() / "jobs-test" / "w" / "latest.json").exists()
    assert (paths.job_logs_dir() / f"{job_id}.log").exists()


def test_worker_marks_a_crashing_child_failed_with_its_error(home):
    job_id = add("run", {"model": "no_such_module:Nope", "preset": "tinystories-5min"})
    worker = Worker("cpu", 1, poll_seconds=0.05)
    drive(worker, lambda: queue.get(job_id)["state"] == "failed")
    assert "no_such_module" in queue.get(job_id)["error"]


def test_worker_sigterm_hands_the_job_back(home, tmp_path):
    job_id = add("prepare-data", {"preset": "tinystories-5min"})
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent("""
        import sys
        from nanoscope.jobs.runner import SubprocessRunner
        from nanoscope.jobs.worker import Worker

        sleeper = lambda job_id: [sys.executable, "-c", "import time; time.sleep(60)"]
        Worker("cpu", 1, runner_factory=lambda: SubprocessRunner(sleeper),
               poll_seconds=0.05, stop_grace=2).run()
    """))
    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        end = time.time() + 30
        while queue.get(job_id)["state"] != "running" and time.time() < end:
            time.sleep(0.05)
        assert queue.get(job_id)["state"] == "running"
        while not list(paths.workers_dir().glob("*.json")) and time.time() < end:
            time.sleep(0.05)  # the worker writes its file at the end of the tick that claimed
        assert list(paths.workers_dir().glob("*.json"))
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    row = queue.get(job_id)
    assert (row["state"], row["worker_id"], row["attempts"]) == ("queued", None, 0)
    assert not list(paths.workers_dir().glob("*.json"))


def test_status_workers_lists_each_worker(home, capsys):
    main(["status", "--workers"])
    assert "no workers running" in capsys.readouterr().out
    job_id = add("prepare-data", {"preset": "p"})
    worker = Worker("cuda:1", 3, worker_id="box-1", runner_factory=FakeRunner,
                    probe=lambda job, payload: 0, free_memory=lambda d: GIB)
    worker.tick()
    main(["status", "--workers"])
    out = capsys.readouterr().out
    assert f"box-1  cuda:1  1/3 slots  #{job_id}  seen" in out


def test_timeout_stops_the_child_and_fails_the_job(home):
    job_id = add("prepare-data", {"preset": "p"})
    worker = Worker("cpu", 1, runner_factory=FakeRunner, timeout=0.2, stop_grace=0.2,
                    poll_seconds=0.05)
    drive(worker, lambda: queue.get(job_id)["state"] == "failed")
    assert queue.get(job_id)["error"] == "timeout after 0.2s"


def test_a_job_can_set_its_own_timeout(home):
    job_id = add("prepare-data", {"preset": "p", "timeout": 0.1})
    worker = Worker("cpu", 1, runner_factory=FakeRunner, timeout=999, stop_grace=0.1)
    drive(worker, lambda: queue.get(job_id)["state"] == "failed")
    assert queue.get(job_id)["error"] == "timeout after 0.1s"


def test_threads_cpu_children_split_the_cores_and_gpu_children_split_the_memory(monkeypatch):
    threads, fractions = [], []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: fractions.append((fraction, str(device))))
    hardware.configure_device("cpu", 2)
    assert threads == [hardware.cpu_threads(2)]
    hardware.configure_device("cuda:1", 4)
    assert fractions == [(hardware.HEADROOM / 4, "cuda:1")]


def test_admission_keeps_a_job_that_does_not_fit_queued(home):
    free = iter([5 * GIB, 1 * GIB, 1 * GIB])
    probes = []

    def probe(job, payload):
        probes.append(job["id"])
        return 4 * GIB

    ids = [add("run", run_payload(), ref=f"jobs-test/adm{i}") for i in range(2)]
    worker = Worker("cuda:0", 2, runner_factory=FakeRunner, probe=probe,
                    free_memory=lambda device: next(free))
    worker.tick()
    assert [queue.get(i)["state"] for i in ids] == ["running", "queued"]
    assert len(worker.active) == 1
    assert probes == [ids[0]]  # the second job's need came from the cache
    assert queue.get(ids[1])["attempts"] == 0


def test_admission_fails_a_job_that_never_fits(home):
    job_id = add("run", run_payload(), ref="jobs-test/big")
    worker = Worker("cuda:0", 2, runner_factory=FakeRunner, probe=lambda j, p: 40 * GIB,
                    free_memory=lambda device: 8 * GIB)
    worker.tick()
    row = queue.get(job_id)
    assert row["state"] == "failed" and "needs about 40.0 GiB" in row["error"]


def test_admission_cache_is_keyed_by_the_job_and_the_device(home):
    from nanoscope.jobs.worker import memory_key

    a = memory_key(run_payload(), "cuda:0")
    assert a == memory_key(run_payload(), "cuda:0")
    assert a != memory_key(run_payload(), "cuda:1")
    assert a != memory_key({**run_payload(), "kwargs": {"d_model": 16}}, "cuda:0")


def test_interactive_first_a_worker_starts_the_interactive_job_before_older_batch_jobs(home):
    class Quick(FakeRunner):
        def start(self, job_id, log, env):
            super().start(job_id, log, env)
            self.code = 0

    batch = [add("prepare-data", {"preset": f"b{i}"}) for i in range(2)]
    urgent = add("prepare-data", {"preset": "now"}, lane="interactive")
    worker = Worker("cpu", 1, runner_factory=Quick, exit_when_idle=True, poll_seconds=0.01)
    worker.run()
    assert FakeRunner.started == [urgent, *batch]
    assert [queue.get(i)["state"] for i in (urgent, *batch)] == ["done"] * 3


@pytest.mark.usefixtures("fake_data")
def test_sigkill_a_killed_worker_loses_its_lease_and_the_run_resumes(tmp_path, monkeypatch):
    from nanoscope.compare import load_runs

    model = tmp_path / "slow_model.py"
    model.write_text(textwrap.dedent("""
        import time
        from nanoscope.models import Bigram

        class SlowBigram(Bigram):
            def forward(self, tokens):
                time.sleep(0.03)
                return super().forward(tokens)
    """))
    path = tmp_path / "study.py"
    path.write_text(PRESET_SRC + textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from slow_model import SlowBigram

        study = Study("toy", preset=preset, seeds=1, budget=Tokens(4 * 32 * 60), baseline="a")
        study.add("a", SlowBigram, d_model=8)
        study.add("b", SlowBigram, d_model=16)
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # the worker's children import the model too
    study = load_study(path)
    load_data(study.preset)
    ids = study.enqueue()
    worker_cmd = [sys.executable, "-m", "nanoscope.cli", "worker", "--lease-seconds", "2",
                  "--exit-when-idle"]
    worker = subprocess.Popen(worker_cmd)
    child = None
    try:
        end = time.time() + 120
        while time.time() < end and child is None:
            for status in study.dir.glob("*/seed-*/status.json"):
                run_dir = status.parent
                if (run_dir / "latest.json").exists() and json.loads(
                        status.read_text())["state"] == "running":
                    child = json.loads(status.read_text())["pid"]
            time.sleep(0.05)
        assert child, "no run reached a checkpoint"
        worker.kill()  # its training process must die with it (PR_SET_PDEATHSIG)
        worker.wait()
        end = time.time() + 10
        while time.time() < end and os.path.exists(f"/proc/{child}"):
            time.sleep(0.05)
        assert not os.path.exists(f"/proc/{child}"), "the child outlived its killed worker"
    finally:
        if worker.poll() is None:
            worker.kill()
    assert any(queue.get(i)["state"] == "running" for i in ids)

    subprocess.run(worker_cmd, check=True, timeout=180)

    rows = [queue.get(i) for i in ids]
    assert [r["state"] for r in rows] == ["done", "done"]
    assert sum(r["attempts"] for r in rows) == 1
    interrupted = {v: load_runs(study.dir / v)[0].final("val_loss") for v in ("a", "b")}

    monkeypatch.setenv("NANOSCOPE_HOME", str(tmp_path / "home-uninterrupted"))
    load_data(study.preset)
    clean = load_study(path)
    clean.run()
    straight = {v: load_runs(clean.dir / v)[0].final("val_loss") for v in ("a", "b")}
    assert interrupted == pytest.approx(straight, abs=1e-6)


@pytest.mark.skipif(sys.platform != "linux", reason="uses PR_SET_PDEATHSIG")
def test_a_sigkilled_worker_does_not_leave_its_child_running(home, tmp_path):
    pidfile = tmp_path / "child.pid"
    add("prepare-data", {"preset": "tinystories-5min"})
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        from nanoscope.jobs.runner import SubprocessRunner
        from nanoscope.jobs.worker import Worker

        code = ("import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); "
                "time.sleep(60)")
        Worker("cpu", 1, runner_factory=lambda: SubprocessRunner(
            lambda job_id: [sys.executable, "-c", code]), poll_seconds=0.05).run()
    """))
    worker = subprocess.Popen([sys.executable, str(script)])
    try:
        end = time.time() + 30
        while not (pidfile.exists() and pidfile.read_text()) and time.time() < end:
            time.sleep(0.05)
        child = int(pidfile.read_text())
        worker.kill()
        worker.wait()
        end = time.time() + 10
        while time.time() < end:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        os.kill(child, signal.SIGKILL)
        raise AssertionError("the child outlived its killed worker")
    finally:
        if worker.poll() is None:
            worker.kill()
