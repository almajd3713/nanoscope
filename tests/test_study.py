import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch
from fakes import tiny
from helpers import assert_valid

from nanoscope import FLOPs, Study, Tokens, compare, paths, run
from nanoscope.dataset import load_data
from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.sizing import count_params, match_params
from nanoscope.study import load_study

pytestmark = pytest.mark.usefixtures("fake_data")
SMALL = dict(d_model=16, n_layers=1, n_heads=2)

PRESET_SRC = """
from nanoscope import Preset, Study, Tokens
from nanoscope.models import Bigram

preset = Preset(
    name="test-tiny", dataset="fake/stories", tokenizer="bpe", vocab_size=300,
    context_length=32, max_steps=20, batch_size=4, learning_rate=1e-2, warmup_steps=2,
    train_docs=200, tokenizer_train_docs=100, eval_docs=10, eval_interval=10,
    sample_interval=10, sample_length=20, checkpoint_interval=10,
)
"""


def write_study(path, mode="explore", prediction=1.5, seeds=3):
    path.write_text(PRESET_SRC + textwrap.dedent(f"""
        study = Study("toy", preset=preset, seeds={seeds}, budget=Tokens(4 * 32 * 6),
                      baseline="small", mode="{mode}")
        study.add("small", Bigram, d_model=8)
        study.add("wide", Bigram, d_model=32)
        study.predict("wide", val_bpb={prediction})
    """))


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    (tmp_path / ".gitignore").write_text("runs/\ncache/\nexperiments/\nhome/\n")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def commit_all(repo, message="study"):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)


def test_study_trains_every_variant_and_seed_then_reports(tmp_path):
    study = Study("toy", preset=tiny(), seeds=2, budget=Tokens(4 * 32 * 6), baseline="small")
    study.add("small", Bigram, d_model=8)
    study.add("wide", Bigram, d_model=32)
    study.run(devices=["cpu"])

    for variant in ("small", "wide"):
        for seed in (0, 1):
            config = json.loads((study.dir / variant / f"seed-{seed}" / "config.json").read_text())
            assert config["study"] == {"name": "toy", "variant": variant, "mode": "explore"}
            assert config["preset"]["max_steps"] == 6
    report = study.report()
    assert [row["label"] for row in report.comparison.rows] == ["small", "wide"]
    assert report.comparison.baseline == "small"
    out = paths.reports_dir() / "toy"
    assert {p.name for p in out.iterdir()} == {"report.md", "results.json", "curves.png"}
    assert "## Results" in (out / "report.md").read_text()


def test_rerunning_a_study_skips_finished_runs():
    study = Study("toy", preset=tiny(), seeds=1, budget=Tokens(4 * 32 * 4))
    study.add("a", Bigram)
    study.run(devices=["cpu"])
    ckpt = next((study.dir / "a" / "seed-0" / "checkpoints").iterdir())
    before = ckpt.stat().st_mtime_ns
    study.run(devices=["cpu"])
    assert ckpt.stat().st_mtime_ns == before


def test_flops_budget_gives_cheaper_models_more_tokens():
    study = Study("flops", preset=tiny(), seeds=1, budget=FLOPs(5e8))
    study.add("bigram", Bigram)
    study.add("gpt2", GPT2, **SMALL)
    sizes = study.sizes()
    jobs = {j.variant.name: j for j in study.jobs()}

    cheap, costly = sorted(jobs, key=lambda name: sizes[name]["flops_per_token"])
    assert jobs[cheap].preset.max_steps > jobs[costly].preset.max_steps
    tokens_per_step = 4 * 32
    for name, job in jobs.items():
        spent = job.preset.max_steps * tokens_per_step * sizes[name]["flops_per_token"]
        assert 5e8 <= spent < 5e8 + tokens_per_step * sizes[name]["flops_per_token"]


def test_match_params_refuses_mismatched_variants_until_resized():
    study = Study("match", preset=tiny(), seeds=1, match="params")
    study.add("gpt2", GPT2, **SMALL)
    study.add("modern", Modern, **SMALL)
    with pytest.raises(ValueError, match="within 2%"):
        study.jobs()

    target = study.sizes()["gpt2"]["non_embedding_params"]
    resized = match_params(Modern, target, "ffn_hidden", range(8, 256, 2), vocab_size=300,
                           **SMALL)
    study.variants.pop("modern")
    study.add("modern", Modern, ffn_hidden=resized["ffn_hidden"], **SMALL)
    assert len(study.jobs()) == 2


def test_match_params_helper_lands_close():
    gpt2 = count_params(GPT2(vocab_size=300, **SMALL))[1]
    kw = match_params(Modern, gpt2, "ffn_hidden", range(8, 256, 2), vocab_size=300, **SMALL)
    assert abs(count_params(Modern(**kw))[1] - gpt2) / gpt2 < 0.02


def test_record_mode_needs_a_committed_study_and_a_clean_tree(repo):
    path = repo / "study.py"
    write_study(path, mode="record")
    with pytest.raises(ValueError, match="commit study.py first"):
        load_study(path).run(devices=["cpu"])

    commit_all(repo)
    (repo / "notes.txt").write_text("scratch")
    with pytest.raises(ValueError, match="clean git tree"):
        load_study(path).run(devices=["cpu"])


def test_record_mode_stores_the_commit_and_freezes_predictions(repo):
    path = repo / "study.py"
    write_study(path, mode="record")
    commit_all(repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    study = load_study(path)
    study.run(devices=["cpu"])

    config = json.loads((study.dir / "wide" / "seed-0" / "config.json").read_text())
    assert config["study"]["mode"] == "record"
    assert config["study"]["commit"] == head
    assert_valid("study", json.loads((study.dir / "study.json").read_text()))
    report = str(study.report())
    assert "preregistered: study.py committed in" in report
    assert "| wide | val_bpb | 1.500 |" in report

    write_study(path, mode="record", prediction=1.2)
    commit_all(repo, "move the goalposts")
    with pytest.raises(ValueError, match="predictions .* changed after the study started"):
        load_study(path).run(devices=["cpu"])


def test_record_results_are_never_overwritten(repo):
    path = repo / "study.py"
    write_study(path, mode="record")
    commit_all(repo)
    study = load_study(path)
    study.run(devices=["cpu"])
    job = study.jobs()[0]
    with pytest.raises(ValueError, match="never overwritten"):
        run(job.variant.model_cls, job.preset, seed=job.seed, device="cpu",
            output_dir=job.output_dir, resume=False, **job.variant.kwargs)


def test_compare_flags_explore_runs_mixed_into_record_results(repo):
    path = repo / "study.py"
    write_study(path, mode="record")
    commit_all(repo)
    study = load_study(path)
    study.run(devices=["cpu"])
    explore = run(Bigram, study.jobs()[0].preset, device="cpu", d_model=8)

    result = compare(explore, str(study.dir / "small"))
    assert any("mixes record and explore" in n for n in result.notes)
    assert result.rows[0]["label"].endswith("[explore]")


def test_record_mode_warns_with_fewer_than_three_seeds():
    with pytest.warns(UserWarning, match="fewer than 3 seeds"):
        Study("few", preset=tiny(), seeds=2, mode="record")


def test_studies_run_on_several_devices_in_parallel(tmp_path, monkeypatch, capsys):
    path = tmp_path / "study.py"
    write_study(path, seeds=2)
    study = load_study(path)
    load_data(study.preset)  # workers find the data cached instead of downloading it

    study.run(devices=["cpu", "cpu"])

    finished = sorted(p.parent.relative_to(study.dir).as_posix()
                      for p in study.dir.glob("*/seed-*/latest.json"))
    assert finished == ["small/seed-0", "small/seed-1", "wide/seed-0", "wide/seed-1"]
    assert (study.dir / "logs" / "worker-0.log").exists()
    assert "[nanoscope] 4/4 done" in capsys.readouterr().out
def test_wandb_logs_every_step(monkeypatch):
    logged, finished = [], []

    class FakeRun:
        def log(self, row, step):
            logged.append((step, row))

        def finish(self):
            finished.append(True)

    fake = types.SimpleNamespace(init=lambda **kw: FakeRun())
    monkeypatch.setitem(sys.modules, "wandb", fake)
    run(Bigram, tiny(max_steps=3), device="cpu", wandb="my-project")

    assert [step for step, _ in logged] == [1, 2, 3]
    assert "loss" in logged[0][1] and "sample" not in logged[-1][1]
    assert finished == [True]


class FakeHub:
    """An in-memory Hub: upload_folder copies into a folder, snapshot_download copies back."""

    def __init__(self, root):
        self.root, self.commits = root, []

    def api(self):
        hub = self

        class Api:
            def create_repo(self, repo_id, **kw):
                pass

            def list_repo_files(self, repo_id, **kw):
                return [p.relative_to(hub.root).as_posix()
                        for p in hub.root.rglob("*") if p.is_file()]

            def upload_folder(self, repo_id, folder_path, path_in_repo, delete_patterns,
                              commit_message):
                dest = hub.root / path_in_repo
                shutil.rmtree(dest / "checkpoints", ignore_errors=True)
                shutil.copytree(folder_path, dest, dirs_exist_ok=True)
                hub.commits.append(commit_message)

        return Api

    def snapshot_download(self, repo_id, local_dir, allow_patterns):
        prefix = allow_patterns[0].removesuffix("/*")
        shutil.copytree(self.root / prefix, Path(local_dir) / prefix, dirs_exist_ok=True)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path / "hub")
    fake.root.mkdir()
    monkeypatch.setattr("huggingface_hub.HfApi", fake.api())
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake.snapshot_download)
    return fake


def test_push_to_hub_mirrors_checkpoints_and_the_final_run(hub):
    result = run(Bigram, tiny(), device="cpu", push_to_hub="me/runs", progress=False)
    path = "/".join(result.run_dir.parts[-3:])

    assert hub.commits == [f"nanoscope: {path} step 20"]  # throttled: only the final push
    remote = hub.root / path
    assert (remote / "metrics.jsonl").read_text() == (result.run_dir / "metrics.jsonl").read_text()
    local = sorted(p.name for p in (result.run_dir / "checkpoints").iterdir())
    assert sorted(p.name for p in (remote / "checkpoints").iterdir()) == local


def test_a_new_session_resumes_from_the_hub(hub, monkeypatch):
    monkeypatch.setattr("nanoscope.integrations.HubSync.__init__.__defaults__", (0,))  # push often
    straight = run(Bigram, tiny(), device="cpu", output_dir="runs/straight", progress=False)

    def interrupt(step, row):
        if step == 7:
            os.kill(os.getpid(), signal.SIGINT)

    stopped = run(Bigram, tiny(), device="cpu", push_to_hub="me/runs", on_step=interrupt,
                  progress=False)
    shutil.rmtree(stopped.run_dir)  # a fresh Kaggle session: nothing on local disk
    resumed = run(Bigram, tiny(), device="cpu", push_to_hub="me/runs", progress=False)

    assert resumed.metrics[0]["step"] == 1 and resumed.final_step == 20
    assert [r["loss"] for r in resumed.metrics] == [r["loss"] for r in straight.metrics]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_interrupted_cuda_run_resumes_exactly():
    kw = dict(device="cuda", **SMALL)
    straight = run(GPT2, tiny(), output_dir="straight", **kw)

    def interrupt(step, row):
        if step == 7:
            os.kill(os.getpid(), signal.SIGINT)

    run(GPT2, tiny(), output_dir="resumed", on_step=interrupt, **kw)
    resumed = run(GPT2, tiny(), output_dir="resumed", **kw)
    assert [r["loss"] for r in resumed.metrics] == [r["loss"] for r in straight.metrics]


def test_several_workers_can_share_one_device(tmp_path, monkeypatch, capsys):
    path = tmp_path / "study.py"
    write_study(path, seeds=2)
    study = load_study(path)
    load_data(study.preset)

    study.run(devices=["cpu"], workers_per_device=2)

    assert len(list(study.dir.glob("*/seed-*/latest.json"))) == 4
    assert {p.name for p in (study.dir / "logs").iterdir()} == {"worker-0.log"}
    assert "2 at a time" in capsys.readouterr().out


def test_too_many_workers_for_the_free_gpu_memory_stay_queued(tmp_path):
    from nanoscope import queue
    from nanoscope.jobs.worker import Worker

    path = tmp_path / "study.py"
    write_study(path, seeds=2)
    study = load_study(path)
    ids = study.enqueue()
    gib = 2**30

    class Idle:
        def start(self, job_id, log, env): ...
        def poll(self): return None
        def terminate(self, kill=False): ...

    free = iter([5 * gib] + [1 * gib] * 10)
    worker = Worker("cuda:0", 4, runner_factory=Idle, probe=lambda job, payload: 4 * gib,
                    free_memory=lambda device: next(free))
    worker.tick()
    states = [queue.get(i)["state"] for i in ids]
    assert states.count("running") == 1 and states.count("queued") == len(ids) - 1
    assert "failed" not in states


def test_studies_sample_text_only_at_the_last_step():
    study = Study("s", preset=tiny(sample_interval=10), seeds=1)
    study.add("a", Bigram)
    assert all(j.preset.sample_interval == j.preset.max_steps for j in study.jobs())


def test_stopping_a_study_skips_the_jobs_that_have_not_started(capsys):
    from nanoscope.cli import main

    study = Study("toy", preset=tiny(), seeds=3, budget=Tokens(4 * 32 * 6))
    study.add("a", Bigram)
    study._write_plan(study.jobs())
    main(["stop", "toy"])  # a study name; nothing is running, so only the study is flagged
    assert (study.dir / "STOP").exists()
    capsys.readouterr()
    study.run(devices=["cpu"])
    assert "study stopped: skipping 3 remaining run(s)" in capsys.readouterr().out
    assert not list(study.dir.glob("a/seed-*/metrics.jsonl"))


def toml_study(mode="record"):
    study = Study("toy", preset=tiny(), seeds=3, budget=Tokens(4 * 32 * 6), baseline="small",
                  mode=mode)
    study.add("small", Bigram, d_model=8)
    study.add("wide", Bigram, d_model=32)
    return study.to_spec().to_toml()


def test_record_toml_uncommitted(repo):
    path = repo / "study.toml"
    path.write_text(toml_study())
    study = load_study(path)
    assert study.source == path.resolve() and study.mode == "record"
    with pytest.raises(ValueError, match="commit study.toml first"):
        study.run(devices=["cpu"])


def test_record_mode_runs_a_committed_toml_spec(repo):
    path = repo / "study.toml"
    path.write_text(toml_study())
    commit_all(repo)
    study = load_study(path)
    study.run(devices=["cpu"])
    manifest = json.loads((study.dir / "study.json").read_text())
    assert manifest["study_file"] == "study.toml"
    report = str(study.report())
    assert "wide" in report and "- code: commit" in report


def _interrupting_run(monkeypatch, make_hook):
    """Study.run, but the seed-0 run gets an on_step hook that fails or stops it."""
    import nanoscope.study as study_module

    real = study_module.run

    def wrapped(*args, **kwargs):
        if kwargs["output_dir"].name == "seed-0":
            kwargs["on_step"] = make_hook(kwargs["output_dir"])
        return real(*args, **kwargs)

    monkeypatch.setattr(study_module, "run", wrapped)


def test_cancelling_one_run_lets_the_rest_of_the_study_continue(monkeypatch, capsys):
    def make_hook(run_dir):
        return lambda step, row: (run_dir / "STOP").write_text("") if step == 2 else None

    _interrupting_run(monkeypatch, make_hook)
    study = Study("toy", preset=tiny(), seeds=3, budget=Tokens(4 * 32 * 6))
    study.add("a", Bigram)
    study.run(devices=["cpu"])
    out = capsys.readouterr().out
    assert "cancelled at step 2; continuing with the next run" in out
    states = {p.parent.name: json.loads(p.read_text())["state"]
              for p in study.dir.glob("a/seed-*/status.json")}
    assert states == {"seed-0": "cancelled", "seed-1": "done", "seed-2": "done"}


def test_ctrl_c_in_one_run_ends_the_whole_study(monkeypatch, capsys):
    def make_hook(run_dir):
        return lambda step, row: os.kill(os.getpid(), signal.SIGINT) if step == 2 else None

    _interrupting_run(monkeypatch, make_hook)
    study = Study("toy", preset=tiny(), seeds=3, budget=Tokens(4 * 32 * 6))
    study.add("a", Bigram)
    study.run(devices=["cpu"])
    assert "stopped at step 2; skipping the rest" in capsys.readouterr().out
    assert not list(study.dir.glob("a/seed-1/metrics.jsonl"))


def test_enqueue_puts_every_unfinished_run_on_the_batch_lane_seed_major(tmp_path):
    from nanoscope import queue

    path = tmp_path / "study.py"
    write_study(path, seeds=2)
    study = load_study(path)

    ids = study.enqueue()

    rows = [queue.get(i) for i in ids]
    assert [r["ref"] for r in rows] == [
        "studies/toy/small/seed-0", "studies/toy/wide/seed-0",
        "studies/toy/small/seed-1", "studies/toy/wide/seed-1"]
    assert {r["lane"] for r in rows} == {"batch"}
    payload = json.loads(rows[1]["payload"])
    assert payload["seed"] == 0 and payload["kwargs"] == {"d_model": 32}
    assert payload["study"]["variant"] == "wide"
    assert payload["model"].endswith("Bigram") or ":" in payload["model"]
    assert_valid("plan", json.loads((study.dir / "plan.json").read_text()))
    assert study.enqueue() == ids  # the same jobs, not new ones


def test_enqueue_skips_finished_runs(tmp_path):
    path = tmp_path / "study.py"
    write_study(path, seeds=1)
    study = load_study(path)
    study.run()  # in process: both runs finish

    assert study.enqueue() == []


def test_in_process_no_queue(tmp_path):
    path = tmp_path / "study.py"
    write_study(path, seeds=1)
    study = load_study(path)

    study.run()
    study.run(devices=["cpu"])

    assert not paths.queue_db().exists()
    assert not list(study.dir.glob("logs"))


def test_stopping_a_study_cancels_its_queued_jobs(tmp_path):
    from nanoscope import queue
    from nanoscope.store import request_stop

    path = tmp_path / "study.py"
    write_study(path, seeds=1)
    study = load_study(path)
    ids = study.enqueue()

    request_stop("studies/toy")

    assert [queue.get(i)["state"] for i in ids] == ["cancelled", "cancelled"]


def test_kaggle_command_enqueues_the_study_and_starts_a_worker_per_gpu(tmp_path, monkeypatch):
    import nanoscope.study as study_module
    from nanoscope import queue
    from nanoscope.cli import main

    path = tmp_path / "study.py"
    write_study(path, seeds=1)
    started = []

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

        def terminate(self): ...

        def wait(self): ...

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(study_module.subprocess, "Popen",
                        lambda cmd, **kw: started.append(cmd) or FakeProc())
    monkeypatch.setattr(study_module.Study, "_watch", lambda self, ids, workers, total: None)
    monkeypatch.setattr(study_module.Study, "report", lambda self, write=True: "report")

    main(["study", str(path), "--devices", "cuda:0,cuda:1", "--push-to-hub", "me/runs"])

    assert [cmd[cmd.index("--device") + 1] for cmd in started] == ["cuda:0", "cuda:1"]
    jobs = queue.list_jobs()
    assert len(jobs) == 2
    assert {json.loads(j["payload"])["push_to_hub"] for j in jobs} == {"me/runs"}


def test_m1_jobs_stable():
    """The M1 study's sizes and steps, as computed before GPT2 and Modern were rebuilt on
    blocks (pre-rebuild commit 6ba8efb): the rebuild must not move a single width."""
    study = load_study(Path(__file__).parent.parent / "studies" / "m1_ablation.py")
    jobs = study.jobs()
    assert len(jobs) == 24
    first = {j.variant.name: j for j in jobs if j.seed == 0}
    assert {name: j.variant.kwargs for name, j in first.items()} == {
        "gpt2": {},
        "modern": {"ffn_hidden": 384},
        "no-rope": {"rope": False, "ffn_hidden": 384},
        "no-swiglu": {"swiglu": False, "ffn_hidden": 584},
        "no-rmsnorm": {"rmsnorm": False, "ffn_hidden": 384},
        "no-qk-norm": {"qk_norm": False, "ffn_hidden": 384},
        "no-gqa": {"n_kv_heads": 4, "ffn_hidden": 344},
        "no-z-loss": {"z_loss": 0.0, "ffn_hidden": 384},
    }
    assert {j.preset.max_steps for j in jobs} == {1954}
