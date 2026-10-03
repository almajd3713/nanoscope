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

import nanoscope.dataset as dataset
from nanoscope import FLOPs, Study, Tokens, compare, run
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
    (tmp_path / ".gitignore").write_text("runs/\ncache/\nexperiments/\n")
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
    out = tmp_path / "experiments" / "toy"
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
    monkeypatch.setenv("NANOSCOPE_DATA_DIR", str(dataset.CACHE_DIR))

    study.run(devices=["cpu", "cpu"])

    finished = sorted(p.parent.relative_to(study.dir).as_posix()
                      for p in study.dir.glob("*/seed-*/latest.json"))
    assert finished == ["small/seed-0", "small/seed-1", "wide/seed-0", "wide/seed-1"]
    assert (study.dir / "logs" / "worker-1.log").exists()
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


def test_several_workers_can_share_one_device(tmp_path, monkeypatch):
    path = tmp_path / "study.py"
    write_study(path, seeds=2)
    study = load_study(path)
    load_data(study.preset)
    monkeypatch.setenv("NANOSCOPE_DATA_DIR", str(dataset.CACHE_DIR))

    study.run(devices=["cpu"], workers_per_device=2)

    assert len(list(study.dir.glob("*/seed-*/latest.json"))) == 4
    assert {p.name for p in (study.dir / "logs").iterdir()} == {"worker-0.log", "worker-1.log"}


def test_too_many_workers_for_the_free_gpu_memory_are_refused(monkeypatch):
    from nanoscope.hardware import check_gpu_fits

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * gib, 8 * gib))
    check_gpu_fits("cuda:0", 3, gib)  # 3 GiB of 4 free fits
    with pytest.raises(ValueError, match="fewer --workers-per-device"):
        check_gpu_fits("cuda:0", 4, gib)


def test_studies_sample_text_only_at_the_last_step():
    study = Study("s", preset=tiny(sample_interval=10), seeds=1)
    study.add("a", Bigram)
    assert all(j.preset.sample_interval == j.preset.max_steps for j in study.jobs())
