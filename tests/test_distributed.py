from __future__ import annotations

import copy
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from nanoscope.config import ConfigError, DistributedConfig, dump_config, load_config
from nanoscope.model import build_model
from nanoscope.train.acceptance import _scientific_rows, _state_differences, run_acceptance
from nanoscope.train.distributed import worker_count
from nanoscope.train.doctor import doctor
from nanoscope.train.metrics import read_metrics
from nanoscope.train.trainer import train

ROOT = Path(__file__).resolve().parents[1]


def config_for(tmp_path: Path):
    config = load_config(ROOT / "configs/m0/local-smoke.yaml")
    config.run.output_dir = str(tmp_path)
    config.train.max_steps = 4
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2
    config.checkpoint.every_steps = 2
    config.distributed = DistributedConfig(strategy="ddp", devices=2)
    return config


def run_cli(config, root: Path, *extra: str, audit: bool = False, fail: bool = False):
    config_path = root / "config.yaml"
    dump_config(config, config_path)
    arguments = ["train", "--config", str(config_path), *extra]
    command = [sys.executable, "-m", "nanoscope", *arguments]
    if audit:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed_worker.py"),
            *arguments,
        ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(ROOT),
            "NANOSCOPE_TEST_AUDIT": str(root),
            **({"NANOSCOPE_TEST_FAIL_UPLOAD": "1"} if fail else {}),
        },
    )
    if not fail:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def state_for(config):
    return torch.load(
        Path(config.run.output_dir)
        / config.run.id
        / "checkpoints"
        / f"step_{config.train.max_steps:08d}"
        / "state.pt",
        map_location="cpu",
        weights_only=False,
    )


def test_ddp_resume_matches_uninterrupted_with_dropout(tmp_path):
    control = config_for(tmp_path)
    control.run.id = "control"
    run_cli(control, tmp_path, "--resume", "none")
    resumed = copy.deepcopy(control)
    resumed.run.id = "resumed"
    run_cli(resumed, tmp_path, "--resume", "none", "--stop-after-step", "2")
    run_cli(resumed, tmp_path, "--resume", "auto")
    left, right = state_for(control), state_for(resumed)
    for key in ("model", "optimizer", "scheduler", "scaler", "stream", "rank_rng", "tokens_seen"):
        assert not _state_differences(left[key], right[key], key)
    assert _scientific_rows(left["metrics"]) == _scientific_rows(right["metrics"])
    assert left["tokens_seen"] == 4 * 4 * 2 * control.data.sequence_length
    assert len(left["rank_rng"]) == 2
    assert not torch.equal(left["rank_rng"][0]["torch_cpu"], left["rank_rng"][1]["torch_cpu"])
    model, _ = build_model(control.model.name, control.model.params)
    model.load_state_dict(left["model"])  # No module. prefix; ordinary CPU inference works.
    assert model(torch.zeros((1, 3), dtype=torch.long)).logits.shape[1] == 3


def test_ddp_matches_single_process_global_batch(tmp_path):
    config = config_for(tmp_path)
    config.model.params["dropout"] = 0.0
    config.run.id = "distributed"
    run_cli(config, tmp_path, "--resume", "none")
    distributed = state_for(config)
    single = copy.deepcopy(config)
    single.run.id = "single"
    single.distributed = DistributedConfig()
    run_cli(single, tmp_path, "--resume", "none")
    reference = state_for(single)
    assert distributed["stream"] == reference["stream"]
    assert distributed["tokens_seen"] == reference["tokens_seen"]
    assert [r["batch_hash"] for r in distributed["metrics"]] == [
        r["batch_hash"] for r in reference["metrics"]
    ]
    for name in reference["model"]:
        torch.testing.assert_close(distributed["model"][name], reference["model"][name])
    for left, right in zip(distributed["metrics"], reference["metrics"], strict=True):
        assert left["loss"] == pytest.approx(right["loss"], rel=1e-6)


def test_cloud_and_stream_operations_only_on_primary(tmp_path):
    config = config_for(tmp_path)
    run_cli(config, tmp_path, "--resume", "none", audit=True)
    events = (tmp_path / "rank-0.txt").read_text().splitlines()
    assert events.count("build_stream") == 1
    assert events.count("wandb_init") == 1
    assert events.count("wandb_finish") == 1
    assert events.count("wandb_log") == 4
    assert events.count("hub_upload") == 2
    assert not (tmp_path / "rank-1.txt").exists()
    assert len(read_metrics(tmp_path / config.run.id / "metrics.jsonl")) == 4
    expected = state_for(config)
    config.run.id = "cloud-resumed"
    run_cli(config, tmp_path, "--resume", "none", "--stop-after-step", "2", audit=True)
    run_cli(config, tmp_path, "--resume", "auto", audit=True)
    actual = state_for(config)
    for key in ("model", "optimizer", "rank_rng", "stream"):
        assert not _state_differences(expected[key], actual[key], key)


def test_primary_failure_terminates_all_workers(tmp_path):
    config = config_for(tmp_path)
    result = run_cli(config, tmp_path, "--resume", "none", audit=True, fail=True)
    assert result.returncode != 0
    assert "rank 0 operation failed: RuntimeError: simulated Hub outage" in result.stderr


def test_conditional_model_and_auxiliary_loss_contract(tmp_path):
    config = config_for(tmp_path)
    config.model.name = "conditional_test_model"
    config.distributed.find_unused_parameters = True
    run_cli(config, tmp_path, "--resume", "none", audit=True)
    state = state_for(config)
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in state["metrics"])
    assert len(state["metrics"]) == config.train.max_steps


def test_kaggle_launcher_forwards_sigterm(tmp_path):
    config = config_for(tmp_path)
    config.train.max_steps = 40
    path = tmp_path / "config.yaml"
    dump_config(config, path)
    # Exercise the real launcher forwarding without cloud credentials or pip installation.
    command = [
        sys.executable,
        "-c",
        "import sys; from kaggle_run import _run; "
        "_run(sys.executable, '-m', 'nanoscope', 'train', '--config', sys.argv[1], "
        "'--resume', 'none')",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NANOSCOPE_ACCEPTANCE_STEP_DELAY": "0.2",
        },
    )
    metrics = tmp_path / config.run.id / "metrics.jsonl"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and process.poll() is None:
        if metrics.exists() and metrics.stat().st_size:
            process.send_signal(signal.SIGTERM)
            break
        time.sleep(0.02)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, stdout + stderr
    rows = read_metrics(metrics)
    assert 0 < len(rows) < config.train.max_steps
    checkpoint = tmp_path / config.run.id / "checkpoints" / f"step_{rows[-1]['step']:08d}"
    assert (checkpoint / "manifest.json").exists()


def test_changed_world_size_is_rejected(tmp_path):
    config = config_for(tmp_path)
    run_cli(config, tmp_path, "--resume", "none", "--stop-after-step", "2")
    config.distributed.devices = 1
    with pytest.raises(RuntimeError, match="exact resume requires 2 workers"):
        train(config, resume="auto")


def test_worker_count_validates_topology(monkeypatch):
    config = load_config(ROOT / "configs/m0/local-smoke.yaml")
    config.train.device = "cuda"
    config.distributed = DistributedConfig(strategy="ddp")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert worker_count(config) == 2
    config.distributed.devices = 3
    with pytest.raises(ValueError, match="only 2"):
        worker_count(config)
    config.distributed.devices = "auto"
    config.train.batch_size = 3
    with pytest.raises(ValueError, match="divisible"):
        worker_count(config)
    config.distributed.devices = True
    with pytest.raises(ConfigError, match="positive integer"):
        worker_count(config)


def test_doctor_catches_visible_but_incompatible_cuda(monkeypatch):
    config = load_config(ROOT / "configs/m0/local-smoke.yaml")
    config.train.device = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Tesla P100")

    def fail(*args, **kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(torch, "ones", fail)
    report = doctor(config)
    assert report["cuda_available"]
    assert not report["cuda_execution_ok"]
    assert not report["ok"]
    assert "no kernel image" in report["issues"][0]


def test_ddp_sigterm_acceptance(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    config = config_for(tmp_path)
    config.train.max_steps = 12
    report = run_acceptance(config, tmp_path)
    assert report["passed"], report["failures"]


@pytest.mark.gpu
@pytest.mark.parametrize("workers", [1, 2])
def test_gpu_fp16_resume(tmp_path, workers):
    if torch.cuda.device_count() < workers:
        pytest.skip(f"requires {workers} CUDA GPUs")
    config = config_for(tmp_path)
    config.distributed.devices = workers
    config.train.device = "cuda"
    config.train.precision = "fp16"
    config.run.id = "gpu-control"
    run_cli(config, tmp_path, "--resume", "none")
    expected = state_for(config)
    config.run.id = "gpu-resumed"
    run_cli(config, tmp_path, "--resume", "none", "--stop-after-step", "2")
    run_cli(config, tmp_path, "--resume", "auto")
    actual = state_for(config)
    for key in ("model", "optimizer", "scaler", "stream", "rank_rng"):
        assert not _state_differences(expected[key], actual[key], key)
