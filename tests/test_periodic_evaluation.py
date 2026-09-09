from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nanoscope.config import PartitionConfig, load_config
from nanoscope.eval.config import CorpusConfig, EvalConfig
from nanoscope.eval.corpus import prepare_corpus
from nanoscope.eval.runner import EvaluationCancelled, load_result, score_model
from nanoscope.model import LMOutput
from nanoscope.train.acceptance import _scientific_rows, _state_differences
from nanoscope.train.checkpoint import CheckpointManager, validate_checkpoint
from nanoscope.train.determinism import capture_rng_state
from nanoscope.train.trainer import train


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    config = load_config("configs/m0/local-smoke.yaml")
    config.run.id = "periodic"
    config.run.output_dir = str(tmp_path / "runs")
    config.data.documents = [f"Document {i} contains enough text to evaluate." for i in range(50)]
    config.data.partition = PartitionConfig(buckets=10, validation_buckets=2, test_buckets=2)
    config.data.sequence_length = 8
    config.train.max_steps = 4
    config.scheduler.warmup_steps = 0
    config.checkpoint.every_steps = 4
    config.checkpoint.keep_last = 5
    corpus = prepare_corpus(
        CorpusConfig(
            replace(config.data, partition=replace(config.data.partition, split="validation")),
            config.tokenizer,
            5,
            tmp_path / "corpus",
        )
    )
    return config, EvalConfig(corpus.path, tmp_path / "standalone", every_steps=2)


def load_state(result):
    return CheckpointManager(result.run_dir, 5).load(result.checkpoint, torch.device("cpu"))[0]


def assert_training_equal(left, right):
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "stream",
        "rank_rng",
        "step",
        "tokens_seen",
    ):
        assert not _state_differences(left[key], right[key]), key
    assert _scientific_rows(left["metrics"]) == _scientific_rows(right["metrics"])


def test_periodic_preserves_training_and_recovers_history(setup):
    config, evaluation = setup
    baseline = copy.deepcopy(config)
    baseline.run.id = "baseline"
    control = load_state(train(baseline, resume="none"))
    first = train(config, resume="none", stop_after_step=2, eval_config=evaluation)
    # Stopping saves weights without spending time validating; resume fills this gap.
    assert not load_state(first)["evaluation"]["results"]
    result = train(config, eval_config=evaluation)
    state = load_state(result)
    assert_training_equal(control, state)
    assert [r["checkpoint"]["step"] for r in state["evaluation"]["results"]] == [2, 4]
    files = list((result.run_dir / "evaluations").glob("*.json"))
    assert len(files) == 2
    journal = result.run_dir / "evaluations.jsonl"
    assert len(journal.read_text().splitlines()) == 2
    before = journal.read_bytes()
    train(config, eval_config=evaluation)
    assert journal.read_bytes() == before
    # Recover outputs solely from the checkpoint, even with evaluation disabled.
    shutil.rmtree(result.run_dir / "evaluations")
    journal.unlink()
    train(config)
    assert journal.read_bytes() == before
    assert all(load_result(p)["complete"] for p in files)


def test_periodic_uninterrupted_and_rollback(setup):
    config, evaluation = setup
    result = train(config, resume="none", eval_config=evaluation)
    original = load_state(result)
    checkpoint2 = result.run_dir / "checkpoints" / "step_00000002"
    train(config, resume=str(checkpoint2), eval_config=evaluation)
    assert_training_equal(original, load_state(result))
    assert len(list((result.run_dir / "evaluations").glob("*.json"))) == 2
    assert list((result.run_dir / "evaluations" / "orphaned").glob("*.json"))
    assert list((result.run_dir / "checkpoints").glob(".superseded-*"))


def test_failed_evaluation_keeps_checkpoint_and_can_retry(setup, monkeypatch):
    import nanoscope.eval.runner as runner

    config, evaluation = setup
    original = runner.score_model
    monkeypatch.setattr(
        runner,
        "score_model",
        lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("evaluation exceeded max_seconds")),
    )
    with pytest.raises(TimeoutError, match="max_seconds"):
        train(config, resume="none", eval_config=evaluation)
    run_dir = config.run.output_dir + "/" + config.run.id
    manager = CheckpointManager(Path(run_dir), 5)
    checkpoint = manager.latest()
    assert checkpoint is not None and validate_checkpoint(checkpoint)
    assert manager.load(checkpoint, torch.device("cpu"))[0]["step"] == 2
    monkeypatch.setattr(runner, "score_model", original)
    result = train(config, eval_config=evaluation)
    assert [r["checkpoint"]["step"] for r in load_state(result)["evaluation"]["results"]] == [2, 4]


def test_periodic_preflight_rejects_unverified_training(setup):
    config, evaluation = setup
    config.data.partition = None
    with pytest.raises(ValueError, match="exclusion provenance"):
        train(config, resume="none", eval_config=evaluation)


def test_sidecar_publish_failure_preserves_prior_manifest(setup, monkeypatch):
    import nanoscope.train.checkpoint as checkpoints

    config, evaluation = setup
    result = train(config, resume="none", eval_config=evaluation)
    manager = CheckpointManager(result.run_dir, 5)
    before = load_state(result)["evaluation"]
    original = checkpoints.write_json

    def fail_manifest(path, value):
        if path.name == "manifest.json":
            raise OSError("simulated interruption")
        return original(path, value)

    monkeypatch.setattr(checkpoints, "write_json", fail_manifest)
    with pytest.raises(OSError):
        manager.attach_evaluation(result.checkpoint, {"results": []})
    assert validate_checkpoint(result.checkpoint)
    assert load_state(result)["evaluation"] == before


@pytest.mark.parametrize("mode", ["success", "error", "cancel"])
def test_scoring_restores_mutated_buffers_and_rng(mode):
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.tensor(0))

        def forward(self, tokens):
            self.counter.add_(1)
            self.register_buffer("temporary", torch.rand(1))
            self.counter = torch.tensor(99)
            if mode == "error":
                raise ValueError("failed forward")
            return LMOutput(torch.randn(*tokens.shape, 3))

    model = Stateful().train()
    original = model.counter
    rng = capture_rng_state(torch.device("cpu"))
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return mode == "cancel" and calls > 1

    if mode == "success":
        score_model(model, np.zeros((2, 3), dtype=np.int64), batch_size=1, cancelled=cancelled)
    else:
        with pytest.raises((ValueError, EvaluationCancelled)):
            score_model(model, np.zeros((2, 3), dtype=np.int64), batch_size=1, cancelled=cancelled)
    assert model.counter is original and model.counter.item() == 0
    assert not hasattr(model, "temporary") and model.training
    assert not _state_differences(rng, capture_rng_state(torch.device("cpu")))


def test_three_seed_report_from_periodic_results(setup, tmp_path):
    from nanoscope.eval.compare import build_comparison
    from nanoscope.eval.report import write_report

    config, evaluation = setup
    variants = []
    for name, dropout in (("base", 0.1), ("increment", 0.2)):
        paths = []
        for seed in (11, 22, 33):
            current = copy.deepcopy(config)
            current.run.id = f"{name}-{seed}"
            current.run.seed = seed
            current.train.max_steps = 2
            current.model.params["dropout"] = dropout
            result = train(current, resume="none", eval_config=evaluation)
            paths.append(str(result.run_dir / "evaluations"))
        variants.append({"name": name, "results": paths})
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "name": "Three seed study",
                "baseline": "base",
                "axis": "tokens",
                "budget": 32,
                "variants": variants,
            }
        )
    )
    comparison = build_comparison(study)
    summary = comparison["paired_summaries"][0]
    assert summary["n"] == 3 and summary["ci95_low"] is not None
    report = write_report(comparison, tmp_path / "report", plots=True)
    assert "Paired seed differences" in report.read_text()
    assert (report.parent / "paired-summary.csv").exists()
    assert (report.parent / "loss-vs-tokens_seen.svg").exists()


def test_ddp_periodic_preserves_training_and_resume(setup, tmp_path):
    from test_distributed import run_cli

    from nanoscope.config import DistributedConfig

    config, evaluation = setup
    config.distributed = DistributedConfig(strategy="ddp", devices=2)
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        yaml.safe_dump(
            {"corpus": str(evaluation.corpus), "output": str(evaluation.output), "every_steps": 2}
        )
    )
    control_config = copy.deepcopy(config)
    control_config.run.id = "ddp-control"
    run_cli(control_config, tmp_path, "--resume", "none")
    run_cli(
        config,
        tmp_path,
        "--resume",
        "none",
        "--stop-after-step",
        "2",
        "--eval-config",
        str(evaluation_path),
    )
    run_cli(config, tmp_path, "--eval-config", str(evaluation_path))
    states = []
    for current in (control_config, config):
        manager = CheckpointManager(Path(current.run.output_dir) / current.run.id, 5)
        checkpoint = manager.latest()
        assert checkpoint is not None
        states.append(manager.load(checkpoint, torch.device("cpu"))[0])
    assert_training_equal(*states)
    assert [row["checkpoint"]["step"] for row in states[1]["evaluation"]["results"]] == [2, 4]


def test_sigterm_during_evaluation_stops_with_resumable_checkpoint(setup, monkeypatch):
    import os
    import signal

    import nanoscope.eval.runner as runner

    config, evaluation = setup
    original = runner.score_model

    def stop_during_score(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "score_model", stop_during_score)
    interrupted = train(config, resume="none", eval_config=evaluation)
    assert interrupted.stopped_early and interrupted.final_step == 2
    assert not load_state(interrupted)["evaluation"]["results"]
    assert not list((interrupted.run_dir / "evaluations").glob("*.json"))
    retried = train(config, eval_config=evaluation)
    assert retried.final_step == 2 and retried.stopped_early
    monkeypatch.setattr(runner, "score_model", original)
    resumed = train(config, eval_config=evaluation)
    assert [r["checkpoint"]["step"] for r in load_state(resumed)["evaluation"]["results"]] == [2, 4]


def test_training_clock_excludes_evaluation_work(setup, monkeypatch):
    import time

    from nanoscope.eval.periodic import PeriodicEvaluation

    config, evaluation = setup
    clock = 0.0

    def tick():
        nonlocal clock
        clock += 1
        return clock

    original = PeriodicEvaluation.evaluate

    def costly_evaluation(*args, **kwargs):
        nonlocal clock
        clock += 100_000
        return original(*args, **kwargs)

    monkeypatch.setattr(time, "perf_counter", tick)
    monkeypatch.setattr(PeriodicEvaluation, "evaluate", costly_evaluation)
    result = train(config, resume="none", eval_config=evaluation)
    assert all(row["train_step_seconds"] < 100_000 for row in result.metrics)
    assert load_state(result)["train_seconds"] < 100_000
