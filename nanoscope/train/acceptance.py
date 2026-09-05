from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanoscope.config import Config, dump_config
from nanoscope.train.metrics import read_metrics


def _scientific_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ignored = {
        "tokens_per_second",
        "mfu",
        "active_seconds",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    }
    return [{key: value for key, value in row.items() if key not in ignored} for row in rows]


def _state_differences(left: Any, right: Any, path: str = "state") -> list[str]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return [] if torch.equal(left.cpu(), right.cpu()) else [path]
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return [] if np.array_equal(left, right) else [path]
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return [f"{path}.keys"]
        return [
            difference
            for key in left
            for difference in _state_differences(left[key], right[key], f"{path}.{key}")
        ]
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}.length"]
        return [
            difference
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True))
            for difference in _state_differences(
                left_item, right_item, f"{path}[{index}]"
            )
        ]
    return [] if left == right else [path]


def run_acceptance(config: Config, work_dir: str | Path | None = None) -> dict[str, Any]:
    """Compare a control with a child process terminated three times."""
    if config.data.source != "fixture":
        raise ValueError(
            "local acceptance requires fixture data; use Kaggle for FineWeb acceptance"
        )
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="nanoscope-m0-"))
    root.mkdir(parents=True, exist_ok=True)
    baseline = copy.deepcopy(config)
    baseline.run.id = "m0-control"
    baseline.run.output_dir = str(root)
    baseline.checkpoint.hub_policy = "disabled"
    baseline.checkpoint.hub_repo = None
    baseline.logging.wandb_mode = "disabled"
    baseline_path = root / "control.yaml"
    dump_config(baseline, baseline_path)

    interrupted = copy.deepcopy(config)
    interrupted.run.id = "m0-interrupted"
    interrupted.run.output_dir = str(root)
    interrupted.checkpoint.hub_policy = "disabled"
    interrupted.checkpoint.hub_repo = None
    interrupted.logging.wandb_mode = "disabled"
    if interrupted.train.max_steps < 8:
        raise ValueError("M0 acceptance requires at least 8 training steps")
    cuts = [
        max(1, interrupted.train.max_steps // 4 - 1),
        max(2, interrupted.train.max_steps // 2 - 1),
        max(3, interrupted.train.max_steps * 3 // 4 - 1),
    ]
    interrupted_path = root / "interrupted.yaml"
    dump_config(interrupted, interrupted_path)

    def command(path: Path, resume: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "nanoscope",
            "train",
            "--config",
            str(path),
            "--resume",
            resume,
        ]

    baseline_process = subprocess.run(
        command(baseline_path, "none"), capture_output=True, text=True, check=False
    )
    if baseline_process.returncode:
        raise RuntimeError(f"control run failed: {baseline_process.stderr}")

    env = {**os.environ, "NANOSCOPE_ACCEPTANCE_STEP_DELAY": "0.05"}
    interrupted_metrics = root / interrupted.run.id / "metrics.jsonl"
    actual_cuts: list[int] = []
    for cut in cuts:
        process = subprocess.Popen(
            command(interrupted_path, "auto"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        deadline = time.monotonic() + 30
        observed = 0
        while time.monotonic() < deadline and process.poll() is None:
            rows = read_metrics(interrupted_metrics)
            observed = int(rows[-1]["step"]) if rows else 0
            if observed >= cut:
                process.send_signal(signal.SIGTERM)
                break
            time.sleep(0.01)
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode:
            raise RuntimeError(f"interrupted run failed: {stderr or stdout}")
        actual_cuts.append(observed)

    final_process = subprocess.run(
        command(interrupted_path, "auto"), capture_output=True, text=True, check=False, env=env
    )
    if final_process.returncode:
        raise RuntimeError(f"final resumed run failed: {final_process.stderr}")

    control_dir = root / baseline.run.id
    interrupted_dir = root / interrupted.run.id
    expected = _scientific_rows(read_metrics(control_dir / "metrics.jsonl"))
    actual = _scientific_rows(read_metrics(interrupted_dir / "metrics.jsonl"))
    failures: list[str] = []
    if len(expected) != len(actual):
        failures.append(f"metric lengths differ: {len(expected)} != {len(actual)}")
    for left, right in zip(expected, actual, strict=False):
        if left["batch_hash"] != right["batch_hash"]:
            failures.append(f"batch mismatch at step {left['step']}")
        relative = abs(left["loss"] - right["loss"]) / max(abs(left["loss"]), 1e-12)
        if relative >= 0.001:
            failures.append(f"loss mismatch at step {left['step']}: {relative:.6%}")
        for key in set(left) - {"loss"}:
            if left[key] != right[key]:
                failures.append(f"{key} mismatch at step {left['step']}")

    control_state = torch.load(
        control_dir / "checkpoints" / f"step_{config.train.max_steps:08d}" / "state.pt",
        map_location="cpu",
        weights_only=False,
    )
    interrupted_state = torch.load(
        interrupted_dir
        / "checkpoints"
        / f"step_{config.train.max_steps:08d}"
        / "state.pt",
        map_location="cpu",
        weights_only=False,
    )
    compared_state = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "stream",
        "rng",
        "step",
        "tokens_seen",
    }
    state_mismatches = [
        mismatch
        for key in sorted(compared_state)
        for mismatch in _state_differences(control_state[key], interrupted_state[key], key)
    ]
    failures.extend(f"final state mismatch: {path}" for path in state_mismatches)

    report = {
        "passed": not failures,
        "interruptions": actual_cuts,
        "steps": config.train.max_steps,
        "failures": failures,
        "state_mismatches": state_mismatches,
        "control": str(control_dir),
        "interrupted": str(interrupted_dir),
    }
    report_path = root / "acceptance-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report
