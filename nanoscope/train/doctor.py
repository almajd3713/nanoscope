from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from typing import Any

import torch

from nanoscope.config import Config
from nanoscope.train.distributed import worker_count


def doctor(config: Config) -> dict[str, Any]:
    requested = config.train.device
    cuda_ok = torch.cuda.is_available()
    packages = {}
    for name in ("torch", "numpy", "datasets", "tiktoken", "wandb", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    issues: list[str] = []
    workers = None
    cuda_execution_ok = None
    try:
        workers = worker_count(config)
    except (ValueError, RuntimeError) as exc:
        issues.append(str(exc))
    if cuda_ok and requested != "cpu" and workers is not None:
        cuda_execution_ok = True
        for index in range(workers):
            try:
                # Device discovery alone passes on unsupported GPUs (e.g. P100/cu128).
                dtype = torch.float16 if config.train.precision == "fp16" else torch.float32
                sample = torch.ones((2, 2), device=f"cuda:{index}", dtype=dtype)
                result = (sample + 1) @ sample
                torch.cuda.synchronize(index)
                del result, sample
            except Exception as exc:
                cuda_execution_ok = False
                issues.append(
                    f"CUDA execution failed on GPU {index}: {exc}. "
                    "On Kaggle select T4 instead of P100, or install a compatible PyTorch build."
                )
    hub_repo = config.checkpoint.hub_repo or os.getenv("HF_REPO_ID")
    if config.checkpoint.hub_policy == "required" and not hub_repo:
        issues.append("required Hub persistence needs checkpoint.hub_repo or HF_REPO_ID")
    if config.checkpoint.hub_policy == "required" and not os.getenv("HF_TOKEN"):
        issues.append("required Hub persistence needs HF_TOKEN")
    if config.logging.wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
        issues.append("online W&B logging needs WANDB_API_KEY")

    resolved_revision = config.data.revision
    if config.data.source == "fineweb" and resolved_revision is None:
        try:
            from huggingface_hub import dataset_info

            resolved_revision = dataset_info(config.data.dataset_name).sha
        except Exception as exc:
            issues.append(f"could not resolve dataset revision: {exc}")

    return {
        "ok": not issues,
        "issues": issues,
        "python": platform.python_version(),
        "packages": packages,
        "config_digest": config.digest,
        "run_id": config.run.id,
        "requested_device": requested,
        "cuda_available": cuda_ok,
        "cuda_execution_ok": cuda_execution_ok,
        "distributed_strategy": config.distributed.strategy,
        "training_workers": workers,
        "global_batch_size": config.train.batch_size,
        "per_rank_batch_size": config.train.batch_size // workers if workers else None,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if cuda_ok else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda_ok
        else [],
        "git_dirty": bool(git.stdout.strip()),
        "hub_token_present": bool(os.getenv("HF_TOKEN")),
        "hub_repo": hub_repo,
        "wandb_token_present": bool(os.getenv("WANDB_API_KEY")),
        "resolved_data_revision": resolved_revision,
    }


def print_doctor(config: Config) -> int:
    report = doctor(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1
