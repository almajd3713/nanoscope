from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from typing import Any

import torch

from nanoscope.config import Config


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
    if requested == "cuda" and not cuda_ok:
        issues.append("CUDA was requested but is unavailable")
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
