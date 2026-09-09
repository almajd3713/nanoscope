from __future__ import annotations

import subprocess
from typing import Any

import torch


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def runtime_provenance(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "git": git_provenance(),
    }
