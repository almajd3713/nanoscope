from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    # This must be set before the first cuBLAS operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def capture_rng_state(device: torch.device | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device is not None and device.type == "cuda":
        state["torch_cuda_local"] = torch.cuda.get_rng_state(device)
    elif device is None and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_local" in state and device is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda_local"], device)
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
