"""Running several workers on the same hardware without oversubscribing it."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from nanoscope.presets import Preset
from nanoscope.train_loop import _split_output

HEADROOM = 0.9  # use at most this fraction of the free GPU memory


def probe_memory(build_model: Callable[[], torch.nn.Module], preset: Preset,
                 device: torch.device) -> int:
    """Bytes one training process needs on this GPU: the CUDA context plus a real step."""
    free_before = torch.cuda.mem_get_info(device)[0]
    torch.zeros(1, device=device)  # creates the CUDA context
    context = free_before - torch.cuda.mem_get_info(device)[0]
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, 256, (preset.batch_size, preset.context_length), device=device)
    logits, aux = _split_output(model(tokens))
    (F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1)) + aux).backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    del model, optimizer, logits
    torch.cuda.empty_cache()
    return context + peak


def cgroup_cpu_limit(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """The CPUs a container may use (`docker run --cpus 4` is 4.0), or None when unlimited."""
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = (root / "cpu.max").read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError, ZeroDivisionError):
        return None


def available_cpus() -> int:
    """The cores this process may really use: the CPU affinity, capped by a container's quota.
    `os.cpu_count()` is the host's core count, so a 4-CPU container on a 16-core machine would
    start 16 threads and be throttled."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        cpus = os.cpu_count() or 1
    limit = cgroup_cpu_limit()
    if limit is not None:
        cpus = min(cpus, max(1, math.floor(limit)))
    return max(1, cpus)


def cpu_threads(workers_on_cpu: int) -> int:
    """Threads per worker so that CPU workers share the cores instead of fighting over them."""
    return max(1, available_cpus() // max(workers_on_cpu, 1))


def free_memory(device: str) -> int:
    """Free bytes on a CUDA device."""
    return torch.cuda.mem_get_info(torch.device(device))[0]


def configure_device(device: str | None, slots: int) -> None:
    """Called in a job's own process: its share of the CPU threads or of the GPU memory."""
    if device is None or device == "cpu":
        explicit = os.environ.get("NANOSCOPE_THREADS")  # `nanoscope study --threads N`
        torch.set_num_threads(int(explicit) if explicit else cpu_threads(slots))
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(HEADROOM / max(slots, 1),
                                                   torch.device(device))
