"""Running several workers on the same hardware without oversubscribing it."""

from __future__ import annotations

import os
from collections.abc import Callable

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


def check_gpu_fits(device: str, workers: int, per_worker: int) -> None:
    free = torch.cuda.mem_get_info(torch.device(device))[0]
    if workers * per_worker > free * HEADROOM:
        raise ValueError(
            f"{workers} workers on {device} need about {workers * per_worker / 2**30:.1f} GiB "
            f"({per_worker / 2**30:.1f} GiB each) but {free / 2**30:.1f} GiB is free. "
            "Use fewer --workers-per-device."
        )


def cpu_threads(workers_on_cpu: int) -> int:
    """Threads per worker so that CPU workers share the cores instead of fighting over them."""
    return max(1, (os.cpu_count() or 1) // max(workers_on_cpu, 1))
