"""Single-node DDP orchestration; model implementations remain ordinary modules."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist

from nanoscope.config import Config, validate_config

T = TypeVar("T")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def worker_count(config: Config) -> int:
    validate_config(config)
    device = resolve_device(config.train.device)
    available = torch.cuda.device_count() if device.type == "cuda" else 1
    count = 1
    if config.distributed.strategy == "ddp":
        count = (
            available if config.distributed.devices == "auto" else int(config.distributed.devices)
        )
    if device.type == "cuda" and count > available:
        raise ValueError(f"requested {count} GPUs but only {available} are visible")
    if config.train.batch_size % count:
        raise ValueError("train.batch_size is global and must be divisible by the worker count")
    return count


@dataclass
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    owns_group: bool = False

    @property
    def primary(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @classmethod
    def create(cls, config: Config) -> DistributedContext:
        expected = worker_count(config)
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        if expected != world_size:
            raise RuntimeError(
                f"configuration needs {expected} workers, launcher supplied {world_size}; "
                "use nanoscope train --config ... to launch automatically"
            )
        device = resolve_device(config.train.device)
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if int(os.getenv("LOCAL_WORLD_SIZE", str(world_size))) != world_size or rank != local_rank:
            raise RuntimeError("only single-node distributed training is supported")
        if device.type == "cuda":
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        context = cls(device, rank, world_size)
        if context.enabled:
            if dist.is_initialized():
                raise RuntimeError("training must own its process group")
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=10),
            )
            context.owns_group = True
        return context

    def close(self) -> None:
        if self.owns_group:
            dist.destroy_process_group()
            self.owns_group = False

    def primary_call(self, action: Callable[[], T]) -> T:
        """Run shared I/O once, propagating errors before the next collective."""
        if not self.enabled:
            return action()
        packet: list[Any] = [None, None]
        if self.primary:
            try:
                packet[0] = action()
            except Exception as exc:
                packet[1] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(packet, src=0)
        if packet[1] is not None:
            raise RuntimeError(f"rank 0 operation failed: {packet[1]}")
        return cast(T, packet[0])

    def reduce(self, value: float, operation: str = "sum") -> float:
        if not self.enabled:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX if operation == "max" else dist.ReduceOp.SUM)
        return float(tensor.item())

    def gather(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def scatter_batch(self, batch: torch.Tensor | None, shape: tuple[int, int]) -> torch.Tensor:
        if not self.enabled:
            assert batch is not None
            return batch.to(self.device)
        chunks = None
        if self.primary:
            assert batch is not None
            chunks = [part.contiguous().to(self.device) for part in batch.chunk(self.world_size)]
        local = torch.empty(shape, dtype=torch.long, device=self.device)
        dist.scatter(local, scatter_list=chunks, src=0)
        return local
