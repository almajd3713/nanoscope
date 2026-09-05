from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from nanoscope.config import Config, dump_config
from nanoscope.data import build_batch_stream
from nanoscope.model import LMOutput, build_model
from nanoscope.model.registry import (
    default_parameter_groups,
    estimate_training_flops,
)
from nanoscope.train.checkpoint import CheckpointManager
from nanoscope.train.determinism import (
    capture_rng_state,
    configure_reproducibility,
    restore_rng_state,
)
from nanoscope.train.distributed import DistributedContext
from nanoscope.train.hub import HubCheckpointStore, download_hub_checkpoint
from nanoscope.train.metrics import JsonlLogger, WandbLogger


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    final_step: int
    checkpoint: Path
    metrics: list[dict[str, Any]]
    stopped_early: bool


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def _runtime_provenance(device: torch.device) -> dict[str, Any]:
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
        "git": _git_provenance(),
    }


def _peak_tflops(config: Config, device: torch.device) -> float | None:
    if config.performance.peak_tflops is not None:
        return config.performance.peak_tflops
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device).lower()
    if "t4" in name:
        return 65.0
    if "p100" in name:
        return 21.2
    return None


def _make_scheduler(config: Config, optimizer: torch.optim.Optimizer):
    warmup = config.scheduler.warmup_steps
    total = config.train.max_steps

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        if config.scheduler.name == "constant":
            return 1.0
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # PyTorch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _batch_hash(batch: torch.Tensor) -> str:
    return hashlib.sha256(batch.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def _resume_path(resume: str, manager: CheckpointManager, run_dir: Path) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        return manager.latest()
    if resume.startswith("hf://"):
        return download_hub_checkpoint(resume, run_dir / "checkpoints")
    return Path(resume)


def train(config: Config, resume: str = "auto", stop_after_step: int | None = None) -> TrainResult:
    configure_reproducibility(seed=config.run.seed, deterministic=config.train.deterministic)
    context = DistributedContext.create(config)
    try:
        with contextlib.ExitStack() as resources:
            return _train(config, resume, stop_after_step, context, resources)
    finally:
        context.close()


def _train(
    config: Config,
    resume: str,
    stop_after_step: int | None,
    context: DistributedContext,
    resources: contextlib.ExitStack,
) -> TrainResult:
    device = context.device
    run_dir = Path(config.run.output_dir) / config.run.id
    provenance = _runtime_provenance(device)
    provenance["world_size"] = context.world_size
    if config.run.require_clean_repo and provenance["git"]["dirty"]:
        raise RuntimeError("run.require_clean_repo is true but the Git worktree is dirty")

    def prepare_run() -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        if resume == "none" and (
            (run_dir / "metrics.jsonl").exists() or (run_dir / "latest.json").exists()
        ):
            raise RuntimeError("run already exists; choose a new run.id or resume it")
        dump_config(config, run_dir / "resolved-config.yaml")
        (run_dir / "environment.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )

    context.primary_call(prepare_run)
    stream = None

    def prepare_stream() -> str | None:
        nonlocal stream
        stream = build_batch_stream(
            config.data, config.tokenizer, seed=config.run.seed, batch_size=config.train.batch_size
        )
        resources.callback(stream.close)
        return stream.source_revision

    source_revision = context.primary_call(prepare_stream)
    model, model_spec = build_model(config.model.name, config.model.params)
    model.to(device)
    training_model = (
        DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=config.distributed.find_unused_parameters,
        )
        if context.enabled
        else model
    )
    if context.enabled:
        # Initialize replicas identically, then give dropout independent rank streams.
        configure_reproducibility(
            seed=config.run.seed + context.rank, deterministic=config.train.deterministic
        )
    groups = (
        model_spec.parameter_groups(model, config.optimizer.weight_decay)
        if model_spec.parameter_groups
        else default_parameter_groups(model, config.optimizer.weight_decay)
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )
    scheduler = _make_scheduler(config, optimizer)
    scaler = _grad_scaler(device.type == "cuda" and config.train.precision == "fp16")
    manager = CheckpointManager(run_dir, config.checkpoint.keep_last)
    hub = HubCheckpointStore(
        config.checkpoint.hub_repo, config.run.id, config.checkpoint.hub_policy
    )

    step = 0
    tokens_seen = 0
    active_seconds = 0.0
    metrics_history: list[dict[str, Any]] = []
    selected_resume = context.primary_call(lambda: _resume_path(resume, manager, run_dir))
    if selected_resume is not None:
        # RNG byte tensors must stay on CPU; optimizer.load_state_dict moves its tensors.
        state, metadata = manager.load(selected_resume, torch.device("cpu"))
        saved_world_size = int(metadata.get("world_size", 1))
        if saved_world_size != context.world_size:
            raise RuntimeError(
                f"exact resume requires {saved_world_size} workers; got {context.world_size}. "
                "Model weights remain portable, but training RNG topology cannot be changed."
            )
        if metadata["config_digest"] != config.digest:
            raise RuntimeError("checkpoint configuration is incompatible with this run")
        if metadata.get("source_revision") != source_revision:
            raise RuntimeError(
                "checkpoint dataset revision differs from the configured/resolved revision"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])

        def restore_stream() -> None:
            assert stream is not None
            stream.load_state_dict(state["stream"])

        context.primary_call(restore_stream)
        step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        active_seconds = float(state["active_seconds"])
        metrics_history = list(state.get("metrics", []))
        rank_rng = state.get("rank_rng", [state["rng"]])[context.rank]
        restore_rng_state(rank_rng, device)
        del state

    metrics_path = run_dir / "metrics.jsonl"
    local_logger = None
    cloud_logger = None

    def prepare_logging() -> None:
        nonlocal local_logger, cloud_logger
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics_history),
            encoding="utf-8",
        )
        local_logger = JsonlLogger(metrics_path, resume_step=step)
        cloud_logger = WandbLogger(
            cast(Literal["disabled", "offline", "online"], config.logging.wandb_mode),
            config.run.id,
            config.logging.project,
            config.logging.entity or os.getenv("WANDB_ENTITY"),
            run_dir,
            config.to_dict(),
            resumed=selected_resume is not None,
        )
        resources.callback(cloud_logger.finish)

    # Logging SDK initialization must not perturb model/dropout RNG, particularly
    # after restoring a checkpoint in a new process.
    training_rng = capture_rng_state(device)
    context.primary_call(prepare_logging)
    restore_rng_state(training_rng, device)

    stop_requested = False

    def request_stop(_signum=None, _frame=None):  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    peak_tflops = _peak_tflops(config, device)
    if peak_tflops is not None:
        peak_tflops = context.reduce(peak_tflops)
    last_checkpoint: Path | None = selected_resume
    interval_start = time.perf_counter()

    def save_checkpoint() -> Path:
        rank_rng = context.gather(capture_rng_state(device))

        def persist() -> Path:
            assert stream is not None
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "stream": stream.state_dict(),
                "rng": rank_rng[0],
                "rank_rng": rank_rng,
                "step": step,
                "tokens_seen": tokens_seen,
                "active_seconds": active_seconds,
                "metrics": metrics_history,
            }
            metadata = {
                "config_digest": config.digest,
                "run_id": config.run.id,
                "step": step,
                "source_revision": source_revision,
                "provenance": provenance,
                "world_size": context.world_size,
            }
            archive_every = config.checkpoint.archive_every_steps
            archive = bool(archive_every and step % archive_every == 0)
            path = manager.save(step, state, metadata, archive=archive)
            hub.upload(path, step)
            return path

        path = context.primary_call(persist)
        # Upload retries and third-party clients may use global random generators.
        # The next forward must see the RNG state that the checkpoint contains.
        restore_rng_state(rank_rng[context.rank], device)
        return path

    try:
        while step < config.train.max_steps:
            training_model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            batch_hashes: list[str] = []
            step_tokens = 0

            for microstep in range(config.train.gradient_accumulation_steps):
                global_batch = None

                def read_batch() -> str:
                    nonlocal global_batch
                    assert stream is not None
                    global_batch = stream.next_batch()
                    return _batch_hash(global_batch)

                batch_hashes.append(context.primary_call(read_batch))
                batch = context.scatter_batch(
                    global_batch,
                    (
                        config.train.batch_size // context.world_size,
                        config.data.sequence_length + 1,
                    ),
                )
                inputs, targets = batch[:, :-1], batch[:, 1:]
                step_tokens += targets.numel()
                sync = (
                    training_model.no_sync()
                    if isinstance(training_model, DistributedDataParallel)
                    and microstep + 1 < config.train.gradient_accumulation_steps
                    else contextlib.nullcontext()
                )
                with sync:
                    with _autocast(device, config.train.precision):
                        output = training_model(inputs)
                        if not isinstance(output, LMOutput):
                            raise TypeError("models must return nanoscope.model.LMOutput")
                        base_loss = F.cross_entropy(
                            output.logits.reshape(-1, output.logits.size(-1)), targets.reshape(-1)
                        )
                        total_loss = base_loss + sum(output.auxiliary_losses.values())
                        scaled_loss = total_loss / config.train.gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                losses.append(float(total_loss.detach()))

            scaler.unscale_(optimizer)
            if config.train.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.grad_clip
                )
            else:
                norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                grad_norm = torch.stack(norms).norm(2) if norms else torch.tensor(0.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if not skipped:
                scheduler.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = context.reduce(now - interval_start, "max")
            interval_start = now
            active_seconds += elapsed
            step += 1
            step_tokens = int(context.reduce(step_tokens))
            tokens_seen += step_tokens
            flops = estimate_training_flops(model, model_spec, step_tokens)
            mfu = flops / (elapsed * peak_tflops * 1e12) if peak_tflops and elapsed else None
            row: dict[str, Any] = {
                "step": step,
                "loss": context.reduce(sum(losses) / len(losses)) / context.world_size,
                "world_size": context.world_size,
                "global_batch_size": config.train.batch_size,
                "per_rank_batch_size": config.train.batch_size // context.world_size,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": float(grad_norm),
                "loss_scale": float(scaler.get_scale()),
                "skipped_update": bool(skipped),
                "tokens": step_tokens,
                "tokens_seen": tokens_seen,
                "tokens_per_second": step_tokens / elapsed,
                "training_flops": flops,
                "cumulative_training_flops": estimate_training_flops(
                    model, model_spec, tokens_seen
                ),
                "mfu": mfu,
                "batch_hash": hashlib.sha256("".join(batch_hashes).encode()).hexdigest()[:16],
                "active_seconds": active_seconds,
            }
            if device.type == "cuda":
                row["gpu_peak_allocated_bytes"] = int(
                    context.reduce(torch.cuda.max_memory_allocated(device), "max")
                )
                row["gpu_peak_reserved_bytes"] = int(
                    context.reduce(torch.cuda.max_memory_reserved(device), "max")
                )
            metrics_history.append(row)

            def log_metrics(values: dict[str, Any] = row, current_step: int = step) -> None:
                assert local_logger is not None and cloud_logger is not None
                local_logger.log(values)
                if current_step % config.logging.every_steps == 0:
                    cloud_logger.log(values)

            context.primary_call(log_metrics)

            acceptance_delay = float(os.getenv("NANOSCOPE_ACCEPTANCE_STEP_DELAY", "0"))
            if acceptance_delay:
                time.sleep(acceptance_delay)

            should_checkpoint = step % config.checkpoint.every_steps == 0
            should_archive = bool(
                config.checkpoint.archive_every_steps
                and step % config.checkpoint.archive_every_steps == 0
            )
            should_stop = stop_requested or (
                stop_after_step is not None and step >= stop_after_step
            )
            stop_file = os.getenv("NANOSCOPE_STOP_FILE")
            should_stop = should_stop or bool(stop_file and Path(stop_file).exists())
            should_stop = bool(context.reduce(float(should_stop), "max"))
            if should_checkpoint or should_archive or should_stop or step == config.train.max_steps:
                last_checkpoint = save_checkpoint()
            if should_stop:
                break
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if last_checkpoint is None:
        last_checkpoint = save_checkpoint()
    return TrainResult(
        run_dir=run_dir,
        final_step=step,
        checkpoint=last_checkpoint,
        metrics=metrics_history,
        stopped_early=step < config.train.max_steps,
    )
