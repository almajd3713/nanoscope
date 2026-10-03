from __future__ import annotations

import contextlib
import json
import math
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope.dataset import Data
from nanoscope.presets import Preset
from nanoscope.tokenizer import Tokenizer


@dataclass
class TrainResult:
    run_dir: Path
    final_step: int
    metrics: list[dict[str, Any]]
    stopped_early: bool

    def loss_at(self, step: int) -> float | None:
        for row in self.metrics:
            if row["step"] == step:
                return row["loss"]
        return None


def _cosine_lr(step: int, warmup: int, total: int, lr: float) -> float:
    if step < warmup:
        return lr * max(step, 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _peak_tflops(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device).lower()
    if "t4" in name:
        return 65.0
    if "p100" in name:
        return 21.2
    return None


def _non_embedding_params(model: nn.Module) -> int:
    emb_ids = {
        id(p) for m in model.modules() if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    return sum(p.numel() for p in model.parameters() if id(p) not in emb_ids)


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _split_output(output: Any) -> tuple[torch.Tensor, Any]:
    """Models return logits, or (logits, aux_loss)."""
    if isinstance(output, torch.Tensor):
        return output, 0.0
    return output[0], output[1] if len(output) > 1 else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, preset: Preset, device: torch.device) -> float:
    """Mean loss per token over the whole fixed validation set, in non-overlapping windows."""
    model.eval()
    val = torch.from_numpy(data.val.astype(np.int64))
    ctx, bs = preset.context_length, preset.batch_size
    n_full = (len(val) - 1) // ctx
    x = val[: n_full * ctx].view(n_full, ctx)
    y = val[1 : n_full * ctx + 1].view(n_full, ctx)
    batches = [(x[i : i + bs], y[i : i + bs]) for i in range(0, n_full, bs)]
    rest = len(val) - 1 - n_full * ctx
    if rest:
        batches.append((val[-rest - 1 : -1][None], val[-rest:][None]))

    total, count = 0.0, 0
    for inputs, targets in batches:
        with _autocast(device, preset.precision):
            logits, _ = _split_output(model(inputs.to(device)))
        total += F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets.to(device).reshape(-1),
            reduction="sum",
        ).item()
        count += targets.numel()
    model.train()
    return total / count


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    context_length: int = 256,
    seed: int = 42,
) -> str:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt) if prompt else [tokenizer.eos_token_id]
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new_tokens):
        ctx = torch.tensor([tokens[-context_length:]], dtype=torch.long, device=device)
        logits, _ = _split_output(model(ctx))
        probs = torch.softmax(logits[:, -1, :].float() / temperature, dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=gen).item())
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)
    model.train(was_training)
    return tokenizer.decode(tokens)


def _save_checkpoint(run_dir: Path, step: int, state: dict[str, Any], keep: int) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, **state}, tmp)
    tmp.replace(path)
    latest_tmp = run_dir / "latest.json.tmp"
    latest_tmp.write_text(json.dumps({"checkpoint": path.name, "step": step}), encoding="utf-8")
    latest_tmp.replace(run_dir / "latest.json")
    for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
        old.unlink()


def _load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any] | None:
    latest_file = run_dir / "latest.json"
    if not latest_file.exists():
        return None
    info = json.loads(latest_file.read_text(encoding="utf-8"))
    path = run_dir / "checkpoints" / info["checkpoint"]
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def _rng_state(gen: torch.Generator) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data": gen.get_state(),
    }


def _restore_rng(state: dict[str, Any], gen: torch.Generator) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    gen.set_state(state["data"].cpu())


def clear_run(run_dir: Path) -> None:
    """Remove a run's progress (checkpoints and metrics) so it starts from step 0."""
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    for name in ("latest.json", "metrics.jsonl", "samples.txt"):
        (run_dir / name).unlink(missing_ok=True)


def train(
    model: nn.Module,
    data: Data,
    preset: Preset,
    run_dir: Path,
    device: torch.device,
    seed: int = 0,
    resume: bool = True,
    on_step: Any = None,
    on_eval: Any = None,
) -> TrainResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)  # drives which batches the model sees

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        _param_groups(model, preset.weight_decay),
        lr=preset.learning_rate, betas=preset.betas, eps=preset.eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_lr(step, preset.warmup_steps, preset.max_steps, 1.0)
    )
    scaler = _grad_scaler(device, preset.precision)

    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        clear_run(run_dir)
    start_step = 0
    state = _load_checkpoint(run_dir, device)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng"], gen)
        start_step = int(state["step"])
        if start_step >= preset.max_steps:
            print(f"[nanoscope] already trained ({start_step} steps); loaded the final "
                  "checkpoint. Pass resume=False to train again.", flush=True)
        else:
            print(f"[nanoscope] resuming from step {start_step}", flush=True)

    # Keep metrics.jsonl in step with the checkpoint: drop rows a crash left past it.
    metrics_path = run_dir / "metrics.jsonl"
    metrics: list[dict[str, Any]] = []
    if start_step > 0 and metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["step"] <= start_step:
                metrics.append(row)
    metrics_path.write_text("".join(json.dumps(r) + "\n" for r in metrics), encoding="utf-8")

    def checkpoint(step: int) -> None:
        _save_checkpoint(run_dir, step, {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _rng_state(gen),
        }, preset.keep_checkpoints)

    peak = _peak_tflops(device)
    n_params = _non_embedding_params(model)

    # Ctrl-C finishes the current step, saves a checkpoint and returns.
    stop_requested = False
    can_trap = threading.current_thread() is threading.main_thread()
    prev_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    if can_trap:
        signal.signal(signal.SIGINT, _on_sigint)

    step = start_step
    try:
        for step in range(start_step + 1, preset.max_steps + 1):
            t0 = time.perf_counter()

            batch = data.train.get_batch(preset.batch_size, gen).to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            with _autocast(device, preset.precision):
                logits, aux_loss = _split_output(model(inputs))
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                ) + aux_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            tokens = targets.numel()
            flops = 6 * n_params * tokens
            mfu = flops / (elapsed * peak * 1e12) if peak and elapsed else None

            row: dict[str, Any] = {
                "step": step,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens / elapsed,
                "mfu": mfu,
                "elapsed": elapsed,
            }

            last = step == preset.max_steps
            if step % preset.eval_interval == 0 or last:
                val_loss = evaluate(model, data, preset, device)
                row["val_loss"] = val_loss
                row["val_bpb"] = data.bits_per_byte(val_loss)
                if on_eval:
                    on_eval(step, val_loss)

            if step % preset.sample_interval == 0 or last:
                row["sample"] = generate(
                    model, data.tokenizer, preset.sample_prompt, preset.sample_length,
                    preset.sample_temperature, preset.context_length,
                )

            metrics.append(row)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if on_step:
                on_step(step, row)

            if step % preset.checkpoint_interval == 0 or last or stop_requested:
                checkpoint(step)
            if stop_requested:
                break
    finally:
        if can_trap:
            signal.signal(signal.SIGINT, prev_handler)

    with (run_dir / "samples.txt").open("w", encoding="utf-8") as f:
        for r in metrics:
            if "sample" in r:
                f.write(f"--- step {r['step']} ---\n{r['sample']}\n\n")

    return TrainResult(
        run_dir=run_dir,
        final_step=step,
        metrics=metrics,
        stopped_early=stop_requested,
    )
