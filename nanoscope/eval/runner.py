from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nanoscope.config import config_from_dict, load_config
from nanoscope.data.tokenizer import tokenizer_identity
from nanoscope.eval.artifacts import file_hash, fingerprint, read_json, write_json
from nanoscope.eval.config import EvalConfig, positive_int
from nanoscope.eval.corpus import FrozenCorpus, load_corpus, source_identity
from nanoscope.model import LMOutput, build_model
from nanoscope.model.registry import non_embedding_parameter_count
from nanoscope.provenance import runtime_provenance as _runtime_provenance
from nanoscope.train.checkpoint import validate_checkpoint
from nanoscope.train.determinism import capture_rng_state, restore_rng_state

EVALUATOR_VERSION = "next-token-cross-entropy-v1"
CPU = torch.device("cpu")


class EvaluationCancelled(RuntimeError):
    """A stop request interrupted evaluation; no partial score may be published."""


def score_model(
    model: nn.Module,
    tokens: np.ndarray,
    *,
    batch_size: int = 4,
    device: torch.device = CPU,
    precision: str = "fp32",
    expected_vocab_size: int | None = None,
    max_seconds: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Score fixed blocks without repacking, dropout, or auxiliary penalties."""
    positive_int(batch_size, "batch_size")
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError("evaluation requires non-empty [sequences, context + 1] tokens")
    if not np.issubdtype(tokens.dtype, np.integer) or int(tokens.min()) < 0:
        raise ValueError("evaluation token IDs must be non-negative integers")
    if precision not in {"fp32", "fp16"} or (precision == "fp16" and device.type != "cuda"):
        raise ValueError("fp16 evaluation requires CUDA; otherwise select fp32")
    flags = [(module, module.training) for module in model.modules()]
    # Restore registered buffers even when a model replaces/creates them in forward.
    buffers = [
        (module, dict(module._buffers), set(module._non_persistent_buffers_set))
        for module in model.modules()
    ]
    saved_buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    rng = capture_rng_state(device)
    nll_sum = 0.0
    count = 0
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(tokens), batch_size):
                if cancelled is not None and cancelled():
                    raise EvaluationCancelled("evaluation interrupted by a stop request")
                if max_seconds is not None and time.perf_counter() - started > max_seconds:
                    raise TimeoutError("evaluation exceeded max_seconds; no score was published")
                # Copy read-only mmap data, keeping only one batch on the accelerator.
                batch = torch.tensor(
                    np.array(tokens[offset : offset + batch_size]), dtype=torch.long, device=device
                )
                inputs, targets = batch[:, :-1], batch[:, 1:]
                autocast = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if precision == "fp16"
                    else contextlib.nullcontext()
                )
                with autocast:
                    output = model(inputs)
                if not isinstance(output, LMOutput):
                    raise TypeError("models must return nanoscope.model.LMOutput")
                if output.logits.ndim != 3 or output.logits.shape[:2] != targets.shape:
                    raise ValueError("model logits must have shape [batch, context, vocabulary]")
                vocab_size = output.logits.size(-1)
                if expected_vocab_size is not None and vocab_size != expected_vocab_size:
                    raise ValueError("model vocabulary size differs from the corpus tokenizer")
                if vocab_size < 1 or int(batch.max()) >= vocab_size:
                    raise ValueError("corpus token ID exceeds model vocabulary")
                losses = F.cross_entropy(
                    output.logits.float().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                )
                subtotal = float(losses.sum(dtype=torch.float64).item())
                if not math.isfinite(subtotal):
                    raise ValueError("non-finite evaluation cross-entropy")
                nll_sum += subtotal
                count += targets.numel()
                del output, losses, batch, inputs, targets
            if cancelled is not None and cancelled():
                raise EvaluationCancelled("evaluation interrupted by a stop request")
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                raise TimeoutError("evaluation exceeded max_seconds; no score was published")
    finally:
        with torch.no_grad():
            for buffer, saved in saved_buffers:
                if buffer.shape != saved.shape:
                    buffer.resize_(saved.shape)
                buffer.copy_(saved)
        for module, original, non_persistent in buffers:
            module._buffers.clear()
            module._buffers.update(original)
            module._non_persistent_buffers_set = non_persistent
        for module, training in flags:
            module.training = training
        restore_rng_state(rng, device)
    loss = nll_sum / count
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None
    return {
        "cross_entropy_nats": loss,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
        "nll_sum": nll_sum,
        "scored_tokens": count,
        "sequence_count": len(tokens),
        "eval_seconds": time.perf_counter() - started,
    }


def evaluator_code_hash() -> str:
    root = Path(__file__).parent
    return fingerprint({path.name: file_hash(path) for path in sorted(root.glob("*.py"))})


def implementation_hash() -> str:
    root = Path(__file__).parent.parent
    return fingerprint(
        {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))}
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    config: EvalConfig,
    training_config: str | Path | None = None,
    *,
    model: nn.Module | None = None,
    corpus: FrozenCorpus | None = None,
    snapshot: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    if not validate_checkpoint(checkpoint):
        raise ValueError(f"invalid or corrupt checkpoint: {checkpoint}")
    corpus = corpus if corpus is not None else load_corpus(config.corpus, config.corpus_hash)
    metadata = read_json(checkpoint / "metadata.json")
    manifest = read_json(checkpoint / "manifest.json")
    # Evaluation sidecars may be attached later without changing the scored checkpoint.
    checkpoint_hash = fingerprint(
        {name: manifest["files"][name] for name in ("state.pt", "metadata.json")}
    )
    state = (
        snapshot
        if snapshot is not None
        else torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    )
    if training_config is not None:
        training = load_config(training_config)
    elif "config" in state:
        training = config_from_dict(state["config"])
    else:
        candidate = checkpoint.parent.parent / "resolved-config.yaml"
        if not candidate.is_file():
            raise ValueError(
                "legacy checkpoint requires --training-config with its resolved config"
            )
        training = load_config(candidate)
    if training.digest != metadata["config_digest"]:
        raise ValueError("training configuration does not match checkpoint config_digest")
    if state["step"] != metadata["step"] or metadata["run_id"] != training.run.id:
        raise ValueError("checkpoint step/run metadata disagrees with state or configuration")
    token_identity = tokenizer_identity(training.tokenizer)
    if token_identity != corpus.manifest["recipe"]["tokenizer"]:
        raise ValueError("checkpoint tokenizer differs from evaluation corpus tokenizer")
    if training.data.sequence_length != corpus.manifest["recipe"]["sequence_length"]:
        raise ValueError("checkpoint training context differs from evaluation corpus context")
    if metadata.get("tokenizer") is not None and metadata["tokenizer"] != token_identity:
        raise ValueError("training tokenizer implementation differs from current implementation")
    device = torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cpu" and config.precision != "fp32":
        raise ValueError("CPU evaluation requires fp32")
    runtime = _runtime_provenance(device)
    runtime["num_threads"] = torch.get_num_threads()
    runtime["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    runtime["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    runtime["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    identity = {
        "checkpoint_hash": checkpoint_hash,
        "corpus_hash": corpus.fingerprint,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_code_hash": evaluator_code_hash(),
        "implementation_hash": implementation_hash(),
        "batch_size": config.batch_size,
        "precision": config.precision,
        "runtime": runtime,
        "training_config_digest": training.digest,
    }
    result_id = fingerprint(identity)
    destination = (output_dir or config.output / training.run.id) / f"{result_id}.json"
    if destination.exists():
        existing = load_result(destination)
        if existing["identity"] != identity:
            raise ValueError("existing evaluation identity mismatch")
        return destination
    live_model = model is not None
    if model is None:
        model, _spec = build_model(training.model.name, training.model.params)
        model.load_state_dict(state["model"], strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    non_embedding = non_embedding_parameter_count(model)
    history = state.get("metrics", [])
    last = next((row for row in reversed(history) if row["step"] == state["step"]), {})
    training_info = {
        "tokens_seen": state.get("tokens_seen"),
        "cumulative_training_flops": last.get("cumulative_training_flops"),
        "flop_estimator": metadata.get("flop_estimator"),
        "flops_are_estimates": True,
        "total_parameters": parameters,
        "non_embedding_parameters": non_embedding,
        "active_seconds": state.get("active_seconds"),
        "tokens_per_second": last.get("train_tokens_per_second", last.get("tokens_per_second")),
        "train_seconds": state.get("train_seconds"),
        "gpu_peak_allocated_bytes": last.get("gpu_peak_allocated_bytes"),
        "timing_definition": last.get("timing_definition", "legacy-inter-step-wall-v1"),
        "source": source_identity(training.data, metadata.get("source_revision")),
        "tokenizer_verified": metadata.get("tokenizer") == token_identity,
    }
    partition = training.data.partition
    corpus_partition = dict(corpus.manifest["recipe"]["partition"])
    corpus_partition.pop("split")
    training_info["held_out_verified"] = bool(
        metadata.get("partition_version") == 1
        and partition is not None
        and partition.split == "train"
        and partition.scheme() == corpus_partition
        and training_info["source"] == corpus.manifest["recipe"]["source"]
        and training_info["tokenizer_verified"]
    )
    # Do not move optimizer state or allocate a second model replica on the GPU.
    del state, history
    if not live_model:
        model.to(device=device, dtype=torch.float32)
    elif any(
        parameter.device.type != device.type or parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("periodic evaluation requires FP32 model weights on the training device")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    metrics = score_model(
        model,
        corpus.tokens,
        batch_size=config.batch_size,
        device=device,
        precision=config.precision,
        expected_vocab_size=int(token_identity["vocab_size"]),
        max_seconds=config.max_seconds,
        cancelled=cancelled,
    )
    if device.type == "cuda":
        metrics["eval_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result = {
        "schema_version": 1,
        "complete": True,
        "result_id": result_id,
        "identity": identity,
        "checkpoint": {"path": str(checkpoint), "hash": checkpoint_hash, "step": metadata["step"]},
        "run": {
            "id": training.run.id,
            "seed": training.run.seed,
            "config": training.to_dict(),
            "provenance": metadata.get("provenance", {}),
        },
        "corpus": corpus.manifest,
        "metrics": metrics,
        "training": training_info,
    }
    result["result_hash"] = fingerprint(result)
    write_json(destination, result)
    return destination


def load_result(path: Path) -> dict[str, Any]:
    result = read_json(path)
    if result.get("schema_version") != 1 or result.get("complete") is not True:
        raise ValueError(f"unsupported or incomplete evaluation result: {path}")
    if result.get("result_hash") != fingerprint(
        {key: value for key, value in result.items() if key != "result_hash"}
    ):
        raise ValueError(f"evaluation result checksum mismatch: {path}")
    if result["result_id"] != fingerprint(result["identity"]):
        raise ValueError(f"evaluation identity checksum mismatch: {path}")
    metrics = result["metrics"]
    if not math.isfinite(metrics["cross_entropy_nats"]) or metrics["scored_tokens"] <= 0:
        raise ValueError(f"invalid evaluation score: {path}")
    return result
