from __future__ import annotations

import inspect
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from nanoscope.config import config_from_dict, load_config
from nanoscope.data.tokenizer import build_tokenizer, tokenizer_identity
from nanoscope.eval.artifacts import file_hash, fingerprint, read_json
from nanoscope.inference.config import InferenceConfig, load_inference_config
from nanoscope.model import LMOutput, build_model
from nanoscope.train.checkpoint import validate_checkpoint


def complete_checkpoint(path: Path) -> bool:
    """Require the core inference files to be covered by the checksum manifest."""
    try:
        manifest = read_json(path / "manifest.json")
        files = manifest.get("files")
        return (
            isinstance(files, dict)
            and {"state.pt", "metadata.json"} <= files.keys()
            and validate_checkpoint(path)
        )
    except (OSError, ValueError, TypeError):
        return False


def resolve_checkpoint(config: InferenceConfig) -> Path:
    """Select without creating directories or updating the run's latest pointer."""
    if config.checkpoint != "latest":
        path = Path(config.checkpoint).resolve()
        if not complete_checkpoint(path):
            raise ValueError(f"invalid or corrupt checkpoint: {path}")
        return path
    assert config.run_dir is not None
    root = Path(config.run_dir) / "checkpoints"
    candidates = sorted(
        (p for p in root.glob("step_*") if p.name.removeprefix("step_").isdigit()),
        key=lambda p: int(p.name.removeprefix("step_")),
        reverse=True,
    )
    for path in candidates:
        if complete_checkpoint(path):
            return path.resolve()
    raise ValueError(f"no complete checkpoints in {root}; train or attach a checkpoint first")


def sample_token(
    logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator
) -> int:
    if temperature == 0:
        return int(logits.argmax().item())
    # Subtract before dividing so small positive temperatures cannot overflow.
    values = logits.double()
    probabilities = torch.softmax((values - values.max()) / temperature, dim=-1)
    ordered, indices = probabilities.sort(descending=True)
    remove = ordered.cumsum(dim=-1) - ordered >= top_p
    ordered = ordered.masked_fill(remove, 0)
    index = torch.multinomial(ordered, 1, generator=generator)
    return int(indices[index].item())


class GenerationSession:
    """Load once; generate repeatedly without initializing training or cloud services."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        checkpoint = resolve_checkpoint(config)
        metadata = read_json(checkpoint / "metadata.json")
        manifest = read_json(checkpoint / "manifest.json")
        if not {"state.pt", "metadata.json"} <= manifest["files"].keys():
            raise ValueError("checkpoint manifest must cover state.pt and metadata.json")
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        if config.training_config is not None:
            training = load_config(config.training_config)
        elif "config" in state:
            training = config_from_dict(state["config"])
        else:
            legacy = checkpoint.parent.parent / "resolved-config.yaml"
            if not legacy.is_file():
                raise ValueError(
                    "legacy checkpoint requires training_config with its resolved config"
                )
            training = load_config(legacy)
        if training.digest != metadata["config_digest"]:
            raise ValueError("training configuration does not match checkpoint config_digest")
        if (
            state["step"] != metadata["step"]
            or manifest["step"] != metadata["step"]
            or metadata["run_id"] != training.run.id
        ):
            raise ValueError("checkpoint step/run metadata disagrees with state or configuration")
        token_identity = tokenizer_identity(training.tokenizer)
        if metadata.get("tokenizer") is not None and metadata["tokenizer"] != token_identity:
            raise ValueError("checkpoint tokenizer differs from current tokenizer implementation")
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        # Model initialization should not consume the notebook's global CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            self.model, _ = build_model(training.model.name, training.model.params)
        self.model.load_state_dict(state["model"], strict=True)
        del state  # Release optimizer/RNG/training state before moving weights to the GPU.
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.tokenizer = build_tokenizer(training.tokenizer)
        self.context_length = training.data.sequence_length
        self.vocab_size = int(token_identity["vocab_size"])
        self.byte_tokenizer = training.tokenizer.name == "byte"
        self.identity = {
            "checkpoint": {
                "path": str(checkpoint),
                "hash": fingerprint(
                    {name: manifest["files"][name] for name in ("state.pt", "metadata.json")}
                ),
                "step": metadata["step"],
            },
            "run_id": training.run.id,
            "training_config_digest": training.digest,
            "tokenizer": token_identity,
            "tokenizer_verified": metadata.get("tokenizer") == token_identity,
            "implementation": {
                "generation": file_hash(Path(__file__)),
                "model": file_hash(Path(inspect.getfile(type(self.model)))),
                "tokenizer": file_hash(Path(inspect.getfile(type(self.tokenizer)))),
            },
        }

    @classmethod
    def from_config(cls, config: str | Path | InferenceConfig) -> GenerationSession:
        return cls(
            load_inference_config(config) if not isinstance(config, InferenceConfig) else config
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        overrides = {
            key: value
            for key, value in {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }.items()
            if value is not None
        }
        settings = replace(self.config, **overrides)
        prompt_ids = self.tokenizer.encode(prompt)
        context = prompt_ids or [self.tokenizer.eos_token_id]
        context = context[-self.context_length :]
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(settings.seed)
        generated: list[int] = []
        finish_reason = "length"
        started = time.perf_counter()
        self.model.eval()
        for _ in range(settings.max_new_tokens):
            output = self.model(input_ids)
            if not isinstance(output, LMOutput) or output.logits.shape != (
                1,
                input_ids.shape[1],
                self.vocab_size,
            ):
                raise ValueError(
                    "model must return LMOutput with shape [1, context, tokenizer vocab]"
                )
            logits = output.logits[0, -1].float().clone()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("model returned non-finite generation logits")
            if self.byte_tokenizer:
                logits[256 : self.tokenizer.eos_token_id] = -torch.inf
            token = sample_token(logits, settings.temperature, settings.top_p, generator)
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                finish_reason = "eos"
                break
            input_ids = torch.cat((input_ids, input_ids.new_tensor([[token]])), dim=1)
            input_ids = input_ids[:, -self.context_length :]
        elapsed = time.perf_counter() - started
        completion = self.tokenizer.decode(generated)
        result = {
            "schema_version": 1,
            **self.identity,
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion,
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "finish_reason": finish_reason,
            "sampling": {
                "max_new_tokens": settings.max_new_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "seed": settings.seed,
            },
            "context": {
                "length": self.context_length,
                "policy": "sliding_window",
                "prompt_tokens_dropped": max(0, len(prompt_ids) - self.context_length),
                "empty_prompt_uses_eos": not prompt_ids,
            },
            "runtime": {
                "device": str(self.device),
                "device_name": (
                    torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
                ),
                "precision": "fp32",
                "torch_version": str(torch.__version__),
                "num_threads": torch.get_num_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "seconds": elapsed,
            },
        }
        if settings.output is not None:
            destination = Path(settings.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
        return result


def read_prompts(path: str | Path) -> list[str]:
    """One JSON string per line, preserving empty and multiline prompts exactly."""
    prompts = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            prompt = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid prompt JSON on line {number}") from exc
        if not isinstance(prompt, str):
            raise ValueError(f"prompt on line {number} must be a JSON string")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("prompt file must contain at least one JSON string")
    return prompts
