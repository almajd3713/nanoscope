from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

from nanoscope.eval.config import check_keys, read_yaml
from nanoscope.paths import config_path


@dataclass(frozen=True)
class InferenceConfig:
    run_dir: Path | None = None
    checkpoint: Path | str = "latest"
    training_config: Path | None = None
    device: str = "auto"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 1337
    output: Path | None = None

    def __post_init__(self) -> None:
        if self.checkpoint == "latest" and self.run_dir is None:
            raise ValueError("run_dir is required when checkpoint is latest")
        if not isinstance(self.checkpoint, (str, Path)) or not str(self.checkpoint):
            raise ValueError("checkpoint must be latest or a checkpoint directory")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for name, value in (("temperature", self.temperature), ("top_p", self.top_p)):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0 (0 means greedy)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def load_inference_config(path: str | Path) -> InferenceConfig:
    raw = read_yaml(path)
    check_keys(raw, {field.name for field in fields(InferenceConfig)}, set())
    for name in ("run_dir", "training_config", "output"):
        if raw.get(name) is not None:
            raw[name] = config_path(raw[name], path)
    if raw.get("checkpoint", "latest") != "latest":
        raw["checkpoint"] = config_path(raw["checkpoint"], path)
    return InferenceConfig(**raw)
