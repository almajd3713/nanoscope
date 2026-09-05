from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration is invalid."""


@dataclass
class RunConfig:
    id: str
    seed: int = 1337
    output_dir: str = "runs"
    require_clean_repo: bool = False


@dataclass
class DataConfig:
    source: str = "fixture"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    revision: str | None = None
    split: str = "train"
    text_field: str = "text"
    sequence_length: int = 128
    shuffle_buffer: int = 10_000
    documents: list[str] = field(default_factory=list)


@dataclass
class TokenizerConfig:
    name: str = "gpt2"
    eos_token_id: int = 50_256


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 0


@dataclass
class TrainConfig:
    max_steps: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    device: str = "auto"
    precision: str = "fp32"
    deterministic: bool = True
    grad_clip: float | None = 1.0


@dataclass
class CheckpointConfig:
    every_steps: int = 100
    archive_every_steps: int = 0
    keep_last: int = 3
    hub_policy: str = "disabled"
    hub_repo: str | None = None


@dataclass
class LoggingConfig:
    every_steps: int = 1
    wandb_mode: str = "disabled"
    project: str = "nanoscope"
    entity: str | None = None


@dataclass
class PerformanceConfig:
    peak_tflops: float | None = None


@dataclass
class Config:
    run: RunConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    train: TrainConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    performance: PerformanceConfig

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def immutable_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value["run"].pop("output_dir", None)
        value["logging"].pop("wandb_mode", None)
        value["logging"].pop("entity", None)
        value["checkpoint"].pop("hub_policy", None)
        value["checkpoint"].pop("hub_repo", None)
        return value

    @property
    def digest(self) -> str:
        payload = json.dumps(self.immutable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _section(cls: type[Any], raw: Mapping[str, Any], name: str) -> Any:
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {name}: {', '.join(sorted(unknown))}")
    try:
        if cls is OptimizerConfig and isinstance(raw.get("betas"), list):
            raw = {**raw, "betas": tuple(raw["betas"])}
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"invalid {name} section: {exc}") from exc


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")

    section_types = {
        "run": RunConfig,
        "data": DataConfig,
        "tokenizer": TokenizerConfig,
        "model": ModelConfig,
        "optimizer": OptimizerConfig,
        "scheduler": SchedulerConfig,
        "train": TrainConfig,
        "checkpoint": CheckpointConfig,
        "logging": LoggingConfig,
        "performance": PerformanceConfig,
    }
    unknown = set(raw) - set(section_types)
    missing = set(section_types) - set(raw)
    if unknown:
        raise ConfigError(f"unknown top-level sections: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing top-level sections: {', '.join(sorted(missing))}")

    config = Config(**{name: _section(cls, raw[name], name) for name, cls in section_types.items()})
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if not config.run.id or "/" in config.run.id or ".." in config.run.id:
        raise ConfigError("run.id must be a non-empty filesystem-safe identifier")
    if config.run.seed < 0:
        raise ConfigError("run.seed must be non-negative")
    if config.data.source not in {"fixture", "fineweb"}:
        raise ConfigError("data.source must be fixture or fineweb")
    if config.data.source == "fixture" and not config.data.documents:
        raise ConfigError("fixture data requires at least one document")
    if config.data.sequence_length < 2:
        raise ConfigError("data.sequence_length must be at least 2")
    if config.data.shuffle_buffer < 1:
        raise ConfigError("data.shuffle_buffer must be positive")
    if config.tokenizer.name not in {"byte", "gpt2"}:
        raise ConfigError("tokenizer.name must be byte or gpt2")
    if config.optimizer.name != "adamw":
        raise ConfigError("M0 supports only the adamw optimizer")
    if config.scheduler.name not in {"constant", "cosine"}:
        raise ConfigError("scheduler.name must be constant or cosine")
    if config.train.max_steps < 1 or config.train.batch_size < 1:
        raise ConfigError("max_steps and batch_size must be positive")
    if config.train.gradient_accumulation_steps < 1:
        raise ConfigError("gradient_accumulation_steps must be positive")
    if config.train.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("train.device must be auto, cpu, or cuda")
    if config.train.precision not in {"fp32", "fp16"}:
        raise ConfigError("train.precision must be fp32 or fp16")
    if config.train.device == "cpu" and config.train.precision != "fp32":
        raise ConfigError("CPU runs must use fp32")
    if config.checkpoint.every_steps < 1 or config.checkpoint.keep_last < 1:
        raise ConfigError("checkpoint.every_steps and keep_last must be positive")
    if config.checkpoint.hub_policy not in {"disabled", "best_effort", "required"}:
        raise ConfigError("checkpoint.hub_policy is invalid")
    if config.logging.wandb_mode not in {"disabled", "offline", "online"}:
        raise ConfigError("logging.wandb_mode must be disabled, offline, or online")


def dump_config(config: Config, path: str | Path) -> None:
    # JSON normalization converts tuples to YAML-safe lists without Python tags.
    normalized = json.loads(json.dumps(config.to_dict()))
    Path(path).write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")
