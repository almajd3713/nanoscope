from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nanoscope.config import DataConfig, TokenizerConfig, _section


def read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def check_keys(raw: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    if set(raw) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(raw) - allowed)}")
    if required - set(raw):
        raise ValueError(f"missing configuration keys: {sorted(required - set(raw))}")


def positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def relative_path(value: str, config_path: str | Path) -> Path:
    """Evaluation/study paths are relative to their configuration file."""
    if not isinstance(value, str) or not value:
        raise ValueError("paths must be non-empty strings")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config_path).parent / path).resolve()


@dataclass(frozen=True)
class CorpusConfig:
    data: DataConfig
    tokenizer: TokenizerConfig
    sequences: int
    output: Path
    max_documents: int = 1_000_000


def load_corpus_config(path: str | Path) -> CorpusConfig:
    raw = read_yaml(path)
    check_keys(
        raw,
        {"data", "tokenizer", "sequences", "output", "max_documents"},
        {"data", "tokenizer", "sequences", "output"},
    )
    data = _section(DataConfig, raw["data"], "data")
    tokenizer = _section(TokenizerConfig, raw["tokenizer"], "tokenizer")
    positive_int(raw["sequences"], "sequences")
    positive_int(raw.get("max_documents", 1_000_000), "max_documents")
    positive_int(data.sequence_length, "data.sequence_length")
    if data.sequence_length < 2:
        raise ValueError("data.sequence_length must be at least 2")
    if data.source not in {"fixture", "fineweb"}:
        raise ValueError("data.source must be fixture or fineweb")
    if data.partition is None or data.partition.split not in {"validation", "test"}:
        raise ValueError("corpus requires a validation or test data.partition")
    if data.source == "fixture" and (
        not isinstance(data.documents, list)
        or not data.documents
        or any(not isinstance(text, str) for text in data.documents)
    ):
        raise ValueError("fixture corpus requires a non-empty list of text documents")
    if data.source == "fineweb" and (
        not isinstance(data.revision, str)
        or len(data.revision) != 40
        or any(c not in "0123456789abcdef" for c in data.revision)
    ):
        raise ValueError("FineWeb corpus requires a pinned 40-character commit revision")
    return CorpusConfig(
        data,
        tokenizer,
        raw["sequences"],
        relative_path(raw["output"], path),
        raw.get("max_documents", 1_000_000),
    )


@dataclass(frozen=True)
class EvalConfig:
    corpus: Path
    output: Path
    batch_size: int = 4
    device: str = "cpu"
    precision: str = "fp32"
    corpus_hash: str | None = None
    every_steps: int = 100
    final: bool = True
    max_seconds: float = 300.0

    def __post_init__(self) -> None:
        positive_int(self.batch_size, "batch_size")
        positive_int(self.every_steps, "every_steps")
        if type(self.final) is not bool:
            raise ValueError("final must be a boolean")
        if (
            type(self.max_seconds) not in {int, float}
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU evaluation requires fp32")
        if self.corpus_hash is not None and (
            not isinstance(self.corpus_hash, str)
            or len(self.corpus_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.corpus_hash)
        ):
            raise ValueError("corpus_hash must be a SHA-256 hex digest")


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = read_yaml(path)
    check_keys(
        raw,
        {
            "corpus",
            "output",
            "batch_size",
            "device",
            "precision",
            "corpus_hash",
            "every_steps",
            "final",
            "max_seconds",
        },
        {"corpus", "output"},
    )
    return EvalConfig(
        **{
            **raw,
            "corpus": relative_path(raw["corpus"], path),
            "output": relative_path(raw["output"], path),
        }
    )
