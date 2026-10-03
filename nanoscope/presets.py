from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Preset:
    name: str
    dataset: str
    tokenizer: str  # "bpe" (trained on this dataset), "gpt2" or "bytes"
    vocab_size: int | None  # target size for "bpe"; fixed by the tokenizer otherwise
    context_length: int
    max_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 100
    grad_clip: float = 1.0
    precision: str = "fp32"
    train_docs: int | None = None  # None = the whole train split
    tokenizer_train_docs: int = 20_000
    eval_interval: int = 50
    eval_docs: int = 200  # the first N validation documents, the same for every run
    sample_interval: int = 100
    sample_length: int = 200
    sample_prompt: str = ""
    sample_temperature: float = 0.8
    checkpoint_interval: int = 500
    keep_checkpoints: int = 2

    def override(self, **kwargs: Any) -> Preset:
        known = {f.name for f in fields(self)}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"unknown preset fields: {', '.join(sorted(unknown))}")
        return Preset(**{**{f.name: getattr(self, f.name) for f in fields(self)}, **kwargs})


_PRESETS: dict[str, Preset] = {}


def register_preset(preset: Preset) -> None:
    _PRESETS[preset.name] = preset


def get_preset(name: str) -> Preset:
    if name not in _PRESETS:
        choices = ", ".join(sorted(_PRESETS)) or "<none>"
        raise KeyError(f"unknown preset {name!r}; available: {choices}")
    return _PRESETS[name]


def list_presets() -> list[str]:
    return sorted(_PRESETS)


register_preset(Preset(
    name="tinystories-5min",
    dataset="roneneldan/TinyStories",
    tokenizer="bpe",
    vocab_size=4096,
    context_length=256,
    train_docs=100_000,
    max_steps=500,
    batch_size=8,
    learning_rate=3e-3,
    warmup_steps=50,
    eval_interval=50,
    eval_docs=200,
    sample_interval=100,
    checkpoint_interval=500,
))

register_preset(Preset(
    name="tinystories-30min",
    dataset="roneneldan/TinyStories",
    tokenizer="bpe",
    vocab_size=4096,
    context_length=256,
    train_docs=1_000_000,
    max_steps=5000,
    batch_size=64,
    learning_rate=3e-4,
    warmup_steps=200,
    eval_interval=100,
    eval_docs=1000,
    sample_interval=500,
    checkpoint_interval=1000,
))
