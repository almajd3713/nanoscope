from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Preset:
    name: str
    dataset: str  # a Hugging Face dataset with a "text" column
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
    dataset_config: str | None = None
    hub_data: str | None = None  # Hub dataset repo with this preset's tokens, see publish_data
    train_docs: int | None = None  # None = the whole train split
    holdout_docs: int = 0  # >0: hold out the first N train documents as validation
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


HUB_DATA = "RedhouaneLazib/nanoscope-tokens"

register_preset(Preset(
    name="tinystories-5min",
    dataset="roneneldan/TinyStories",
    tokenizer="bpe",
    vocab_size=4096,
    context_length=256,
    train_docs=100_000,
    hub_data=HUB_DATA,
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
    hub_data=HUB_DATA,
    max_steps=5000,
    batch_size=64,
    learning_rate=3e-4,
    warmup_steps=200,
    eval_interval=100,
    eval_docs=1000,
    sample_interval=500,
    checkpoint_interval=1000,
))

# Research scale (M1-M3): FineWeb-Edu with the GPT-2 tokenizer, 1024-token context.
# About 2B training tokens (4 GB) and 10,000 held-out documents; a study's budget sets
# how many steps each run takes. fp16 because the T4 has no bf16.
register_preset(Preset(
    name="fineweb-edu",
    dataset="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",
    tokenizer="gpt2",
    vocab_size=None,
    context_length=1024,
    train_docs=2_000_000,
    holdout_docs=10_000,
    hub_data=HUB_DATA,
    max_steps=20_000,
    batch_size=32,
    learning_rate=6e-4,
    betas=(0.9, 0.95),
    warmup_steps=500,
    precision="fp16",
    eval_interval=500,
    eval_docs=1000,
    sample_interval=2000,
    checkpoint_interval=500,
))
