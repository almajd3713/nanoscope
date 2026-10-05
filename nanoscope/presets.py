from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


def _help(text: str, **kwargs: Any) -> Any:
    return field(metadata={"help": text}, **kwargs)


@dataclass(frozen=True)
class Preset:
    name: str = _help("Name the preset is registered and listed under.")
    dataset: str = _help('A Hugging Face dataset with a "text" column.')
    tokenizer: str = _help('"bpe" (trained on this dataset), "gpt2" or "bytes".')
    vocab_size: int | None = _help('Target vocabulary size for "bpe"; fixed by the tokenizer '
                                   "otherwise.")
    context_length: int = _help("Tokens the model sees at once.")
    max_steps: int = _help("Training steps. Each step reads batch_size sequences.")
    batch_size: int = _help("Sequences per training step.")
    learning_rate: float = _help("Peak learning rate; it warms up, then decays on a cosine.")
    weight_decay: float = _help("AdamW weight decay (not applied to norms and biases).",
                                default=0.1)
    betas: tuple[float, float] = _help("AdamW beta1 and beta2.", default=(0.9, 0.999))
    eps: float = _help("AdamW epsilon.", default=1e-8)
    warmup_steps: int = _help("Steps over which the learning rate rises from zero.", default=100)
    grad_clip: float = _help("Largest gradient norm; bigger gradients are scaled down.",
                             default=1.0)
    precision: str = _help('"fp32", or "fp16" for mixed precision on a CUDA GPU.', default="fp32")
    dataset_config: str | None = _help("The dataset's configuration name, if it has one.",
                                       default=None)
    hub_data: str | None = _help("Hub dataset repo holding this preset's tokens (see "
                                 "publish_data), so they download instead of tokenizing.",
                                 default=None)
    train_docs: int | None = _help("Documents to train on; None is the whole train split.",
                                   default=None)
    holdout_docs: int = _help("When above 0, hold out the first N train documents as "
                              "validation.", default=0)
    tokenizer_train_docs: int = _help('Documents used to train a "bpe" tokenizer.',
                                      default=20_000)
    eval_interval: int = _help("Evaluate on the validation documents every N steps.",
                               default=50)
    eval_docs: int = _help("Validation documents used: the first N, the same for every run.",
                           default=200)
    sample_interval: int = _help("Generate sample text every N steps.", default=100)
    sample_length: int = _help("Tokens in each generated sample.", default=200)
    sample_prompt: str = _help("Text the samples continue; empty starts from scratch.",
                               default="")
    sample_temperature: float = _help("Sampling temperature; lower is more predictable.",
                                      default=0.8)
    checkpoint_interval: int = _help("Save a checkpoint every N steps.", default=500)
    keep_checkpoints: int = _help("How many recent checkpoints to keep.", default=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preset:
        """Rebuild a preset from its JSON form (a config.json's `preset`)."""
        return cls(**{**data, "betas": tuple(data.get("betas", cls.betas))})

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
