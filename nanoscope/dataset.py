"""Turn a text dataset into flat token files, once, and cache them.

Layout under ~/.nanoscope/data (or $NANOSCOPE_DATA_DIR):

    <dataset>/<tokenizer-id>/tokenizer.json     only for "bpe", trained on the train split
    <dataset>/<tokenizer-id>/train-<docs>.bin   uint16 token ids, documents separated by EOS
    <dataset>/<tokenizer-id>/val-<docs>.bin     the first N validation documents
    <dataset>/<tokenizer-id>/*.json             token and byte counts for each .bin
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from nanoscope.presets import Preset
from nanoscope.tokenizer import BPETokenizer, ByteTokenizer, GPT2Tokenizer, Tokenizer

CACHE_DIR = Path(os.environ.get("NANOSCOPE_DATA_DIR", Path.home() / ".nanoscope" / "data"))
_CHUNK_DOCS = 1024


def _log(msg: str) -> None:
    print(f"[nanoscope] {msg}", flush=True)


def _iter_texts(dataset: str, split: str, max_docs: int | None) -> Iterator[str]:
    from datasets import load_dataset

    if max_docs is None:
        rows = load_dataset(dataset, split=split)
    else:
        # Streaming fetches only the documents we use, not the whole dataset.
        rows = load_dataset(dataset, split=split, streaming=True).take(max_docs)
    for row in rows:
        yield row["text"]


def tokenizer_id(preset: Preset) -> str:
    if preset.tokenizer == "bpe":
        if preset.vocab_size is None:
            raise ValueError("tokenizer='bpe' needs a vocab_size")
        return f"bpe-{preset.vocab_size}-{preset.tokenizer_train_docs}"
    if preset.tokenizer in ("gpt2", "bytes"):
        return preset.tokenizer
    raise ValueError(f"unknown tokenizer {preset.tokenizer!r}; use 'bpe', 'gpt2' or 'bytes'")


def _tokenizer_dir(preset: Preset) -> Path:
    return CACHE_DIR / preset.dataset.replace("/", "_") / tokenizer_id(preset)


def load_tokenizer(preset: Preset) -> Tokenizer:
    if preset.tokenizer == "bytes":
        return ByteTokenizer()
    if preset.tokenizer == "gpt2":
        return GPT2Tokenizer()
    tokenizer_id(preset)  # validates
    path = _tokenizer_dir(preset) / "tokenizer.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _log(f"training a {preset.vocab_size}-token BPE tokenizer on "
             f"{preset.tokenizer_train_docs:,} {preset.dataset} documents (once)")
        texts = _iter_texts(preset.dataset, "train", preset.tokenizer_train_docs)
        return BPETokenizer.train(texts, preset.vocab_size, path)
    return BPETokenizer(path)


def _write_tokens(
    texts: Iterator[str], tokenizer: Tokenizer, out: Path, label: str, total: int | None,
) -> None:
    from tqdm.auto import tqdm

    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} does not fit in uint16 token files")
    n_docs = n_tokens = n_bytes = 0
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f, tqdm(total=total, desc=f"tokenizing {label}", unit="doc") as bar:
        while chunk := list(islice(texts, _CHUNK_DOCS)):
            ids = tokenizer.encode_batch(chunk)
            eos = [tokenizer.eos_token_id]
            flat = np.concatenate([np.asarray(doc + eos, dtype=np.uint16) for doc in ids])
            flat.tofile(f)
            n_docs += len(chunk)
            n_tokens += len(flat)
            n_bytes += sum(len(t.encode("utf-8")) for t in chunk)
            bar.update(len(chunk))
    tmp.replace(out)
    meta = {"docs": n_docs, "tokens": n_tokens, "bytes": n_bytes}
    out.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def _token_file(preset: Preset, tokenizer: Tokenizer, split: str, docs: int | None) -> Path:
    name = "train" if split == "train" else "val"
    path = _tokenizer_dir(preset) / f"{name}-{docs if docs is not None else 'all'}.bin"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        texts = _iter_texts(preset.dataset, split, docs)
        _write_tokens(texts, tokenizer, path, f"{preset.dataset} {split}", docs)
    return path


class TokenDataset:
    def __init__(self, path: Path, context_length: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        if len(self) <= 0:
            raise ValueError(f"{path} has fewer tokens than context_length={context_length}")

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def get_batch(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        ix = torch.randint(len(self), (batch_size,), generator=generator).tolist()
        seq_len = self.context_length + 1
        buf = np.stack([self.data[i : i + seq_len] for i in ix]).astype(np.int64)
        return torch.from_numpy(buf)


@dataclass
class Data:
    train: TokenDataset
    val: np.ndarray
    val_bytes: int
    tokenizer: Tokenizer

    def bits_per_byte(self, mean_loss: float) -> float:
        """Nats per token -> bits per byte of text, which is comparable across tokenizers."""
        return mean_loss / math.log(2) * len(self.val) / self.val_bytes


def load_data(preset: Preset) -> Data:
    tokenizer = load_tokenizer(preset)
    train_path = _token_file(preset, tokenizer, "train", preset.train_docs)
    val_path = _token_file(preset, tokenizer, "validation", preset.eval_docs)
    val_meta = json.loads(val_path.with_suffix(".json").read_text(encoding="utf-8"))
    return Data(
        train=TokenDataset(train_path, preset.context_length),
        val=np.memmap(val_path, dtype=np.uint16, mode="r"),
        val_bytes=val_meta["bytes"],
        tokenizer=tokenizer,
    )
