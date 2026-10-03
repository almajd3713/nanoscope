"""Turn a text dataset into flat token files, once, and cache them.

Layout under ~/.nanoscope/data (or $NANOSCOPE_DATA_DIR). Presets with Preset.hub_data
download the same layout from that Hub dataset repo instead of tokenizing:

    <dataset>/<tokenizer-id>/tokenizer.json          only for "bpe", trained on the train split
    <dataset>/<tokenizer-id>/train-<docs>/shard-*.bin uint16 token ids, documents ended by EOS
    <dataset>/<tokenizer-id>/train-<docs>/meta.json  token, byte and shard counts
    <dataset>/<tokenizer-id>/val-<docs>/...          the first N held-out documents
"""

from __future__ import annotations

import fnmatch
import functools
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
SHARD_TOKENS = 100_000_000
_CHUNK_DOCS = 1024


def _log(msg: str) -> None:
    print(f"[nanoscope] {msg}", flush=True)


def _iter_texts(
    dataset: str, config: str | None, split: str, max_docs: int | None,
) -> Iterator[str]:
    from datasets import load_dataset

    if max_docs is None:
        rows = load_dataset(dataset, config, split=split)
    else:
        # Streaming fetches only the documents we use, not the whole dataset.
        rows = load_dataset(dataset, config, split=split, streaming=True).take(max_docs)
    for row in rows:
        yield row["text"]


def _split_texts(preset: Preset, split: str, docs: int | None) -> Iterator[str]:
    """Documents of a split. With holdout_docs, validation is the start of the train split
    and training skips it, for datasets that ship without a validation split."""
    h = preset.holdout_docs
    if not h:
        return _iter_texts(preset.dataset, preset.dataset_config, split, docs)
    if split == "validation":
        if docs is not None and docs > h:
            raise ValueError(f"eval_docs={docs} exceeds holdout_docs={h}")
        return _iter_texts(preset.dataset, preset.dataset_config, "train", docs or h)
    total = None if docs is None else h + docs
    return islice(_iter_texts(preset.dataset, preset.dataset_config, "train", total), h, None)


def tokenizer_id(preset: Preset) -> str:
    if preset.tokenizer == "bpe":
        if preset.vocab_size is None:
            raise ValueError("tokenizer='bpe' needs a vocab_size")
        return f"bpe-{preset.vocab_size}-{preset.tokenizer_train_docs}"
    if preset.tokenizer in ("gpt2", "bytes"):
        return preset.tokenizer
    raise ValueError(f"unknown tokenizer {preset.tokenizer!r}; use 'bpe', 'gpt2' or 'bytes'")


def _dataset_dir(preset: Preset) -> Path:
    name = preset.dataset.replace("/", "_")
    if preset.dataset_config:
        name += f"_{preset.dataset_config}"
    if preset.holdout_docs:  # changes which documents train, so it's part of the identity
        name += f"_holdout-{preset.holdout_docs}"
    return CACHE_DIR / name


def _tokenizer_dir(preset: Preset) -> Path:
    return _dataset_dir(preset) / tokenizer_id(preset)


@functools.cache
def _hub_files(repo_id: str) -> frozenset[str]:
    """Files in a Hub dataset repo; empty when it's missing, private or we're offline."""
    from huggingface_hub import HfApi

    try:
        return frozenset(HfApi().list_repo_files(repo_id, repo_type="dataset"))
    except Exception as exc:
        _log(f"can't reach hf.co/datasets/{repo_id} ({type(exc).__name__}); "
             "preparing data locally instead")
        return frozenset()


def _from_hub(preset: Preset, pattern: str) -> bool:
    """Download files matching pattern (relative to the tokenizer dir) from preset.hub_data."""
    if not preset.hub_data:
        return False
    remote = f"{_tokenizer_dir(preset).relative_to(CACHE_DIR).as_posix()}/{pattern}"
    if not any(fnmatch.fnmatch(f, remote) for f in _hub_files(preset.hub_data)):
        return False
    from huggingface_hub import snapshot_download

    _log(f"downloading {remote} from hf.co/datasets/{preset.hub_data}")
    snapshot_download(
        preset.hub_data, repo_type="dataset", local_dir=CACHE_DIR, allow_patterns=[remote],
    )
    return True


def load_tokenizer(preset: Preset) -> Tokenizer:
    if preset.tokenizer == "bytes":
        return ByteTokenizer()
    if preset.tokenizer == "gpt2":
        return GPT2Tokenizer()
    tokenizer_id(preset)  # validates
    path = _tokenizer_dir(preset) / "tokenizer.json"
    if not path.exists():
        _from_hub(preset, "tokenizer.json")
    if not path.exists():
        from tqdm.auto import tqdm

        path.parent.mkdir(parents=True, exist_ok=True)
        _log(f"training a {preset.vocab_size}-token BPE tokenizer on "
             f"{preset.tokenizer_train_docs:,} {preset.dataset} documents (once)")
        texts = tqdm(_split_texts(preset, "train", preset.tokenizer_train_docs),
                     total=preset.tokenizer_train_docs, desc="reading tokenizer documents",
                     unit="doc")
        tokenizer = BPETokenizer.train(texts, preset.vocab_size, path)
        _log(f"tokenizer ready: {tokenizer.vocab_size:,} tokens")
        return tokenizer
    return BPETokenizer(path)


def _write_tokens(
    texts: Iterator[str], tokenizer: Tokenizer, out: Path, label: str, total: int | None,
) -> None:
    """Tokenize documents into out/shard-*.bin, then write out/meta.json last, atomically."""
    from tqdm.auto import tqdm

    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} does not fit in uint16 token files")
    out.mkdir(parents=True, exist_ok=True)
    n_docs = n_tokens = n_bytes = 0
    shards: list[str] = []
    f = None
    in_shard = SHARD_TOKENS
    eos = [tokenizer.eos_token_id]
    try:
        with tqdm(total=total, desc=f"tokenizing {label}", unit="doc") as bar:
            while chunk := list(islice(texts, _CHUNK_DOCS)):
                flat = np.concatenate([np.asarray(d + eos, dtype=np.uint16)
                                       for d in tokenizer.encode_batch(chunk)])
                if in_shard >= SHARD_TOKENS:  # documents never straddle two shards
                    if f:
                        f.close()
                    shards.append(f"shard-{len(shards):05d}.bin")
                    f = (out / shards[-1]).open("wb")
                    in_shard = 0
                flat.tofile(f)
                in_shard += len(flat)
                n_docs += len(chunk)
                n_tokens += len(flat)
                n_bytes += sum(len(t.encode("utf-8")) for t in chunk)
                bar.update(len(chunk))
                bar.set_postfix(tokens=f"{n_tokens / 1e6:.1f}M", refresh=False)
    finally:
        if f:
            f.close()
    meta = {"docs": n_docs, "tokens": n_tokens, "bytes": n_bytes, "shards": shards}
    tmp = out / "meta.json.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(out / "meta.json")


def _token_dir(preset: Preset, tokenizer: Tokenizer, split: str, docs: int | None) -> Path:
    name = f"{'train' if split == 'train' else 'val'}-{docs if docs is not None else 'all'}"
    path = _tokenizer_dir(preset) / name
    if not (path / "meta.json").exists():
        _from_hub(preset, f"{name}/*")
    if not (path / "meta.json").exists():
        _write_tokens(_split_texts(preset, split, docs), tokenizer, path,
                      f"{preset.dataset} {split}", docs)
    return path


def _shards(path: Path) -> tuple[list[np.ndarray], dict]:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return [np.memmap(path / s, dtype=np.uint16, mode="r") for s in meta["shards"]], meta


class TokenDataset:
    """Random windows of context_length + 1 tokens, drawn uniformly across shards."""

    def __init__(self, path: Path, context_length: int) -> None:
        self.shards, self.meta = _shards(path)
        self.context_length = context_length
        windows = [max(0, len(s) - context_length) for s in self.shards]
        self.offsets = np.cumsum([0, *windows])
        if len(self) <= 0:
            raise ValueError(f"{path} has fewer tokens than context_length={context_length}")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def get_batch(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        ix = torch.randint(len(self), (batch_size,), generator=generator).tolist()
        seq_len = self.context_length + 1
        rows = []
        for i in ix:
            k = int(np.searchsorted(self.offsets, i, side="right")) - 1
            j = i - int(self.offsets[k])
            rows.append(self.shards[k][j : j + seq_len])
        return torch.from_numpy(np.stack(rows).astype(np.int64))


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
    train_dir = _token_dir(preset, tokenizer, "train", preset.train_docs)
    val_dir = _token_dir(preset, tokenizer, "validation", preset.eval_docs)
    val_shards, val_meta = _shards(val_dir)
    return Data(
        train=TokenDataset(train_dir, preset.context_length),
        val=np.concatenate(val_shards),
        val_bytes=val_meta["bytes"],
        tokenizer=tokenizer,
    )


def publish_data(preset: Preset, repo_id: str, private: bool = False) -> str:
    """Prepare a preset's tokens locally and upload them, so others download instead."""
    from huggingface_hub import HfApi

    load_data(preset)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    tok = _tokenizer_dir(preset)
    folders = [f"train-{preset.train_docs or 'all'}", f"val-{preset.eval_docs}"]
    patterns = [f"{d}/*" for d in folders] + ["tokenizer.json"]
    _log(f"uploading {tok} ({', '.join(folders)}) to hf.co/datasets/{repo_id}")
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(tok),
        path_in_repo=tok.relative_to(CACHE_DIR).as_posix(), allow_patterns=patterns,
        commit_message=f"nanoscope tokens for preset {preset.name}",
    )
    return f"https://huggingface.co/datasets/{repo_id}"
