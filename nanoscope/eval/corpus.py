from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nanoscope.config import DataConfig
from nanoscope.data.partitions import accepts_document, document_hash
from nanoscope.data.tokenizer import build_tokenizer, tokenizer_identity
from nanoscope.eval.artifacts import file_hash, fingerprint, read_json, write_json
from nanoscope.eval.config import CorpusConfig

PACKING = "concat-eos-disjoint-blocks-v1"


def source_identity(data: DataConfig, revision: str | None = None) -> dict[str, Any]:
    if data.source == "fixture":
        return {
            "source": "fixture",
            "revision": "fixture-v1",
            "documents_sha256": fingerprint(data.documents),
        }
    return {
        "source": data.source,
        "dataset_name": data.dataset_name,
        "dataset_config": data.dataset_config,
        "revision": revision or data.revision,
        "split": data.split,
        "text_field": data.text_field,
    }


def iter_documents(data: DataConfig) -> Generator[str, None, None]:
    """One finite, unshuffled pass; never cycle a small evaluation fixture."""
    if data.source == "fixture":
        yield from data.documents
        return
    from datasets import load_dataset

    dataset = load_dataset(
        data.dataset_name,
        name=data.dataset_config,
        split=data.split,
        revision=data.revision,
        streaming=True,
    )
    iterator = iter(dataset)
    try:
        for row in iterator:
            text = row[data.text_field]
            if not isinstance(text, str):
                raise TypeError("evaluation documents must contain text")
            yield text
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class FrozenCorpus:
    path: Path
    manifest: dict[str, Any]
    tokens: np.ndarray

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]


def prepare_corpus(config: CorpusConfig) -> FrozenCorpus:
    partition = config.data.partition
    if partition is None or partition.split not in {"validation", "test"}:
        raise ValueError("corpus requires a validation or test partition")
    identity = tokenizer_identity(config.tokenizer)
    recipe = {
        "source": source_identity(config.data),
        "partition": asdict(partition),
        "tokenizer": identity,
        "packing": PACKING,
        "sequence_length": config.data.sequence_length,
        "sequences": config.sequences,
    }
    if config.output.exists():
        existing = load_corpus(config.output)
        if existing.manifest["recipe"] != recipe:
            raise ValueError("corpus output already exists with a different recipe")
        return existing
    tokenizer = build_tokenizer(config.tokenizer)
    required = config.sequences * (config.data.sequence_length + 1)
    values: list[int] = []
    hashes: list[str] = []
    seen: set[str] = set()
    scanned = 0
    discarded = 0
    documents = iter_documents(config.data)
    try:
        for text in documents:
            scanned += 1
            if accepts_document(text, partition):
                digest = document_hash(text)
                if digest not in seen:
                    seen.add(digest)
                    hashes.append(digest)
                    encoded = tokenizer.encode(text) + [tokenizer.eos_token_id]
                    remaining = required - len(values)
                    values.extend(encoded[:remaining])
                    discarded = max(0, len(encoded) - remaining)
            if len(values) == required or scanned >= config.max_documents:
                break
    finally:
        documents.close()
    if len(values) != required:
        raise ValueError(
            f"corpus exhausted or scan limit reached: got {len(values)} of {required} tokens; "
            "provide more documents or increase max_documents"
        )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".corpus-", dir=config.output.parent))
    try:
        array = np.asarray(values, dtype="<i8").reshape(config.sequences, -1)
        np.save(temporary / "tokens.npy", array, allow_pickle=False)
        manifest = {
            "schema_version": 1,
            "recipe": recipe,
            "document_hashes": hashes,
            "documents_scanned": scanned,
            "discarded_tail_tokens": discarded,
            "tokens_sha256": file_hash(temporary / "tokens.npy"),
            "scored_tokens": config.sequences * config.data.sequence_length,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, config.output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_corpus(config.output)


def load_corpus(path: Path, expected_hash: str | None = None) -> FrozenCorpus:
    manifest = read_json(path / "manifest.json")
    digest = manifest.get("fingerprint")
    if digest != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("corpus manifest checksum mismatch")
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("corpus does not match the configured corpus_hash")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported corpus schema")
    recipe = manifest["recipe"]
    if recipe["packing"] != PACKING:
        raise ValueError("unsupported corpus packing convention")
    if file_hash(path / "tokens.npy") != manifest["tokens_sha256"]:
        raise ValueError("corpus token checksum mismatch")
    tokens = np.load(path / "tokens.npy", mmap_mode="r", allow_pickle=False)
    shape = (recipe["sequences"], recipe["sequence_length"] + 1)
    if tokens.shape != shape or tokens.dtype != np.dtype("<i8") or tokens.size == 0:
        raise ValueError("invalid corpus token shape or dtype")
    if int(tokens.min()) < 0 or int(tokens.max()) >= recipe["tokenizer"]["vocab_size"]:
        raise ValueError("corpus contains invalid token IDs")
    if manifest["scored_tokens"] != recipe["sequences"] * recipe["sequence_length"]:
        raise ValueError("corpus scored-token count does not match shape")
    return FrozenCorpus(path, manifest, tokens)
