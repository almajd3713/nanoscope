from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, Protocol

import torch

from nanoscope.config import DataConfig, PartitionConfig, TokenizerConfig
from nanoscope.data.partitions import accepts_document
from nanoscope.data.tokenizer import Tokenizer, build_tokenizer


class StatefulDocumentSource(Protocol):
    @property
    def revision(self) -> str | None: ...

    def next_document(self) -> str: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class FixtureDocumentSource:
    revision = "fixture-v1"

    def __init__(
        self, documents: list[str], seed: int, partition: PartitionConfig | None = None
    ) -> None:
        self._partition = partition
        self._documents = [
            text for text in documents if partition is None or accepts_document(text, partition)
        ]
        if not self._documents:
            raise ValueError("no fixture documents remain in the requested partition")
        self._seed = seed
        self._epoch = 0
        self._position = 0
        self._order = self._make_order()

    def _make_order(self) -> list[int]:
        order = list(range(len(self._documents)))
        random.Random(self._seed + self._epoch).shuffle(order)
        return order

    def next_document(self) -> str:
        if self._position == len(self._order):
            self._epoch += 1
            self._position = 0
            self._order = self._make_order()
        index = self._order[self._position]
        self._position += 1
        return self._documents[index]

    def state_dict(self) -> dict[str, Any]:
        state = {"epoch": self._epoch, "position": self._position, "order": self._order}
        if self._partition is not None:
            state["partition"] = asdict(self._partition)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = asdict(self._partition) if self._partition is not None else None
        if state.get("partition") != expected:
            raise ValueError("checkpoint document partition differs from configuration")
        self._epoch = int(state["epoch"])
        self._position = int(state["position"])
        self._order = [int(item) for item in state["order"]]

    def close(self) -> None:
        return None


class FineWebDocumentSource:
    def __init__(self, config: DataConfig, seed: int) -> None:
        try:
            from datasets import load_dataset
            from huggingface_hub import dataset_info
        except ImportError as exc:
            raise RuntimeError("FineWeb streaming requires datasets and huggingface-hub") from exc

        self.revision = config.revision or dataset_info(config.dataset_name).sha
        self._text_field = config.text_field
        self._partition = config.partition
        self._shuffle_size = config.shuffle_buffer
        self._shuffle_rng = random.Random(seed)
        self._shuffle_buffer: list[str] = []
        self._exhausted = False
        self._dataset = load_dataset(
            config.dataset_name,
            name=config.dataset_config,
            split=config.split,
            streaming=True,
            revision=self.revision,
        )
        if config.partition is None:
            # Preserve the legacy stream and its checkpoint format.
            self._dataset = self._dataset.shuffle(seed=seed, buffer_size=config.shuffle_buffer)
        self._iterator: Iterator[dict[str, Any]] | None = None

    def next_document(self) -> str:
        if self._partition is not None:
            return self._next_partitioned_document()
        try:
            return self._read_text()
        except StopIteration:
            raise RuntimeError("FineWeb stream was exhausted unexpectedly") from None

    def _read_text(self) -> str:
        if self._iterator is None:
            self._iterator = iter(self._dataset)
        row = next(self._iterator)
        value = row[self._text_field]
        if not isinstance(value, str):
            raise TypeError(f"FineWeb field {self._text_field!r} was not text")
        return value

    def _read_partitioned_text(self) -> str:
        assert self._partition is not None
        while True:
            text = self._read_text()
            if accepts_document(text, self._partition):
                return text

    def _next_partitioned_document(self) -> str:
        # Own the filter/shuffle buffer: a datasets cursor alone does not retain
        # all prefetched shuffled documents. Persist both documents and RNG.
        while not self._exhausted and len(self._shuffle_buffer) < self._shuffle_size:
            try:
                self._shuffle_buffer.append(self._read_partitioned_text())
            except StopIteration:
                self._exhausted = True
        if not self._shuffle_buffer:
            raise RuntimeError("FineWeb training partition was exhausted unexpectedly")
        index = self._shuffle_rng.randrange(len(self._shuffle_buffer))
        text = self._shuffle_buffer[index]
        if not self._exhausted:
            try:
                self._shuffle_buffer[index] = self._read_partitioned_text()
                return text
            except StopIteration:
                self._exhausted = True
        self._shuffle_buffer.pop(index)
        return text

    def state_dict(self) -> dict[str, Any]:
        state = copy.deepcopy(self._dataset.state_dict())
        if self._partition is not None:
            return {
                "partition": asdict(self._partition),
                "dataset": state,
                "shuffle_version": 1,
                "shuffle_size": self._shuffle_size,
                "shuffle_buffer": list(self._shuffle_buffer),
                "shuffle_rng": self._shuffle_rng.getstate(),
                "exhausted": self._exhausted,
            }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = asdict(self._partition) if self._partition is not None else None
        if state.get("partition") != expected:
            raise ValueError("checkpoint document partition differs from configuration")
        if self._partition is not None:
            if state.get("shuffle_version") != 1 or state.get("shuffle_size") != self._shuffle_size:
                raise ValueError("checkpoint partition shuffle configuration differs")
            self._shuffle_buffer = list(state["shuffle_buffer"])
            self._shuffle_rng.setstate(state["shuffle_rng"])
            self._exhausted = state["exhausted"]
            state = state["dataset"]
        self._dataset.load_state_dict(state)
        self._iterator = None

    def close(self) -> None:
        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if close is not None:
                close()
            self._iterator = None


class StatefulBatchStream:
    """Tokenize, concatenate, and pack documents without prefetching."""

    STATE_VERSION = 1

    def __init__(
        self,
        source: StatefulDocumentSource,
        tokenizer: Tokenizer,
        sequence_length: int,
        batch_size: int,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self._buffer: list[int] = []
        self.documents_seen = 0
        self.tokens_emitted = 0
        self.batches_emitted = 0

    @property
    def source_revision(self) -> str | None:
        return self.source.revision

    def next_batch(self) -> torch.Tensor:
        block_size = self.sequence_length + 1
        required = self.batch_size * block_size
        while len(self._buffer) < required:
            document = self.source.next_document()
            self._buffer.extend(self.tokenizer.encode(document))
            self._buffer.append(self.tokenizer.eos_token_id)
            self.documents_seen += 1

        values = self._buffer[:required]
        del self._buffer[:required]
        self.tokens_emitted += required
        self.batches_emitted += 1
        return torch.tensor(values, dtype=torch.long).view(self.batch_size, block_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "source": self.source.state_dict(),
            "buffer": list(self._buffer),
            "documents_seen": self.documents_seen,
            "tokens_emitted": self.tokens_emitted,
            "batches_emitted": self.batches_emitted,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["version"]) != self.STATE_VERSION:
            raise ValueError("unsupported batch-stream state version")
        if int(state["sequence_length"]) != self.sequence_length:
            raise ValueError("checkpoint sequence length does not match configuration")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("checkpoint batch size does not match configuration")
        self.source.load_state_dict(state["source"])
        self._buffer = [int(item) for item in state["buffer"]]
        self.documents_seen = int(state["documents_seen"])
        self.tokens_emitted = int(state["tokens_emitted"])
        self.batches_emitted = int(state["batches_emitted"])

    def close(self) -> None:
        self.source.close()


def build_batch_stream(
    data_config: DataConfig,
    tokenizer_config: TokenizerConfig,
    seed: int,
    batch_size: int,
) -> StatefulBatchStream:
    tokenizer = build_tokenizer(tokenizer_config)
    if data_config.source == "fixture":
        source: StatefulDocumentSource = FixtureDocumentSource(
            data_config.documents, seed, data_config.partition
        )
    else:
        source = FineWebDocumentSource(data_config, seed)
    return StatefulBatchStream(source, tokenizer, data_config.sequence_length, batch_size)
