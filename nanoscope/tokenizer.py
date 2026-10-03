"""Tokenizers: bytes, GPT-2, and a BPE trained on the preset's own data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

EOS = "<|endoftext|>"


class Tokenizer(Protocol):
    vocab_size: int
    eos_token_id: int
    def encode(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...
    def decode(self, tokens: list[int]) -> str: ...


class ByteTokenizer:
    """One token per UTF-8 byte, plus an end-of-text token. Nothing to train."""

    vocab_size = 257
    eos_token_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, tokens: list[int]) -> str:
        return bytes(t for t in tokens if t < 256).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    def __init__(self) -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab
        self.eos_token_id = self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._enc.encode_ordinary_batch(texts)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode([t for t in tokens if t != self.eos_token_id])


class BPETokenizer:
    """Byte-level BPE (the GPT-2 scheme) with a vocabulary learned from your data."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer as HFTokenizer
        self._tok = HFTokenizer.from_file(str(path))
        self.vocab_size = self._tok.get_vocab_size()
        self.eos_token_id = self._tok.token_to_id(EOS)

    @staticmethod
    def train(texts: Iterable[str], vocab_size: int, path: Path) -> BPETokenizer:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers

        tok = HFTokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOS],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer)
        tmp = path.with_suffix(".tmp")
        tok.save(str(tmp))
        tmp.replace(path)
        return BPETokenizer(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode([t for t in tokens if t != self.eos_token_id])
