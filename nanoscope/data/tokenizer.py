from __future__ import annotations

import hashlib
import importlib.metadata
from typing import Protocol

from nanoscope.config import TokenizerConfig


class Tokenizer(Protocol):
    eos_token_id: int

    def encode(self, text: str) -> list[int]: ...


class ByteTokenizer:
    def __init__(self, eos_token_id: int = 256) -> None:
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class GPT2Tokenizer:
    def __init__(self, eos_token_id: int = 50_256) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("GPT-2 tokenization requires the tiktoken package") from exc
        self._encoding = tiktoken.get_encoding("gpt2")
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode_ordinary(text)


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    if config.name == "byte":
        return ByteTokenizer(config.eos_token_id)
    if config.name == "gpt2":
        return GPT2Tokenizer(config.eos_token_id)
    raise ValueError(f"unknown tokenizer: {config.name}")


def tokenizer_identity(config: TokenizerConfig) -> dict[str, str | int]:
    """Identify token meanings as well as the implementation used to encode text."""
    if config.name == "byte":
        if type(config.eos_token_id) is not int or config.eos_token_id < 256:
            raise ValueError("byte tokenizer EOS must be outside the byte vocabulary (>=256)")
        return {
            "name": "byte",
            "implementation": "utf8-byte-v1",
            "eos_token_id": config.eos_token_id,
            "vocab_size": config.eos_token_id + 1,
        }
    if config.name != "gpt2":
        raise ValueError(f"unknown tokenizer: {config.name}")
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    if type(config.eos_token_id) is not int or not 0 <= config.eos_token_id < encoding.n_vocab:
        raise ValueError("GPT-2 EOS must be a valid token ID")
    digest = hashlib.sha256()
    for token in range(encoding.n_vocab):
        value = encoding.decode_single_token_bytes(token)
        digest.update(len(value).to_bytes(4, "big"))
        digest.update(value)
    return {
        "name": "gpt2",
        "implementation": "tiktoken-encode-ordinary-v1",
        "library_version": importlib.metadata.version("tiktoken"),
        "vocabulary_sha256": digest.hexdigest(),
        "vocab_size": encoding.n_vocab,
        "eos_token_id": config.eos_token_id,
    }
