from __future__ import annotations

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

