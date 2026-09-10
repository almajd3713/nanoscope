"""Stable document partitions shared by training and evaluation."""

from __future__ import annotations

import hashlib

from nanoscope.config import PartitionConfig


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_split(text: str, partition: PartitionConfig) -> str:
    if not isinstance(text, str):
        raise TypeError("partitioned documents must be strings")
    digest = hashlib.sha256((partition.salt + "\0" + text).encode("utf-8")).digest()
    bucket = int.from_bytes(digest, "big") % partition.buckets
    if bucket < partition.validation_buckets:
        return "validation"
    if bucket < partition.validation_buckets + partition.test_buckets:
        return "test"
    return "train"


def accepts_document(text: str, partition: PartitionConfig) -> bool:
    return document_split(text, partition) == partition.split
