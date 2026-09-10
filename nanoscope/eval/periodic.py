"""Primary-rank evaluation scheduling and checkpoint-backed result history."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nanoscope.config import Config
from nanoscope.data.tokenizer import tokenizer_identity
from nanoscope.eval.artifacts import fingerprint, write_json, write_text_atomic
from nanoscope.eval.config import EvalConfig
from nanoscope.eval.corpus import load_corpus, source_identity
from nanoscope.eval.runner import evaluate_checkpoint, evaluator_code_hash, load_result


class PeriodicEvaluation:
    def __init__(
        self,
        run_dir: Path,
        training: Config,
        config: EvalConfig | None,
        device: torch.device,
        source_revision: str | None,
    ) -> None:
        self.run_dir = run_dir
        self.training = training
        self.config = config
        self.corpus = None
        self.history: list[dict[str, Any]] = []
        self.config_history: list[dict[str, Any]] = []
        self.directory = run_dir / "evaluations"
        if config is None:
            return
        if config.max_seconds >= 540:
            raise ValueError("periodic max_seconds must be below 540 (DDP timeout headroom)")
        if config.device not in {"auto", device.type}:
            raise ValueError("periodic evaluation must use the training device")
        self.config = replace(config, device=device.type)
        self.corpus = load_corpus(config.corpus, config.corpus_hash)
        recipe = self.corpus.manifest["recipe"]
        partition = training.data.partition
        scheme = {key: value for key, value in recipe["partition"].items() if key != "split"}
        if recipe["partition"]["split"] != "validation":
            raise ValueError("periodic evaluation requires validation data, never test data")
        if (
            partition is None
            or partition.scheme() != scheme
            or source_identity(training.data, source_revision) != recipe["source"]
        ):
            raise ValueError("periodic validation requires matching training exclusion provenance")
        if (
            training.data.sequence_length != recipe["sequence_length"]
            or tokenizer_identity(training.tokenizer) != recipe["tokenizer"]
        ):
            raise ValueError("periodic validation context/tokenizer differs from training")
        if device.type == "cpu" and config.precision != "fp32":
            raise ValueError("CPU periodic evaluation requires fp32")

    def restore(self, value: dict[str, Any], step: int) -> None:
        self.history = list(value.get("results", []))
        self.config_history = list(value.get("config_history", []))
        for result in self.history:
            if result["run"]["id"] != self.training.run.id or result["checkpoint"]["step"] > step:
                raise ValueError("checkpoint evaluation history belongs to a different run/step")
        descriptor = self.descriptor()
        if not self.config_history or self.config_history[-1]["config"] != descriptor:
            self.config_history.append({"after_step": step, "config": descriptor})
        self.publish(reconcile=True)

    def descriptor(self) -> dict[str, Any] | None:
        if self.config is None or self.corpus is None:
            return None
        return {
            "corpus_hash": self.corpus.fingerprint,
            "batch_size": self.config.batch_size,
            "precision": self.config.precision,
            "device": self.config.device,
            "every_steps": self.config.every_steps,
            "final": self.config.final,
            "max_seconds": self.config.max_seconds,
            "evaluator_code_hash": evaluator_code_hash(),
        }

    def completed(self, step: int) -> bool:
        descriptor = self.descriptor()
        if descriptor is None:
            return False
        return any(
            result["checkpoint"]["step"] == step
            and all(
                result["identity"][key] == descriptor[key]
                for key in ("corpus_hash", "batch_size", "precision", "evaluator_code_hash")
            )
            for result in self.history
        )

    def evaluate(
        self,
        checkpoint: Path,
        model: nn.Module,
        snapshot: dict[str, Any],
        cancelled: Callable[[], bool],
    ) -> dict[str, Any]:
        assert self.config is not None and self.corpus is not None
        path = evaluate_checkpoint(
            checkpoint,
            self.config,
            model=model,
            snapshot=snapshot,
            corpus=self.corpus,
            output_dir=self.directory,
            cancelled=cancelled,
        )
        result = load_result(path)
        if not any(row["result_id"] == result["result_id"] for row in self.history):
            self.history.append(result)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "results": list(self.history),
            "config_history": list(self.config_history),
        }

    def publish(self, *, reconcile: bool = False) -> None:
        expected = {f"{result['result_id']}.json" for result in self.history}
        if reconcile and self.directory.exists():
            for path in self.directory.glob("*.json"):
                if path.name not in expected:
                    orphan = self.directory / "orphaned"
                    orphan.mkdir(exist_ok=True)
                    os.replace(path, orphan / path.name)
        for result in self.history:
            # Validate history restored from the checkpoint before exporting a score.
            if result["result_hash"] != fingerprint(
                {key: value for key, value in result.items() if key != "result_hash"}
            ):
                raise ValueError("checkpoint evaluation result checksum mismatch")
            path = self.directory / f"{result['result_id']}.json"
            write_json(path, result)
        write_text_atomic(
            self.run_dir / "evaluations.jsonl",
            "".join(
                json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
                for result in self.history
            ),
        )
