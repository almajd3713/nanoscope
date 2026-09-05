"""Subprocess-only integration probe for rank-owned I/O and failure propagation."""

import os
import random
from pathlib import Path

import torch
from torch import nn

from nanoscope.cli import main
from nanoscope.model import LMOutput
from nanoscope.model.registry import register_model
from nanoscope.train import trainer


def record(event: str) -> None:
    directory = Path(os.environ["NANOSCOPE_TEST_AUDIT"])
    with (directory / f"rank-{os.environ['RANK']}.txt").open("a") as handle:
        handle.write(event + "\n")


class CloudLogger:
    def __init__(self, *args, **kwargs):
        record("wandb_init")
        random.random()
        torch.rand(3)

    def log(self, row):
        record("wandb_log")

    def finish(self):
        record("wandb_finish")


def upload(self, checkpoint, step, attempts=3):
    record("hub_upload")
    random.random()
    torch.rand(3)
    if os.getenv("NANOSCOPE_TEST_FAIL_UPLOAD"):
        raise RuntimeError("simulated Hub outage")


original_build_stream = trainer.build_batch_stream


def build_stream(*args, **kwargs):
    record("build_stream")
    return original_build_stream(*args, **kwargs)


trainer.WandbLogger = CloudLogger
trainer.HubCheckpointStore.upload = upload
trainer.build_batch_stream = build_stream


class ConditionalModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params["vocab_size"], params["hidden_size"])
        self.head = nn.Linear(params["hidden_size"], params["vocab_size"])
        self.optional = nn.Linear(params["hidden_size"], params["hidden_size"])

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        # Rank-dependent branch exercises unused-parameter discovery through LMOutput.
        if int(tokens[0, 0]) % 2:
            hidden = torch.tanh(self.optional(hidden))
        return LMOutput(self.head(hidden), {"penalty": hidden.square().mean() * 0.01})


register_model("conditional_test_model", ConditionalModel)

if __name__ == "__main__":
    main()
