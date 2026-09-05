# ADR-001: Reproducibility-first M0 boundary

## Status

Accepted

## Context

Nanoscope trains on free notebook GPUs whose sessions can end before a long run
finishes. Later scaling and interpretability results are only credible if the
next batch and update after resume are the same ones an uninterrupted run would
have performed.

## Decision

- M0 is single-process and single-GPU.
- The data path is a stateful document source plus a stateful token packer with
  no worker prefetching.
- Canonical metrics are append-only JSONL; W&B is a secondary presentation sink.
- Recovery checkpoints are atomic locally and can be committed to Hugging Face
  Hub. They contain every state capable of changing the next update.
- Real model code enters through a small registry and `LMOutput` contract. M0's
  toy LM is a fixture, not a research baseline.

## Trade-offs

- Tokenization may become CPU-bound before model compute does. Measure this
  before adding workers; any faster loader must retain the state contract.
- Two-GPU DDP is deferred. A second T4 is better used for an independent seed
  until distributed checkpoint and sampler state are justified.
- Strict deterministic CUDA kernels can be slower. Research runs may introduce
  a faster profile only after demonstrating that its numerical variance is
  below the experiment's error budget.

## Revisit triggers

Revisit the loader when measured GPU idle time exceeds 10%, and revisit DDP when
one model cannot meet the planned wall-clock budget on a single accelerator.

