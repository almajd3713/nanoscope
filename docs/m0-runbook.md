# M0 runbook

## Local setup

Prerequisites are Git, Python 3.10–3.12, and `uv`.

```bash
uv sync --extra dev
uv run pytest -m "not gpu and not network and not cloud"
uv run nanoscope doctor --config configs/m0/local-smoke.yaml
uv run nanoscope m0-acceptance --config configs/m0/local-smoke.yaml
```

The acceptance command trains an uninterrupted control and a run restored at
three checkpoint boundaries. It compares token-batch hashes, losses, learning
rates, counters, and FLOP accounting while excluding wall-clock measurements.

Run a normal smoke test and inspect its final checkpoint:

```bash
uv run nanoscope train --config configs/m0/local-smoke.yaml --resume none
uv run nanoscope inspect-checkpoint runs/m0-local-smoke/checkpoints/step_00000012
```

`--resume auto` selects the newest locally valid checkpoint. A corrupt or
half-written directory is ignored. A particular directory can be passed in
place of `auto`.

## Kaggle acceptance

1. Create a private Hugging Face model repository for recovery checkpoints.
2. Create Kaggle secrets named `HF_TOKEN`, `HF_REPO_ID`, `WANDB_API_KEY`, and
   optionally `WANDB_ENTITY`. The HF token needs write access. Never paste a
   token into a notebook cell, config, or Git file.
3. Create a Kaggle notebook, enable Internet and one GPU, then run:

```bash
git clone https://github.com/almajd3713/nanoscope.git
cd nanoscope
python -m pip install --require-hashes -r requirements-kaggle.lock
python -m pip install -e . --no-deps
nanoscope doctor --config configs/m0/kaggle-acceptance.yaml
nanoscope train --config configs/m0/kaggle-acceptance.yaml --resume none
```

Before the first accepted run, replace the null `data.revision` with the
`resolved_data_revision` printed by `nanoscope doctor` and commit that change.
Also either set
`checkpoint.hub_repo` in a private local config or expose `HF_REPO_ID` as a
secret. `run.require_clean_repo` intentionally refuses uncommitted research
runs.

Interrupt with the notebook stop control only after at least one checkpoint has
been published. A fresh session resumes using:

```bash
nanoscope train --config configs/m0/kaggle-acceptance.yaml \
  --resume hf://OWNER/REPOSITORY/runs/m0-kaggle-acceptance
```

The strict final acceptance consists of a separate uninterrupted run and one
run interrupted three times. Compare their `metrics.jsonl` files: batch hashes
must match, and every common-step loss must differ by less than 0.1%.

## Start model work

Create a branch after the M0 acceptance commit. Your normal edit surface is:

- a module in `nanoscope/model/` containing your `torch.nn.Module`;
- its registration call and model YAML parameters;
- numerical reference and shape tests in `tests/model/`.

The forward method accepts `input_ids` shaped `[batch, time]` and returns
`LMOutput(logits=[batch, time, vocabulary], auxiliary_losses={...})`. The
trainer performs target shifting and cross-entropy. Register a parameter-group
hook when the architecture needs non-default weight decay, and a FLOP hook when
`6ND` is not an adequate estimate.

Recommended loop:

1. Test output shapes, causality, initialization, and finite gradients.
2. Assert each optimized component against a naive reference.
3. Overfit one fixed fixture batch.
4. Run a short local debug configuration and test local resume.
5. Run a short FineWeb configuration on Kaggle and test Hub resume.
6. Only then start the multi-seed baseline experiment.

Do not change the trainer or checkpoint format to implement a model. If the
model cannot fit the public contract, add an explicit model hook and test its
resume behavior rather than coupling architecture logic into the loop.
