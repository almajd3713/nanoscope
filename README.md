# nanoscope

`nanoscope` is a small-model research laboratory built around reproducible,
falsifiable experiments. Milestone zero provides deterministic data packing,
training, metrics, and crash-safe checkpoint/resume so later work can stay
focused on model architecture.

## Start locally

```bash
uv sync --extra dev
uv run pytest -m "not gpu and not network and not cloud"
uv run nanoscope doctor --config configs/m0/local-smoke.yaml
uv run nanoscope train --config configs/m0/local-smoke.yaml --resume none
```

Or you may use the Makefile provided:

```bash
make help
```

See [the M0 runbook](docs/m0-runbook.md) for Kaggle, cloud credentials, resume,
and the model-only handoff.

## Start an experiment

Copy the [commented experiment template](configs/experiment-template.yaml):

```bash
cp configs/experiment-template.yaml configs/my-model-seed-1337.yaml
```

Set a unique `run.id`, replace `model.name` and `model.params` with your registered
model, and choose your data and training budget. The template includes working
toy-model defaults, global batch calculations, DDP settings, HF/W&B configuration,
and local/Kaggle launch and resume instructions. For a local run, change
`run.output_dir` to `runs`. Commit the edited config before starting a research run.

Upload from your local terminal using the Kaggle CLI helper:

```bash
python3 kaggle_sync.py --dry-run
python3 kaggle_sync.py -m "Update model experiment"
```

The target dataset is configured in [kaggle-sync.json](kaggle-sync.json).
See [workspace sync](docs/kaggle-workspace.md) for authentication and ignore behavior.
On Kaggle, update the permanent runner's attached dataset to the new version,
then extract the workspace and run:

```python
!python kaggle_run.py --config configs/my-model-seed-1337.yaml --resume none
```

See [the distributed runbook](docs/distributed-training.md) for resume, model
constraints, CPU testing, and GPU validation for multi-worker training.
