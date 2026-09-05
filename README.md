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

See [the M0 runbook](docs/m0-runbook.md) for Kaggle, cloud credentials, resume,
and the model-only handoff.

