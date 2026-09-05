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

## Multiple GPUs

The harness supports single-node DistributedDataParallel (DDP). Add this optional
section to an experiment config; model code keeps the same `nn.Module`/`LMOutput`
interface:

```yaml
distributed:
  strategy: ddp
  devices: auto
```

Use the usual `nanoscope train --config ...` command. It launches one worker per
visible GPU; `devices: 2` explicitly requires two. `train.batch_size` is the
**global microbatch**, divisible by the worker count. The effective optimizer
batch is `batch_size * gradient_accumulation_steps`, regardless of GPU count.
Omitting `distributed` preserves single-process behavior.

On Kaggle select **T4 x2**, upload the updated workspace dataset, update the
permanent runner's attached dataset version, and run:

```python
!python kaggle_run.py --config configs/m0/kaggle-ddp.yaml --resume none
```

See [the distributed runbook](docs/distributed-training.md) for resume, model
constraints, CPU testing, and GPU validation.
