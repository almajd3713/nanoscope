# Distributed training

## Running

The CLI automatically uses `python -m torch.distributed.run` (torchrun) for more
than one worker. The Kaggle launcher already calls this CLI, so dependency
installation and secret loading happen once before workers start.

```yaml
distributed:
  strategy: ddp
  devices: auto
  find_unused_parameters: false
```

`auto` uses all visible CUDA devices, or one CPU worker when CUDA is not selected
or available. Set `devices: 1` to use one GPU with this profile. Set `devices: 2`
to require two GPUs. CPU DDP is also supported with `train.device: cpu`,
`train.precision: fp32`, and `devices: 2` for local integration tests.

```bash
nanoscope doctor --config configs/m0/kaggle-ddp.yaml
nanoscope train --config configs/m0/kaggle-ddp.yaml --resume none
```

On Kaggle, select **T4 x2**, enable Internet and the notebook's HF/W&B secrets,
upload the current workspace, and update its attached dataset version. In the
permanent runner, after workspace extraction:

```python
!python kaggle_run.py --config configs/m0/kaggle-ddp.yaml --resume none
```

Doctor executes a small tensor operation and matrix multiplication on each
selected GPU. `cuda_available: true` alone does not prove that the installed
PyTorch build can execute on that GPU (notably P100 with cu128 builds).
The report includes `cuda_execution_ok`, `training_workers`, and batch sizes.

The supplied DDP profile uses a separate run ID, `m0-kaggle-ddp`. Use a new run ID
for each new experiment. `--resume none` refuses to reuse a run with metrics or
a checkpoint pointer to prevent silently reusing an old checkpoint at the same
step. Existing single-process configs remain single-process.

## Data and batches

`train.batch_size` is the global microbatch. It must divide evenly across the
workers. With batch size 4, two GPUs, and accumulation 8, each GPU sees two
sequences per microstep; one optimizer step processes 32 sequences in total.
Sequence length times 32 gives the trained token count for that optimizer step.

Rank 0 owns the tokenizing/packing stream, splits each global batch into disjoint
contiguous parts, and scatters tensors to workers. This prevents duplicated
examples without depending on rank-specific FineWeb shuffle state. The checkpoint
contains the existing global stream cursor/buffer, and batch hashes retain their
single-process meaning. Data loading is centralized and can become a throughput
bottleneck at larger scales; this implementation targets a single machine such
as Kaggle T4 x2.

DDP averages gradients, with synchronization delayed until the final accumulated
microstep. Loss is averaged over workers. Token/FLOP counts and throughput are
global; GPU memory metrics report the maximum across workers. `peak_tflops`, if
provided, is per GPU. Only rank 0 writes local metrics, opens W&B, and saves or
uploads checkpoints. A rank-0 I/O error is reported to all workers.

## Checkpoints and stopping

Resume inside the same session with `--resume auto`. From a fresh Kaggle session:

```python
!python kaggle_run.py --config configs/m0/kaggle-ddp.yaml \
    --resume hf://OWNER/REPOSITORY/runs/m0-kaggle-ddp
```

The Hub download happens once, followed by loading on each worker from the shared
filesystem. Exact resume requires the same worker count, compatible config,
dataset revision, and deterministic runtime. Each rank's Python, NumPy, CPU, and
selected-device CUDA RNG state is saved. Changing topology is explicitly rejected;
it changes dropout streams and floating-point reduction order. The model state
has ordinary parameter names without a `module.` prefix and can be loaded into
an ordinary CPU/single-GPU model for inference. This is weight portability, not a
promise of exact optimizer resume across different GPU counts or hardware.

`--stop-after-step N` checkpoints at an optimizer boundary and exits. SIGINT or
SIGTERM sent to the `nanoscope train` launcher requests the same coordinated stop.
Signals received by an individual worker are also coordinated at the next step
boundary. Forced process termination, killing torchrun directly, or Kaggle
session termination may prevent a final save; recover from the latest completed
Hub checkpoint. Workers are not automatically restarted after an error.

## Model contract

Write an ordinary `nn.Module` whose forward returns `LMOutput`. The harness wraps
it in [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
Keep tensors on the input's device, register parameters and buffers, and avoid
device literals or `.cuda()` inside layers. Parameter registration must finish
before training begins and be identical across workers. Auxiliary losses should
be scalar local means with the same normalization convention as cross-entropy.
For conditional unused parameters, enable `find_unused_parameters: true` and
test that model's distributed forward/backward and resume behavior.

DDP keeps a full model and optimizer replica on each GPU. Two 16 GB GPUs do not
become a single 32 GB allocation. Models requiring sharded parameters/optimizers,
tensor parallelism, expert parallelism, or multi-node execution need additional
harness support. Stateful/stochastic custom layers still need their own resume
tests; the harness cannot capture arbitrary unregistered Python state.

## Verification

```bash
uv run pytest tests/test_distributed.py -m "not gpu"
uv run pytest tests/test_distributed.py -m gpu
```

The CPU suite runs real two-process Gloo training: global-batch equivalence,
dropout/accumulation resume equality, repeated SIGTERM recovery, rank-owned cloud
operations, failure propagation, and topology validation. The GPU tests exercise
FP16 scaling and resumed state equality with one GPU and, when two GPUs are
available, NCCL DDP as well.
The CPU tests do not establish NCCL behavior or T4 throughput. Run the GPU test
from a full checkout (the workspace `.kaggleignore` excludes tests), and complete
a live HF/W&B run on Kaggle before treating cloud multi-GPU acceptance as passed.
