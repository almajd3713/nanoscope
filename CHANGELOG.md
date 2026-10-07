# Changelog

## 0.4.0 (2026-10-07)

Phase 12 of the build plan: a docker-compose stack.

### Added
- `docker compose up`: the API and a CPU worker on `127.0.0.1:8765`; profiles add marimo
  notebooks (`--profile notebook`) and an NVIDIA GPU worker (`--profile gpu`). The services run
  as uid 1000 with no capabilities, a read-only root filesystem and resource limits
  (`NANOSCOPE_MEM_LIMIT`, `NANOSCOPE_CPUS`, `NANOSCOPE_PIDS_LIMIT`), and cannot reach the Docker
  socket. See `docs/deploy.md`.
- Images: `ghcr.io/almajd3713/nanoscope:cpu` and `:cuda`, each also tagged with its version.
  A release candidate publishes only its versioned tag.
- `NANOSCOPE_JOBS_OFFLINE=1`: jobs that need the network (Hub sync, `push_to_hub`, W&B) are refused.
- The `notebook` extra, and marimo notebooks `01_first_model` to `04_ablations` in
  `notebooks/marimo/`. A lesson may ship a `notebook.py`.
- Compose tests (`pytest -m compose`) and a `compose` CI job: a bigram trains over the API,
  the services are hardened, and a run that was training when the stack went down resumes
  after `up`.

### Changed
- A run stopped because its worker shut down reads `queued` in `status.json`, not `cancelled`: its
  job is back in the queue and resumes from its last checkpoint.
- Torch thread counts respect the container's CPU limit (a 4-CPU container used to start one
  thread per host core).
- The `.ipynb` notebooks are replaced by the marimo ones (`notebooks/kaggle.ipynb` stays).
- The README describes four levels, `nanoscope serve` and compose.

## 0.3.0 (2026-10-06)

Phases 6-10 of the build plan: the library becomes servable, and gains blocks, a
curriculum engine and an HTTP API.

### Added
- Servable library: runs, studies and baselines are addressed by refs under
  `NANOSCOPE_HOME`; every long operation writes `status.json` and live progress; a `STOP`
  file cancels a run; files other tools read carry a versioned `schema` (`nanoscope/schemas/`).
- Queue and workers: a SQLite job queue, `nanoscope worker`, `nanoscope run-job`, and
  multi-device studies through worker processes.
- Blocks: composable `nanoscope.blocks` (attention, MLP, norms, positional encodings,
  `Block`, `Decoder`), naive reference implementations in `nanoscope.reference`,
  `describe()` shape tracing, per-block statistics, and graph parse/edit of model source.
  `GPT2` and `Modern` are rebuilt on the blocks.
- Curriculum engine: lessons, checks, predictions, unlocks and block gating, with two
  paths (`foundations`, `modern-block`) and `nanoscope learn ...`.
- HTTP API (`pip install "nanoscope-lab[server]"`, `nanoscope serve`): runs, studies,
  compare, models, blocks, files, graph, learn, hardware and jobs, with SSE event streams,
  bearer-token auth beyond loopback, and `docs/openapi.json`. The API never executes user
  code; workers do. See `docs/server.md`.
- MIT `LICENSE`, package metadata, and a version single-sourced from `nanoscope/__init__.py`.

### Changed
- Old `GPT2` checkpoints and the `.ipynb` notebooks from earlier versions may not load or
  run unchanged. Run names and `config.json` kwargs are unchanged.
- `Study.run()` uses the queue only with several devices or `workers_per_device > 1`;
  `--shard` and `check_gpu_fits` are removed.
