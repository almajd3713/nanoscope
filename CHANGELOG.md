# Changelog

## 0.5.0 (2026-10-08)

Phases 13 and 14: the web app, with the model page.

### Added
- The web app (`web/`, built into the wheel and the images): lessons, the run form and the live run
  page, runs, compare, components, settings, the queue. Everything it does has a command-line
  equivalent, shown in Settings on request. See `docs/gui.md`.
- The model page (`/model/<file>`): a block palette, the model's graph, an inspector, the lesson
  template as slots to fill and, from Tinker up, a Monaco editor on the same file. Dragging a block
  onto a slot, editing an argument, adding or removing a layer and writing a layer pattern
  (`sliding:global 3:1`) each change only the lines of the file they touch. Undo and redo re-send an
  earlier version of the file. A depth dial shows more per box: shapes and parameters from the last
  trace, where each call sits in the file, equivalence to its reference, and the layers' gradient
  norms from a run's block statistics.
- The Models page: your model classes with their last trace, and your own blocks with their
  certification (`register_block(reference=...)`; a worker checks the block against its reference,
  and the badge is stale once the file changes).
- Graph edits in `nanoscope.blocks.graph`: `add_layer`, `remove_layer`, `set_pattern`, `fill_slot`,
  and a class kind `filled` for `class OneHead(AttentionTemplate)`.
- Job kind `certify`, `POST /api/blocks/{name}/certify`, and a `certification` field on your blocks in
  `/api/blocks`.

### Changed
- Starting a lesson opens its starter file on the model page, with Train and Run the check.
- The `equivalent` check accepts `q.weight` for a template slot filled with the `Linear` block
  (`q.linear.weight`), so the template route of lesson 3 passes its check.

### Fixed
- Reloading a page whose path ends in `.py` (`/model/models/my_lm.py`) returned a 404.

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
