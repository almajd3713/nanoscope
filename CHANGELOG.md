# Changelog

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
