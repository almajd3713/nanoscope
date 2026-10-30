# Changelog

## 0.7.0 (unreleased)

Every lesson reaches the model page, the editor's language server is on by default, Python 3.13 is supported, and the repo reads better on GitHub and PyPI.

### Added
- Model page: a model written in plain PyTorch shows the structure its code builds (one box per module, folded layers, shapes, parameters and a link to each line). An unfinished class still gets a report, with the error at its line. A new `structure` job and `POST /api/models/{ref}/structure`.
- Lessons: a lesson can ship a `[template]` built from blocks. `nanoscope learn start --blocks` and the lesson page's "Build it from blocks" start from it, and lessons 3 and 5 have one. A check's `param:` names take alternatives (`ln1.weight|norm1.weight`), so one check judges both ways in.
- New model asks for blocks or plain PyTorch, and the Models page says which models have a graph.
- `nanoscope --version`.
- Python 3.13.
- A favicon.

### Changed
- The editor's language server (basedpyright) starts by default and runs from the CPU image. `./up.sh` adds the GPU worker when Docker can use an NVIDIA GPU (`--cpu` and `--gpu` override it).
- The model page of a plain file says why it has no graph and offers a model built from blocks.
- README: the clone step, a CPU-only torch install line, all 33 lessons, and links that work on PyPI.
- Docs: an index in `docs/README.md`; the plans and the checklist moved to `docs/dev/`.

### Fixed
- Editor: pending edits are sent before hover, completion and signature help, so completions answer for the current text.
- The model page opened one event stream per hook and could hold all of the browser's connections, so its requests waited forever. Streams are now shared per URL.
- The run page no longer flashes "Loading run…" every second while a run is preparing.
- Tests no longer write runs into the repository root.

## 0.6.1 (2026-10-12)

Fixes and polish for the app after comparing the built screens with the mockups, and a new release workflow.

### Added
- Lessons: courses follow a catalog `order`, and a lesson page has previous and next links.
- Study builder: a variant outside the size tolerance gets a one-click suggestion ("Use ffn_hidden = N").
- Models page: file, class, last trace and edited columns, and a New model dialog. The model page has a
  Train button and an inspector column.
- Phone layout: a menu bar, runs as cards, the lesson before its outline, and notices for the model page
  and the study builder.

### Changed
- Levels: below Research the study builder, card export and benchmark controls are hidden (the pages stay).
- Run page: the header says when and how far a run finished. Compare and the forest plot fit long labels;
  the new-run form puts data-set fields last.
- Study page: verdict statements, a primary bundle button, and the report beside the bundle. The Hardware
  devices table fits its panel.
- Jobs show workspace files relative to the workspace.
- Releases: `tag.yml` is the only way to publish (it bumps the version, tags `main`, creates the GitHub
  release from this file and starts the publish workflows). Pull requests build only the CUDA image.

### Fixed
- E2E checks no longer trip on Monaco's empty alert nodes.

## 0.6.0 (2026-10-10)

Phases 15, 16 and 17: studies and hardware in the app, the inspect and authoring tools, and the rest of the curriculum.

### Added
- Studies in the app: a study builder (variants, seeds, size matching, an estimate before it runs), a live
  grid per study, preregistration commits and ablation cards (`nanoscope card export|push|compare`, opt-in
  and record mode only), and a Hardware page with benchmarks. Record-mode runs start after a read-only git check.
- Tools for models: `nanoscope inspect` (attention maps per head and a logit lens at any kept checkpoint,
  also a page in the app), `run(optimizer=make)` with a shipped Muon, `checkpoint_steps`, `ALiBi` and `MoE`
  blocks, typed blocks through a generated stub, `nanoscope learn author-check`, a Workspace tree and a
  Schemas browser at the Extend level, and an optional language-server service (`--profile editor-lsp`).
- Learning paths: honest-ablations, evaluation, efficiency, scaling-lite, interpretability and extend
  (24 lessons; see `docs/learn.md`). Each lesson with a long run has a CPU and a GPU variant; the GPU
  estimates are declared, not measured.
- Library: `run(init_seed=, data_seed=)` splits a seed into weights and batch order, `nanoscope.scaling.fit_power_law`
  (exponent with an interval, and a prediction interval), `nanoscope.interp.induction_curve`.
- New lesson check kinds: `matches`, `function`, `ablation`, `ladder`, `certified` and `preset`; `verdict`
  takes `pairing` and per-side optimizers, and `predicted` takes `must_hit`.

### Changed
- `MoE` is locked (composite) until the extend path's MoE lesson unlocks `block:MoE`.
- A block registered by a user can be registered again when its file is re-run.
- Study reports count finished runs only.

### Fixed
- A registered block's constructor arguments were not found by lesson checks (they read `__new__`).

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
