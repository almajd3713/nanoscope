# Architecture

How nanoscope is put together today. [plan-tool.md](plan-tool.md) is the decision record (why each
choice was made, what was rejected, the user's answers in section 11); this page says what the code
does now, so a new reader does not need the plan to find their way.

## One code path, four levels

Everything a person does goes through the same library calls. The levels only change how much is
visible:

| Level | Python | Web app |
|---|---|---|
| 0 Learn | `run(Model)` | lessons, Train, the live run page |
| 1 Tinker | `run(Model, preset=..., seeds=N)` | the run form, duplicate, compare, the code editor |
| 2 Research | `Study`, `mode="record"`, `compare()` with intervals | studies, the queue per device |
| 3 Extend | hooks, presets, optimizers, `register_block` | the workspace tree, your own blocks and their certification |

The web app has no logic of its own for any of this. It draws what the API returns, and the API calls
the library. Anything the web app does has a command-line or Python equivalent, and the app can show
it ("Equivalent command", off by default in Settings).

## The library (`nanoscope/`)

The core is a training loop (`run.py`, `train_loop.py`) that reaches research features only through
hooks, so the loop has no branches for them. A run is addressed by a **ref** (a path under the runs
folder, or `baselines/...`). Everything on disk lives under `NANOSCOPE_HOME` (`paths.py`): runs,
studies, the queue, learner progress, certificates, the workspace.

Models are plain `nn.Module`s. The shipped ones (`Bigram`, `GPT2`, `Modern`) are built from the block
library, and `GPT2` and `Modern` are `Decoder` subclasses whose initial weights match frozen copies in
`nanoscope/reference/`, written without the block library, so a test can say the blocks still compute
what the textbook formula does.

## State other tools can read

Long operations never keep their state in memory only. A run always writes `status.json`, appends
`metrics.jsonl`, and can be cancelled by a `STOP` file or Ctrl-C. Every file another tool reads has a
JSON Schema in `nanoscope/schemas/` (`schema` and `nanoscope` keys; readers accept the current version
and the one before). The web app, the API and `nanoscope status` all read the same files, which is why
a run started from a terminal shows up live in the browser.

## Blocks, graphs and gating

- **Blocks** (`nanoscope/blocks/`): each block is a class with a family, a tier (primitive blocks are
  never locked) and a reference function in `nanoscope/reference/functional.py` that computes the same
  thing slowly. `describe()` traces a model on the `meta` device and reports shapes, parameters and
  FLOPs per module; a shape error names the module and the line.
- **The graph is a view of the file** (`blocks/graph.py`). `parse` reads a model file with `ast`
  (never importing it) into nodes; `emit` turns an edited graph back into source with libcst, touching
  only the arguments that changed. The edits are `set_arg`, `replace_block`, `remove_arg`, `add_layer`,
  `remove_layer`, `set_pattern` and `fill_slot`. A class outside the representable subset is
  code-only: it is shown with the reason and the line, and edited as text. `parse(emit(g)) == g` is a
  property test over random sequences of every edit.
- **Gating** (`learn/gating.py`): in the guided policy a composite block or feature is locked until the
  learner has built it in the lesson that unlocks it. It is checked where a block is imported, where a
  model is built, statically on a file, and before a run is queued. Gating is a learning aid, not a
  security boundary.
- **Certification** (`blocks/certify.py`, `blocks/certs.py`): a block you register with a reference
  function is checked against it by a worker, and the result is stored against the hash of the file's
  source, so editing the file makes the badge stale.

## Curricula

A lesson is data: `lesson.toml` (checks, unlocks, compute estimates), `lesson.md` and a starter file,
in a path folder (`nanoscope/curricula/`, or yours). Check kinds are `defines`, `equivalent` (a module
against a naive reference with the lesson's weight mapping), `forbid`, `trains`, `reproduces`,
`verdict` and `predicted`. Passing every check of a lesson earns its unlocks. `nanoscope learn ...` and
the Lesson page run the same code.

## Queue and workers

Work that runs a learner's code is a job in a SQLite queue (`nanoscope/queue.py`, WAL mode): `run`,
`study`, `describe`, `check`, `certify`, `generate`, `bench`, `prepare-data` and `sync-hub`. A worker
claims a job, runs it in a fresh process (`nanoscope run-job <id>`) with only the secrets that kind
needs, and writes the result back. A worker that dies mid-job leaves a job that another claims after
its lease expires; the child process dies with its worker. Studies use the queue when several devices
are in play and run in-process otherwise.

## The HTTP API (`nanoscope/server/`)

FastAPI, installed with the `server` extra, under `/api`. Errors are RFC 9457 problem documents that
carry the library's own message word for word. Server-sent events stream a run's state, steps and
samples from the files, the workspace's changes, and unlock and progress changes.

The API process **never imports or runs user code**: it reads workspace files with `ast`, validates
requests from specs, and queues jobs for a worker. A test lists the library modules the server may
import. `docs/openapi.json` is generated from the app and checked into the repo; the web client's
types come from it.

## The web app (`web/`)

React and TypeScript, Radix primitives with CSS Modules and the design system's tokens
([design-system.md](design-system.md)). It is built into `nanoscope/server/static`, which the wheel and
the images ship. The model page loads its graph engine (elkjs) and the Monaco editor only when opened.
Screens and their commands are in [gui.md](gui.md).

The page holds no model state: the file on the server is the model, a graph edit is a patch to that
file, undo is an earlier version of the file sent back, and an edit made elsewhere reaches the open
page through the workspace watcher.

## Running it

`docker compose up` starts the API, a worker, and (profiles) a notebook server and a GPU worker, all
hardened (non-root, no capabilities, read-only root, no Docker socket) and bound to loopback. The
supported use is one person on their own machine. Hosting for other people is a non-goal until there
is a sandbox for user code ([deploy.md](deploy.md), plan section 8). `pip install nanoscope-lab` gives
the library alone, and `nanoscope-lab[server]` adds the API.

## Where the plan and the checklist fit

[plan-tool.md](plan-tool.md) holds the rationale and the user's decisions; it is not edited to match
the code. [checklist.md](checklist.md) is the work list by phase, each item with the command that
shows it done. This page and the other docs under `docs/` describe what is built.
