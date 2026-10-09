# The web GUI

nanoscope's web app is a view of the files the library writes. It shows what `status.json`,
`metrics.jsonl`, `learn/*.json` and the comparison say, word for word, and everything it does is
something the command line or Python can do too. Nothing here is a second implementation: it
calls the HTTP API ([server.md](server.md)), and the API calls the library.

## Running it

- **docker compose** (the usual way): `docker compose up`, then open the login link the `api` service
  logs (`docker compose logs api | grep login`). See [deploy.md](deploy.md).
- **From a checkout**: `pip install "nanoscope-lab[server]"`, then `nanoscope serve --worker cpu`.
  The wheel ships the built app; from a source tree run `make web` first.
- **Developing the app**: `pnpm -C web dev` serves it on port 5173 and proxies `/api` to a running
  server (`NANOSCOPE_API=http://127.0.0.1:8800 pnpm -C web dev` for another port).

The design rules (colour, type, copy, the banned patterns) are in
[design-system.md](design-system.md). Every pull request that touches the UI goes through its review
checklist.

## Screens

Each screen can show the command that does the same thing. That block, "Equivalent command", is off
until you turn on **Settings > Show command-line equivalents**, so most people never see it. The
commands below are what it shows.

| Screen | Path | What it is for | Equivalent command |
|---|---|---|---|
| Welcome | `/welcome` | The first-run question: guided (blocks unlock as you pass lessons) or open | `nanoscope learn start foundations/01-bigram`, `nanoscope learn unlock --all` |
| Lessons | `/learn` | Paths and lessons with state, what passing unlocks, and compute estimates | `nanoscope learn list` |
| Lesson | `/learn/<path>/<lesson>` | The lesson text (Surface, Deep, Reading), Start, Train, the check and its result, a prediction box when the lesson has a `predicted` check | `nanoscope learn start <lesson>`, `nanoscope learn check <lesson>`, `nanoscope learn predict <lesson>` |
| Run | `/runs/<ref>` | One run: state, step, ETA, the validation curve against the shipped baseline's range, samples, stop, resume, generate, duplicate; a failed run shows its error and traceback | `nanoscope status runs/<folder>`, `nanoscope stop <ref>` |
| New run | `/runs/new` | A form built from the model's constructor and the preset, validated as you type, with the request it will send | `nanoscope run <model> --preset <preset> --seeds N --set k=v` |
| Duplicate | `/runs/new?from=<ref>` | Starts from a run's config; change one thing, train the same seeds | the same `nanoscope run`, with the one change in `--set` |
| Runs | `/runs` | Every run with a ref prefix and state filter, live | `nanoscope status runs/<prefix>` |
| Compare | `/compare?runs=a,b[&preset=]` | The library's verdict per model with its 95% interval, a forest plot, every seed's curve, the precision plan | `nanoscope compare a b --preset <preset>` |
| Models | `/models` | Your model classes read from the workspace (never imported), each with the params and FLOPs per token of its last trace, a Trace button, and (Extend) your own blocks with their certification and a Certify button | `nanoscope graph <file>`, `nanoscope describe <file>:<Class>`, `nanoscope blocks --workspace <dir>` |
| Model | `/model/<file>[?class=]` | One model file: the block palette, the graph, the inspector for the selected box, the lesson template as slots to fill, and (Tinker and up) the code beside them | `nanoscope graph <file>`, `nanoscope describe <file>:<Class>` |
| Inspect | `/inspect/<run ref>` | What a trained run attends to and would predict, for a prompt, at any checkpoint it kept: a map per layer and head, the logit lens, and (when several steps were archived) one head across steps. Opens from the run page; each view is an `inspect` job a worker runs | `nanoscope inspect <ref> --step N --prompt TEXT` |
| Components | `/components` | The lock state of every gated block and feature, how each was unlocked, Unlock all and unlock-one | `nanoscope learn status`, `nanoscope learn unlock <id> --reason "..."`, `nanoscope learn unlock --all` |
| Settings | `/settings` | Theme, level and the command toggle (this browser only); the gating policy; each preset's data and Prepare; read-only facts about the server | `nanoscope learn status`, `nanoscope prepare-data <preset>` |

The footer on every screen is the queue: running and queued jobs with cancel, and the workers.
Cancelling a run saves a checkpoint first, so the run can resume.

Studies and Hardware arrive in a later phase; their nav entries are placeholders until then.

### The model page

The page is one file seen four ways, and every edit in any of them is an edit to that file:

- **Palette** (left): every block by family, your own last. A locked block is greyed with a lock and
  names the lesson that unlocks it ("Pass modern-block/02-rope to use it"); it cannot be dragged. The
  palette follows unlocks, file saves and finished certifications without a reload.
- **Graph** (centre): the file parsed with `ast` (it is never run), laid out afresh on every parse;
  nothing about the layout is saved. Click a box to select it. You can **drag** a block from the
  palette onto a box to swap it. Dropping a block of another family, a locked block, or a call the
  graph cannot edit is refused with the reason; a refusal from the server is shown word for word.
- **Inspector**: the selected block's options (each edit is one `set_arg`, which changes one line of the
  file) and a swap menu for its family. Locked options are listed but disabled, with their lesson.
  Every drag has this click path.
- **Template canvas**: a lesson template (`AttentionTemplate`, `BlockTemplate`, or a class that fills
  one) as its slots. Drop a primitive on a slot to fill it, or empty a filled slot.
- **Code** (Tinker and up): Monaco on the same file, bundled with the app (no CDN). Ctrl-S saves with the
  ETag it read; a save the server refuses because the file changed shows the server's diff with **Keep
  mine** or **Take theirs**. A file changed on disk by anything else reloads a clean buffer and asks
  before touching one with unsaved edits. ruff's findings and nanoscope's own (a locked use, a shape
  error from a describe job, a failed equivalence check) sit on their source lines.

**Undo** and **Redo** re-send an earlier version of the file with the ETag it has now, so the file on
the server is always the model. For a file in a lesson's folder the page also has **Train** and **Run
the check**, and a lesson's **Start** opens its starter here.

Each edit the graph sends is an operation of `POST /api/files/<file>/graph/patch`
(`set_arg`, `replace_block`, `remove_arg`, `add_layer`, `remove_layer`, `set_pattern`, `fill_slot`);
[blocks.md](blocks.md) and [server.md](server.md) describe them.

## Levels

The level switch in the top bar changes **which controls are visible**, nothing else: never colours,
layout or what a request contains. A control you leave hidden keeps its default and is never sent
changed.

| Level | Adds |
|---|---|
| Learn | lessons, training from a lesson, the live run page, the baseline range, samples |
| Tinker | the run form, seeds, duplicate and change one thing, compare, predictions |
| Research | the queue's per-device workers, the forest plot, studies and record mode as they arrive |
| Extend | the workspace tree, your own blocks and their certification (Models page), as they arrive |

The single table is `web/src/levels.ts`. A test checks that a level only ever adds controls.

## What the page never does

- It never computes a statistic. Verdicts, intervals, deltas, the precision sentence and every table
  cell come from the API (`rows[].verdict`, `rows[].text`, `precision_plan.text`), and a lint rule
  forbids statistics libraries. The one derived number is the ETA of a running run, from the newest
  step's seconds per step.
- It never rewords an error. A problem+json `detail` from the library is shown as written.
- It never shows a state by colour alone: every state and verdict is a glyph and a word.
- It never runs your code. The API only reads model files; workers run them.

## Tests

- `pnpm -C web test`: unit and component tests (fixtures are real API responses), including axe on
  the main pages.
- `pnpm -C web lint`, `lint:css`, `typecheck`, `tokens --check`: the design system's rules.
- `web/e2e`: Playwright against a real compose stack (`web/e2e/stack.sh up`, then `pnpm -C web e2e`).
  It checks the plan's done-when conditions: lesson 1 start to check passed in at most 6 clicks, Learn
  and Tinker sending the same run request, a failed run showing its error, and Unlock all.
