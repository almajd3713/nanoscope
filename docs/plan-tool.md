# nanoscope as a tool: library, HTTP API, GUI, composable models and curricula

Status: proposal, 2026-10-05. Written against `feat/phase-5-speed` (PR #5). Nothing here is
implemented. Phases continue the redesign numbering (phases 1-5 are done). The survey of
similar projects and the evidence behind section 3's ideas are in
[`plan-tool-landscape.md`](plan-tool-landscape.md).

**Decisions recorded 2026-10-05** (section 11). They changed the plan in these places:
`gpt2.py`/`modern.py` are rebuilt on blocks (5.1), drag-and-drop is in the MVP, lesson
gating is new (6.4), notebooks in the stack use marimo (9.1), the GUI may make the
preregistration commit (11.8), and the roadmap is renumbered (section 10). The working
build list, with a done-check per item, is [`checklist.md`](checklist.md).

## 0. The recommendation on one page

1. **Start with phase 6, "make the library servable", not with the API or GUI.** The run
   folder is almost a database already, but some things only work inside one process:
   run state is guessed from file mtimes, cancellation is Ctrl-C on the main thread, a
   run's model class can't be rebuilt from disk, and studies exist only as executable
   Python. Fixing these helps the CLI and notebooks immediately.
2. **The filesystem is the source of truth, and the server keeps no state of its own.** Add
   `status.json` and a `STOP` file next to `config.json`/`metrics.jsonl`. The server then
   shows and controls runs started from notebooks, terminals or Kaggle, and survives
   restarts.
3. **One job queue (SQLite) and one `nanoscope worker`, used by both the CLI and the
   server.** This replaces the static `--shard` split. Users pick how much the levels expose,
   but every level runs the same scheduler.
4. **Composable models through a `nanoscope.blocks` library.** A graph editor is a second
   view of a **real `.py` file** written in a small, parseable composition style. Code is
   the source of truth, and every block has a naive-reference equivalence test, the same
   discipline as `tests/test_models.py`. `GPT2` and `Modern` are rebuilt on the blocks,
   and each block keeps a small independent naive reference.
5. **Curricula as data.** Paths of lessons. Each lesson has a starter file, an experiment
   (a run, a Study, or a check) and a done-check that the library evaluates. These lessons
   are the old M0-M7 plan made runnable. **Bigger blocks are locked** until the learner
   has built them from primitives in a lesson (6.4). Researchers turn this off with one
   level choice.
6. **Depth comes from views over the same artifacts.** A surface view and a research view
   of a model, a result or a lesson read the same files and call the same functions.
7. **API: FastAPI in a `nanoscope[server]` extra.** Long work is always a job. Progress is
   sent as Server-Sent Events (SSE) built by tailing files.
8. **GUI: a React SPA served by the API container.** It contains Monaco editing workspace
   `.py` files, an architecture graph with drag-and-drop editing (in the MVP), lesson
   pages, and compare views that show the library's verdicts word for word.
9. **docker-compose:** `api` + `worker` (CPU by default, a `gpu` profile per device), with
   named volumes for runs, token data and compile caches, and a host-mounted git workspace.
   A `notebook` profile runs marimo. The package goes to PyPI and a CPU image to GHCR
   before compose is built.
10. **Security: design for a single user on localhost first.** The API process never
    imports user code. All user code runs in workers behind a `JobRunner` interface, so a
    container per job can be added later for hosted use without changing the API.

The GUI's value is not dashboards, where W&B/MLflow/Aim are better. Its value is
**guarded experimental design and learning by measurement**: fixed eval text, bits per
byte, seed-level paired confidence intervals (CIs), parameter/FLOP matching,
preregistration, shipped baselines, and blocks proven equivalent to references, all
behind a zero-config on-ramp.

---

## 1. What exists, and where the seams are

### 1.1 Inventory (about 3,000 lines in `nanoscope/`)

- **`run()`/`RunResult`/`RunGroup`** (`nanoscope/run.py`). `_split_kwargs` routes kwargs to
  the model or the preset. Run dirs are named by a hash of what changed. `config.json`
  records `stats`.
- **Trainer** (`nanoscope/train_loop.py`). AdamW plus cosine schedule, exact resume
  including RNG, hooks (`on_step`/`on_eval`/`on_checkpoint`), opt-in `torch.compile`.
- **Progress** (`nanoscope/progress.py`). `ProgressBar` hook and `snapshot(root)` read from
  disk, including `plan.json`. `nanoscope status` prints it.
- **Compare/stats** (`nanoscope/compare.py`, `statistics.py`). Paired t or Welch CIs. It
  refuses mismatched eval text or `val_loss` across tokenizers, and notes budget mismatches
  and mixing of explore and record runs.
- **Study** (`nanoscope/study.py`). Variants x seeds, `Tokens`/`FLOPs` budgets,
  `match="params"`, record mode (git provenance, frozen predictions), reports in
  `experiments/<name>/`. Parallel workers use static shards (`_run_parallel`, `study.py:315`).
- **Hardware** (`hardware.py`, `bench.py`). `probe_memory` runs one real step,
  `check_gpu_fits`, `cpu_threads`. `bench` gives a CPU-bound or GPU-bound verdict.
- **Data** (`dataset.py`). Token data comes from the Hub first and is tokenized locally as
  a fallback. `publish_data` uploads it. **Presets** (`presets.py`): a frozen dataclass,
  `register_preset`, and three presets.
- **Models** (`nanoscope/models/`). `Bigram`, `GPT2`, `Modern` (every component switchable).
  Naive-reference tests are in `tests/test_models.py` (attention vs loop, RoPE vs complex
  rotation, RMSNorm formula, causality).
- **Baselines** (`nanoscope/baselines/tinystories-5min/*/seed-{0,1,2}`), **integrations**
  (`wandb_hook`, `HubSync`), **CLI** (`cli.py`: run, presets, bench, prepare-data,
  publish-data, status, study, report, compare).
- **Guard test**: `tests/test_first_model_notebook.py` requires 3 code cells and under
  120 s on CPU.

### 1.2 Clean seams

- Run folders. `config.json` has enough to label and compare runs (`compare._label`,
  `compare.py:136`). `metrics.jsonl` is append-only per step (`train_loop.py:338-340`).
- `progress.snapshot(root)` (`progress.py:68`) is already a status endpoint in shape.
- `Comparison.rows` are JSON-ready (mean, CI, paired, status), and `StudyReport.write`
  writes `results.json` (`study.py:416`). Nothing outside the library needs to compute
  statistics.
- `bench()` returns a dataclass. `Study.sizes()` counts on the `meta` device.
  `probe_memory`/`check_gpu_fits` are the admission check a scheduler needs.
- Model constructors are typed with defaults, e.g. `GPT2(vocab_size, context_length=256,
  d_model=128, n_layers=4, n_heads=4)`, and `run()` already introspects them. Forms and
  graphs can be generated from them.
- `Modern` already works as a component palette ("every component has a switch",
  `models/modern.py` docstring). That is the starting inventory for `nanoscope.blocks`.

### 1.3 Couplings to fix first

| Coupling | Evidence | Why it matters |
|---|---|---|
| Roots are relative to the working directory | `RUNS_DIR = Path("runs")` (`run.py:20`), `REPORTS_DIR` (`study.py:50`), name lookup in `compare._resolve` (`compare.py:110`) | Server, workers and notebooks must agree on one root. |
| State is inferred, so failures are invisible | "running" if `metrics.jsonl` changed in the last 120 s (`progress.py:15,56`) | A crash, a Ctrl-C and a dead Kaggle session all read as "stopped". |
| Cancellation only through SIGINT on the main thread | `train_loop.py:272-282` | Nothing can stop one run from outside the process. |
| The model class can't be rebuilt | `"class": model_cls.__name__` (`run.py:332`) | No load-and-generate, no resubmit, no "same but change X". |
| Studies are executable Python | `load_study` runs the file (`study.py:434`) | The GUI needs a declarative form that still produces a file you can commit. |
| Static sharding | `_run_parallel`: worker *i* gets jobs `i::n` | A dead or slow worker strands its share of the jobs. |
| torch state is global to the process | `torch.manual_seed` (`run.py:324`, `train_loop.py:211`), `set_num_threads` (`study.py:279`) | Each process can run only one training run. |
| Output goes through `print`/tqdm only | `_log` in `run.py`, `dataset.py` | Data prep, the slowest first-time step, leaves no status on disk. |
| Hub reachability is cached forever | `@functools.cache` on `_hub_files` (`dataset.py:90`) | A long-lived server that started offline never uses the Hub again. |
| Checkpoints are only for resume | `keep_checkpoints=2`, older ones deleted (`train_loop.py:157`) | Training-over-time views (e.g. an induction bump) need archived checkpoints. |
| The optimizer is hard-coded | `torch.optim.AdamW` (`train_loop.py:218`) | "Level 3: optimizers" isn't real yet. |

---

## 2. Users and jobs to be done

| Level | Who | Job to be done | With the tool |
|---|---|---|---|
| 0 Learn | Student, workshop attendee | "I built a model. Did it learn, and how does it compare to a real one?" | Follow a lesson, compose or type a model, click Train, watch the curve and the architecture graph, see the GPT-2 baseline verdict. |
| 1 Tinker | The same person in week two | "What if I change X? Is the difference real?" | Swap a block in the graph or edit code, use seeds, ask "is it within noise?" |
| 2 Research | Grad student or independent researcher | "A fair, preregistered ablation on my hardware, with a table I can defend." | Study builder or file, queue across devices, live forest plot, report bundle with provenance. |
| 3 Extend | Contributor or instructor | "Add a block, preset, optimizer or lesson, and have it work everywhere." | Editor or local IDE on the workspace, registries, block reference tests, curriculum authoring. |

Two deployment personas decide the priorities. The first is the **remote-GPU user**: a
laptop plus one GPU box, which is the main reason for docker-compose. The second is the
**instructor**: one machine and a room of trusted learners, which comes later (section 8).

**What is different about nanoscope** (evidence in the landscape doc):

- Comparisons that refuse to mislead (`compare.py:249-273`).
- Studies that are fair by construction.
- Shipped baselines on a 2-minute CPU preset.
- State that is resumable and visible from outside the process.
- Readable reference models with equivalence tests.

**What is not different**, so we should not compete on it:

- Training throughput (modded-nanogpt, nanochat, torchtitan).
- Dashboards (W&B, Aim, MLflow).
- Cluster scheduling (Ray, SkyPilot, Slurm).
- Serving.
- Static model viewers (Netron).
- Polished in-browser explainers of a fixed GPT-2 (Transformer Explainer, bbycroft).

---

## 3. Product shape

### 3.1 Ideas worth building, ranked (detail: landscape doc section 6)

| # | Idea | Differentiated? |
|---|---|---|
| 1 | **Verdict-first results with precision planning.** Every comparison opens with "better / worse / within noise". Before launching, the user sees how wide the CI would be with n seeds, using the seed spread measured on the shipped baselines, so "how many seeds?" gets an answer up front. | Yes. Trackers show mean ± std bands, not paired CIs or power. |
| 2 | **Predict, then run.** A learner or researcher writes down an expected value or verdict before a run or study (record mode's `predict` generalized). The result page scores the prediction against the CI. | Yes, as teaching. Preregistration tools exist; none are built into a training UI. |
| 3 | **Blocks with equivalence certificates.** Each block in the graph carries a badge showing it matches a naive reference. A user's own block earns the badge by passing a check job. | Yes. |
| 4 | **Lessons with three kinds of check.** *Implement* (equivalence to a reference), *train* (reach a metric on a preset), *claim* (a CI-backed verdict or a scored prediction). | Yes in combination. ARENA has tests but no training claims. Courses have no checks. |
| 5 | **Ablations as shareable data.** A record-mode study exports an *ablation card* (spec, per-seed finals, provenance, commit) to a Hub dataset. Because the eval text is fixed per preset, cards from different people compare directly (unpaired), with a CI. | Yes. It is the speedrun's p<0.01 discipline turned into a library feature. |
| 6 | **Explorables of your own model over training time.** Attention maps, logit lens and per-block stats on the learner's own checkpoints, scrubbing through training. | Partly. Transformer Explainer and bbycroft do this for a fixed pretrained model, not across training. |
| 7 | **Compute-honest launches.** Before every run, lesson or study, an estimated wall time on this machine, from the saved `bench` history. | Partly. Rare in practice, but easy. |
| 8 | **Noise-floor display.** Show the seed noise per preset and metric, so a learner sees which differences can be detected at all (from AI2's Signal-and-Noise result). | Partly. |
| 9 | **CPU speedrun track.** "Beat the bigram/GPT-2 baseline in 2 CPU minutes", gated by the same significance rule. | Engagement. Weak differentiation. |
| - | Live curves, run tables, a code editor, a graph viewer, docker | **Table stakes.** Needed, but not a reason to choose nanoscope. |

### 3.2 Capabilities by phase (effort and order in section 10)

Library first: lifecycle, cancellation, rebuildable runs, schemas, home dir, StudySpec,
queue, and a CI baseline. Then the blocks library and `describe` (with GPT-2/Modern
rebuilt on blocks), the curriculum engine with lesson gating, the API, release prep (PyPI,
GHCR), compose, the GUI MVP in two parts (Learn/Tinker screens, then the model page with
drag-and-drop graph editing), research views, Extend, and the remaining curriculum paths.

### 3.3 Non-goals

- A second training path or an "API trainer". The server calls `run()`/`Study`.
- An IDE in the browser. No terminal, debugger, git operations (except the one-click
  preregistration commit, 11.8), extension marketplace or notebook clone. The editor is
  minimal (section 9.4).
- Free-form DAG wiring. Decoders are stacks. The graph edits the representable subset
  (section 5.2), and lessons that build a block from primitives in the graph use templates
  with named slots (6.4). Anything else is code.
- Statistics computed in the frontend. A results database separate from run folders.
- Multi-tenant hosting before a per-job sandbox exists (section 8).
- Multi-node, Kubernetes, autoscaling. Competing with W&B on dashboards or with vLLM on
  generation.
- Editing or deleting record-mode results from the UI.

---

## 4. Library changes before an API makes sense (phase 6)

`run(Bigram)` stays as it is, and the notebook test gates every phase.

1. **One home.** `nanoscope.paths.home()` returns `$NANOSCOPE_HOME`, or today's defaults
   (`./runs`, `./experiments`, `~/.nanoscope/data`). Roots are read at call time, not at
   import time. Containers set `NANOSCOPE_HOME=/nanoscope`.
2. **Run refs.** A ref is the path relative to the runs root, which is already unique:
   `tinystories-5min/bigram-1a2b3c4d/seed-0`, a set `tinystories-5min/bigram-1a2b3c4d`, a
   study run `studies/m1-ablation/no-rope/seed-2`, and a read-only
   `baselines/tinystories-5min/gpt2`. `nanoscope.store` provides `resolve`, `list_runs` and
   `list_sets`, and rejects `..`. `compare()` accepts refs.
3. **Versioned schemas.** Add `"schema": 1` and `"nanoscope": __version__` to
   `config.json`, `status.json`, `plan.json`, `study.json` and `results.json`. JSON Schema
   files live in `nanoscope/schemas/`. Readers accept versions N and N-1. Old folders and
   the shipped baselines are read as v0 through a small upgrader.
4. **`status.json` lifecycle.** A `StatusFile` hook that `run()` always attaches writes it
   atomically. States: `queued`, `preparing`, `running`, `done`, `stopped`, `cancelled`,
   `failed`. It also records `error` (type, message, the last 20 traceback lines), `pid`,
   `host`, `device`, `job_id`, and `heartbeat_at` (at most once every 5 s). `run()` wraps
   training in `try/except BaseException`. `snapshot()` prefers `status.json` and falls
   back to inferring from mtimes.
5. **Cooperative cancel.** `train()` gains `should_stop: Callable[[], bool]`, checked where
   `stop_requested` is checked now (`train_loop.py:344-349`). `run()` points it at
   `run_dir/"STOP"`, and `nanoscope stop <ref|study>` creates that file. This works across
   processes and containers, and Ctrl-C keeps working as before.
6. **Rebuildable identity.** Record `model.ref` (`module:qualname`, or a workspace file
   path) and `source_sha256`. `load_run(ref)` rebuilds the model and loads the latest
   checkpoint. The run hash does **not** include the ref or the source hash, so a refactor
   doesn't orphan runs. A changed hash is reported as drift. Classes defined in
   `__main__` record their source and are marked as comparable but not rebuildable.
7. **Specs and validation.** `ModelSpec` (parameters, types, defaults, from-data flags,
   docstring) and `PresetSpec` (fields with help text in field metadata).
   `validate_run_request()` returns every problem at once. `run()` still raises on the
   first.
8. **Declarative studies.** `StudySpec` in TOML, with `Study.from_spec`/`to_spec`.
   `load_study` accepts `.py` or `.toml`. Record mode requires the spec file to be
   committed, like a `.py` study. `match_params` gets a declarative form
   (`match_knob = "ffn_hidden"`, `range = [64, 1024, 8]`).
9. **Smaller items.** Data prep writes `prepare.json` (stage, done, total). `_hub_files`
   gets a 5-minute TTL. `bench(save=True)` appends to `hardware/bench.jsonl`. `_log` goes
   through `logging.getLogger("nanoscope")`. `to_dict()` matches the schemas.
   `checkpoint_steps=[...]` (archived, never pruned) is kept separate from resume
   checkpoints.

---

## 5. Composable models, code and graph

### 5.1 The block library

`nanoscope/blocks/` holds short, readable modules, one concept per file, each with a typed
constructor, a docstring, `flops_per_token()`, and a naive reference in `tests/`:

| Family | Blocks (MVP in bold) | Reference test |
|---|---|---|
| Embedding / head | **TokenEmbedding**, **LearnedPosition**, **Head** (tied/untied) | lookup equals one-hot matmul |
| Positional | **RoPE**, **NoPE**, ALiBi | complex rotation; relative-position invariance (exists today) |
| Norm | **LayerNorm**, **RMSNorm**, QK-norm flag | formula (exists) |
| Attention | **Attention**(n_heads, n_kv_heads, pos, qk_norm, window) covering MHA/GQA/MQA/sliding; MLA later | loop over heads (exists for GPT2/Modern) |
| MLP | **GELUMLP**, **SwiGLU**(hidden) | explicit formula; parameter parity (exists) |
| MoE | MoE(experts, top_k, aux_loss) | loop over tokens and experts; aux loss returned through the existing `(logits, aux)` convention (`train_loop._split_output`) |
| Structure | **Block**(norm, attn, mlp, pre/post), **Decoder**(d_model, n_layers, block or layer pattern, final norm, tie, z_loss) | causality; untrained loss near uniform (exists) |
| Primitives | **Linear**, **Activation**, **CausalMask**, **ScaledDotScores**, **Softmax**, **WeightedSum**, **SplitHeads/MergeHeads**, **Residual**, and a **Composite** base with named slots | formula per primitive. Never locked; lesson templates are built from them (6.4) |

**Relation to `gpt2.py`/`modern.py`: rebuild them on the blocks (decision 11.4).**
`GPT2` and `Modern` become short compositions of blocks. They keep their class names,
constructor signatures and defaults, so run folder names, run hashes and `config.json`
stay the same. Equivalence must still be tested against something independent, so:

- Each block has a small naive reference in `nanoscope/reference/`, the functions now in
  `tests/test_models.py` moved into the package, so lesson checks can use them from an
  installed wheel. Reference modules import only `torch` and `math`, and a test enforces
  that.
- Today's hand-written `gpt2.py` and `modern.py` are frozen in `nanoscope/reference/` as
  whole-model references. The rebuilt classes must match them: same logits to 1e-5 after
  loading the same `state_dict`, and the same initial weights for a given seed.
- If keeping the old `state_dict` keys or the init order is impractical, the change is
  made deliberately: a key map in `load_run`, a schema bump, and baselines re-exported
  under the schema policy (11.11). Old local checkpoints from before the rebuild may not
  resume. That is a known cost of this decision.

Two implementations of a model are not two code paths, because everything still trains
through `run()`. The readable one-file teaching models are no longer `gpt2.py`/`modern.py`;
the lessons (6.2) and the frozen references take that role.

### 5.2 Code is the source of truth; the graph is a view

A composable model is a normal `.py` file in a **representable subset**:

```python
from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, RoPE, SwiGLU

class MyLM(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256):
        super().__init__(
            vocab_size, context_length, d_model=128, n_layers=4,
            block=Block(norm=RMSNorm(), mlp=SwiGLU(hidden=344),
                        attn=Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True)),
            tie_weights=True, z_loss=1e-4,
        )
```

- `run(MyLM)` works unchanged. Block arguments are specs that `Decoder` instantiates once
  per layer.
- **Parse without executing.** `nanoscope.blocks.graph.parse(path)` uses `ast` to turn
  each `Decoder` or `Composite` subclass into graph JSON: nodes, arguments and source
  spans. An argument
  that is a literal, a registered block call, or an `__init__` parameter is
  representable. A call to anything else becomes an **opaque node** (`Custom: mymod.Foo`),
  shown with the shapes from `describe` but no editable internals. Any other statement in
  `__init__` makes that class code-only, and the reason and line number are shown.
- **Edit by patching.** `emit(graph, source)` uses libcst to change only the edited
  arguments, keeping comments and formatting. Layout is computed automatically (ELK)
  every time, so there is no sidecar layout file and nothing in a database.
- The file is the only artifact. Git, the CLI, CI and local editors all keep working.
- **Round-trip tests:** `parse(emit(g)) == g` for randomly generated palette graphs
  (property test), code→graph→code is byte-identical on fixtures when nothing is edited,
  and the GPT-2/Modern compositions are equivalence-tested as in 5.1.
- **User blocks.** `@register_block(reference=fn)` puts a user's `nn.Module` in the
  palette. A check job compares it with `fn` on random inputs. If it passes, the block
  gets the certified badge (idea 3).

### 5.3 Visualization and live state

- **Static.** `nanoscope.inspect.describe(model_cls, preset, **kw)` traces on the `meta`
  device with forward hooks. Per module it reports input and output shapes, parameters
  (total and non-embedding), FLOPs per token (analytic for blocks, 6N as fallback) and a
  rough memory estimate (weights + AdamW state + activations). The CLI is
  `nanoscope describe file.py:MyLM`. It executes user code, so it runs in a worker.
- **Live.** An opt-in `BlockStats` hook runs at eval steps only, so its cost is
  negligible. Per block it records activation RMS, gradient norm, update-to-weight ratio
  and attention entropy, appended to `blockstats.jsonl`. The graph colors nodes from it
  over SSE, and `nanoscope status --blocks` prints the same data. This is the
  state-visibility rule applied to model internals: written to disk and readable from
  outside.
- **Over training time.** `checkpoint_steps` (log-spaced in the lessons that need it)
  plus an inspection job: attention maps and logit lens for a prompt at any archived step
  (idea 6).

---

## 6. Curricula as data, and depth without a second path

### 6.1 Format

```
nanoscope/curricula/<path>/path.toml              title, level, prerequisites, estimated compute
nanoscope/curricula/<path>/<nn-slug>/lesson.toml  experiment + check + depth tiers + unlocks/forbid + compute variants
nanoscope/curricula/<path>/<nn-slug>/lesson.md    "## Surface", "## Deep" and "## Reading" sections
nanoscope/curricula/<path>/<nn-slug>/starter.py   copied into workspace/lessons/<slug>/ on start
tests/solutions/<path>/<nn-slug>.py               reference solution, used by CI, not shipped in the wheel
```

Curricula live inside the package so that a `pip install` gets the lessons. A lesson that
needs GPU hours declares two variants (decision 11.6): `[compute.cpu]` (a modest version
that CI runs) and `[compute.gpu]` (the full version). Each has a preset or budget and a
compute estimate. The loader refuses a lesson whose CPU estimate exceeds 15 minutes
unless it also declares the GPU variant.

Kinds of check, implemented in `nanoscope.learn.checks` and run as worker jobs:

- `defines`: the class exists and builds.
- `equivalent`: the user's block or model matches a reference within a tolerance.
- `trains`: a run of the user's model on a preset reaches a metric threshold.
- `verdict`: compare A against B and require a verdict, with n seeds.
- `predicted`: a prediction was recorded before the run and is scored afterwards.
- `reproduces`: the result falls within the CI of a shipped baseline.

Progress is kept in `$NANOSCOPE_HOME/learn/progress.json`, and unlocks in
`learn/unlocks.json` (6.4). The CLI is `nanoscope learn list|start|check|status|unlock`, so
lessons also work without the GUI and on Kaggle. Existing notebooks are referenced by
lessons, not copied, so content lives in one place.

### 6.2 Paths, levels and the old plan

| Path | Level | Content | Source in old docs |
|---|---|---|---|
| Foundations | 0 | bigram → MLP → one attention head → multi-head attention → transformer block → GPT-2; `equivalent` checks unlock Attention, Block and Decoder; `trains` + `reproduces` | notebooks 01-02 |
| The modern block | 1 | implement RoPE, RMSNorm, SwiGLU, GQA, QK-norm, z-loss; `equivalent` checks unlock each | notebook 03, M1 "build" part, `docs/archive/roadmap-v2.md` Stage 3 |
| Honest ablations | 2 | seeds, CIs, matching, preregistration; capstone `studies/m1_ablation.py` with "state the gap with a CI and name one null component" as a `verdict` check | M1 acceptance criteria, notebook 04 |
| Efficiency | 2-3 | bench, compile, KV cache, roofline | M4 (your milestone) |
| Evaluation | 2 | eval noise, paired tests, seed vs eval-sampling variance | M5 |
| Interpretability | 2-3 | induction heads over checkpoints, logit lens | M6 |
| Scaling-lite | 2 | FLOP budgets, a 3-4 point ladder, a preregistered prediction | M2 (needs a fit helper; CPU variant: a 3-point ladder on `tinystories-5min`; GPU variant: a 4-point ladder on `fineweb-edu`) |
| Extend | 3 | write a certified block, add a preset, Muon vs AdamW | M3 items |

The reading lists in `lesson.md` reuse the B/R/S/K tiers of `docs/archive/roadmap-v2.md`. That is where that
document's content ends up (section 13).

### 6.3 Depth is a view setting, not a mode

| Surface | Surface view | Detailed view | Research view |
|---|---|---|---|
| Model | block diagram (Embedding → N× Block → Head) | expanded blocks with shapes, params and FLOPs | code, FLOP formulas, equivalence status |
| Training | progress bar + loss | curves, samples, per-block stats | `metrics.jsonl`, `blockstats.jsonl`, checkpoints |
| Result | verdict sentence | CI table | per-seed values, the test used (paired or Welch), notes |
| Lesson | "## Surface" + one run | "## Deep" + seeds | the same experiment as a Study with record mode |

Each column reads the same files and calls the same functions. In a lesson, the deep tier
is the *same* experiment with more seeds or a bigger budget: the same call with different
arguments.

### 6.4 Lesson gating: build it before you compose with it

Bigger blocks (multi-head attention, the transformer block, the decoder stack, RoPE
variants, GQA, MoE, ...) are **locked** until the learner has built them from simpler
primitives in a lesson, in code or in the graph. Locked blocks show in the palette with
their unlock condition. Imports of them from a learner's file fail with the same message.

**What is locked.** The lock list comes from the curricula, not from a separate list. A
lesson declares `unlocks = ["block:Attention", "feature:gqa"]` in `lesson.toml`, and a
block or feature is lockable exactly when some lesson unlocks it. Primitives (5.1),
embeddings, `Head`, `LayerNorm` and `GELUMLP` are never locked. The MVP lock set:

- Foundations: `block:Attention`, `block:Block`, `block:Decoder`.
- The modern block: `block:RMSNorm`, `block:RoPE`, `block:SwiGLU`, `feature:gqa`
  (`n_kv_heads < n_heads`), `feature:qk_norm`, `feature:z_loss`.
- Later: `block:MoE`, `block:ALiBi`, `feature:sliding_window`.

A test checks two things: every lockable block or feature has exactly one unlocking
lesson, and every unlock id names a registered block or feature.

**State on disk.** `$NANOSCOPE_HOME/learn/unlocks.json` follows the versioned schema
`unlocks.v1`. It is written atomically and readable from outside the process, as the
state-visibility rule requires. It holds:

- `policy`: `"guided"` or `"open"`;
- for each id: `how` (`earned`, `skipped` or `open`), `lesson`, `at`, and `evidence`, the
  path of the check result `learn/checks/<check-id>.json` with the equivalence numbers.

`nanoscope learn status` prints it, and `GET /learn/unlocks` returns it. In deployment B
it moves under `users/<name>/learn/` with the rest of the learner state.

**How unlocks are earned.** A lesson's done-check must pass. For an implement lesson that
means the `equivalent` check against the independent naive reference in
`nanoscope/reference/`, plus the lesson's `forbid` lint (no `F.scaled_dot_product_attention`
or `torch.nn.MultiheadAttention` in the attention lessons, for example). On a pass,
`progress.json` marks the lesson passed and `unlocks.json` gets its ids in the same step.
Nothing else earns an unlock.

**The escape is a level choice, not a second code path.**

- No `unlocks.json` means policy `open`. Level 0 `run(Bigram)`, existing scripts,
  notebooks, Kaggle and researchers never see gating.
- The first `nanoscope learn start`, or "I'm learning" in GUI onboarding, writes policy
  `guided`.
- `nanoscope learn unlock --all` sets policy `open`. In the GUI this is Components →
  Unlock all, and choosing the Research or Extend level offers it once.
- `nanoscope learn unlock <id> --reason "I know this"` skips one lesson's lock.

Both escapes are recorded (`how: open` or `skipped`), so progress shows what was earned
and what was skipped. The same gate function runs at every level; only the policy value
differs.

**Running versus composing.** Locks limit what a learner *composes*, never what they
*run*. These always work under `guided`:

- `run(GPT2)`, `run(Modern, n_kv_heads=1)`, presets and shipped baselines;
- Study variants of shipped models;
- generation from any run.

Shipped models import blocks from their submodules (`nanoscope.blocks.attention`), and
they carry a `shipped` marker that the composition gate skips. Three things are gated:
names imported from `nanoscope.blocks` in user code, a user `Decoder` or `Composite`
subclass that builds a locked block or uses a locked feature, and the GUI palette.
Subclassing a shipped model counts as running it. That loophole is documented, not
closed.

**Where it is enforced.** There is one policy function,
`nanoscope.learn.gating.check(ids) -> list[Locked]`, used at three points:

1. **Import.** `nanoscope.blocks` exports through a module `__getattr__` (PEP 562). Under
   `guided`, a locked name raises `LockedBlockError`, a subclass of `ImportError`, that
   names the lesson and the escape command.
2. **Build.** Building a non-shipped `Decoder` or `Composite` checks the blocks and
   features it uses. This catches feature locks such as GQA, which an import check cannot
   see.
3. **Static.** `gating.scan(path)` reads the file with `ast` and executes nothing. It
   feeds the editor's inline diagnostics, the graph's lock badges,
   `validate_run_request` (a `422` from the API, including on graph patches) and the
   lesson check's `forbid` lint.

**Raw `.py` files and the in-site editor follow the same rules.** A raw file run with
`python`, `nanoscope run`, a Study or a notebook meets points 1 and 2 at runtime. In the
editor, point 3 marks the line on save before anything runs, the graph shows the node as
locked, and a run submission is refused at validation. If it somehow reached a worker,
points 1 and 2 would still stop it there.

**Building from primitives in the graph.** Free-form wiring stays a non-goal (3.3). An
implement lesson that offers a graph route ships a *template*: a `Composite` subclass with
named slots. For attention the slots are `q`, `k`, `v`, `scores`, `mask`, `normalize`,
`mix` and `out`. A template parses and patches like any other composition (5.2). The
learner fills slots by dragging primitives into them; the code route writes the math in
`forward`. The same `equivalent` check judges both.

**Limits.** Import gating is a learning aid, not security. Python cannot stop determined
code: importing `nanoscope.blocks.attention` directly, `importlib`, copying the shipped
source, `torch.nn.MultiheadAttention`. The `forbid` lint catches the obvious cases at check
time; the rest is the honor system. Gating protects nothing in deployments B and C and is
never used for access control.

---

## 7. HTTP API

**FastAPI + uvicorn** in `nanoscope/server/` (same wheel, `server` extra), for these
reasons:

- Typed models produce an OpenAPI document, and the GUI's TypeScript client is generated
  from it.
- SSE is simple to implement.
- The same process serves the static SPA.
- Contributors already know it.

Flask has no typing or OpenAPI story, Django is too heavy, and gRPC doesn't work well in
browsers. A CI test enforces that `nanoscope/server/` imports only public library names,
and `docs/openapi.json` is committed and diffed.

**Sync vs job.** Requests that finish in under about 2 s and need no user code are
synchronous: list, read, compare, validation, parse of a graph, schemas. Everything else
returns `202` with a `Job`: training, tokenizing, downloads, uploads, benches, `describe`,
checks, and checkpoint inspection. Generation is synchronous with a timeout and runs in a
CPU inference slot of a worker (never in the API process, rule 8.1), with an LRU of loaded
models. Submitting a run whose ref is already
`done` returns that run. Library `ValueError`s pass through word for word as RFC 9457
`422` responses, because those messages are the product.

| Resource | Endpoints |
|---|---|
| Presets, models, blocks | `GET /presets[/{name}]`, `GET /models[/{ref}]`, `GET /blocks` (palette + certification), `POST /models/{ref}/describe` (job) |
| Workspace files | `GET/PUT /files/{path}` (ETag), `GET /files?glob=`, `POST /files/{path}/graph` (parse), `POST /files/{path}/graph/patch` (emit; refuses locked blocks under `guided`), `POST /files/{path}/lint` (ruff), `GET /files/events` (SSE on external edits) |
| Validation | `POST /validate/run`, `POST /validate/study` |
| Runs | `GET /runs?prefix=&state=`, `GET /runs/{ref}`, `.../metrics?since_step=`, `.../samples`, `.../blockstats`, `POST /runs`, `POST /runs/{ref}/stop`, `/resume`, `/generate`, `/inspect` (job), `GET .../checkpoints` |
| Events | `GET /runs/{ref}/events`, `GET /events?prefix=` (SSE) |
| Compare | `POST /compare {sets, baseline, metric}` returns rows, notes, curves, precision plan |
| Studies | `GET/POST /studies`, `POST /studies/{name}/run`, `/stop`, `GET .../report`, `.../bundle.zip`, `.../card` (ablation card), `POST .../card/push` (job), `POST .../preregister/preview` and `.../preregister/commit` (jobs, 11.8), `GET /git/status` |
| Learn | `GET /curricula`, `GET /curricula/{path}/{lesson}`, `POST .../start`, `POST .../check` (job), `GET /learn/progress`, `GET /learn/unlocks`, `POST /learn/unlock` (one id or all) |
| Data, hardware, jobs, meta | `GET /data`, `POST /data/{preset}/prepare` (job); `GET /hardware`, `POST /bench`; `GET /jobs`, `POST /jobs/{id}/cancel`; `GET /health`, `/version`, `/schemas/{name}` |

**SSE built from files.**

- `state` events come from `status.json`.
- `step` events come from new `metrics.jsonl` lines, at most 4 per second per run.
- `eval`, `sample`, `checkpoint` and `blockstats` events are never coalesced.
- On resume, `train()` rewrites `metrics.jsonl` truncated to the checkpoint
  (`train_loop.py:257`). The tailer detects the file shrinking or its inode changing and
  emits `reset`.

Runs started outside the API stream the same way, and server restarts lose nothing.

**Scheduling.**

- `queue.db` is SQLite in WAL mode, kept in the home volume.
- The unit of work is one seed run, or one describe/check/inspect job. Studies enqueue in
  seed-major order, as `jobs()` does today.
- Each `nanoscope worker` owns one device slot and runs every job in a fresh child process
  (`nanoscope run-job <id>`), because torch state is global to the process.
- **Leases** are renewed by the heartbeat. When a lease expires the job is requeued, which
  is safe because runs resume.
- **Admission**: a cached `probe_memory` result per (model ref, kwargs hash, preset,
  device name) is compared against free memory with `check_gpu_fits`. A job that doesn't
  fit is skipped, not failed.
- There are two lanes: interactive (single runs, checks, describe, generation) ahead of
  batch (study runs).
- Each job has a `devices_required` field, always 1 for now, so DDP (M2) can be added
  later.
- `nanoscope study --devices ...` enqueues and starts local workers. Without `--devices`
  it runs in-process as today, so notebooks need neither SQLite nor subprocesses.
- Redis, Celery and Ray are not needed for "one host, a few jobs a minute".

---

## 8. Security: running user code

Training a model the user wrote, running `describe`, and running a lesson check all
execute arbitrary Python. Three deployment models:

| Model | Trust | What it needs | Recommendation |
|---|---|---|---|
| **A. Local single-user compose** (laptop or own GPU box) | The user trusts their own code | API on loopback; token if bound elsewhere; least-privilege workers | **Design and build for this first.** |
| B. Trusted shared box (lab, classroom) | Users trust each other but make mistakes | A, plus per-user tokens and namespaces (`users/<name>/` refs), fair queue, a container per job mounting only that user's workspace | Leave room (an `owner` field on runs and jobs), build later. |
| C. Hosted, untrusted users | None | A per-job sandbox (gVisor or Kata/Firecracker), no network egress, quotas, abuse controls; a separate launcher that holds the container runtime (never give the API the Docker socket) | Non-goal until there is demand and someone to run it. |

These rules apply now, so that B and C don't need a redesign later:

1. **The API process never imports or executes user code.** Graph parsing uses only
   `ast`/libcst. The registry discovers workspace blocks and models by AST. `describe`,
   checks, training, inspection and generation of user models all go to workers.
2. **All execution goes through a `JobRunner` interface.** `SubprocessRunner` comes now.
   `ContainerRunner` comes later: one container per job, with an image, mounts and
   limits.
3. **Worker containers are hardened.** They run as non-root, with `cap_drop: [ALL]`,
   `no-new-privileges`, a read-only root filesystem (writable volumes only),
   `mem_limit`/`cpus`/`pids_limit`, a wall-clock timeout per job, and no Docker socket.
   cgroups can't limit GPU memory, so workers rely on admission plus
   `torch.cuda.set_per_process_memory_fraction`.
4. **Network**: workers need the Hub only to prepare data. `NANOSCOPE_JOBS_OFFLINE=1`
   runs jobs without egress once data is cached (default off in A, on in C).
5. **Secrets** (`HF_TOKEN`, `WANDB_API_KEY`) come from the environment only. They are
   never returned and never mounted into check or describe jobs.
6. **Auth**: no auth on 127.0.0.1. On any other address a bearer token is required
   (`NANOSCOPE_TOKEN`, generated on first start, as Jupyter does), and the server refuses
   to start without it. Recommend SSH tunnels or Tailscale over exposing it.
7. The language server (section 9.4) only reads files. It still runs in its own container
   with the workspace mounted read-only.
8. **Git can run user code.** Hooks run on commit, and `core.fsmonitor` and diff drivers
   can run on status and diff. The preregistration commit (11.8) is therefore a worker job.
   The API reads git state only with `-c core.fsmonitor=false --no-ext-diff
   --no-textconv`, or through a job.
9. **Lesson gating (6.4) is not a security boundary** and is never used as one.

---

## 9. GUI and docker-compose

### 9.1 Services and volumes

| Service | Image | Role | Profile |
|---|---|---|---|
| `api` | `nanoscope:cpu` (slim + CPU torch + server + built SPA) | FastAPI, SSE, GUI at `/`, queue schema | default |
| `worker` | `nanoscope:cpu` | `nanoscope worker --device cpu`; also generation and checks | default |
| `worker-gpu` | `nanoscope:cuda` (pinned PyTorch CUDA runtime base) | one slot per GPU x `NANOSCOPE_SLOTS`; NVIDIA reservation `count: all` | `gpu` |
| `lsp` | node + basedpyright | Python language server over WebSocket, workspace read-only | `editor-lsp` (phase 16) |
| `notebook` | `nanoscope:cpu` + marimo | marimo notebooks (plain `.py` files in the workspace) against the same home (decision 11.7) | `notebook` |

| Volume | Mount | Contents |
|---|---|---|
| `nanoscope-home` | `/nanoscope` | `runs/`, `experiments/`, `learn/`, `hardware/`, `queue.db` (back this up) |
| `nanoscope-data` | `/nanoscope/data` | token cache (FineWeb is about 4 GB), read-mostly |
| `hf-cache`, `compile-cache` | `~/.cache/huggingface`, `TORCHINDUCTOR_CACHE_DIR` | downloads; Inductor artifacts, so compile warm-up happens once per machine |
| workspace (host bind) | `/nanoscope/workspace` | the user's git repo: `models/`, `blocks/`, `studies/`, `lessons/` |

A plain `docker compose up` gives CPU only and must still train the bigram in under
2 minutes after data download. The `gpu` profile adds devices through the NVIDIA
Container Toolkit; WSL2 and Docker Desktop use the same file. The `api` image is always
CPU-only. MPS and ROCm users run `nanoscope serve` natively. Kaggle and Colab keep the
notebook + CLI path, and the GUI reads their runs after a `POST /sync/hub` job
(`HubSync.pull`).

Inside its container the API must bind `0.0.0.0`, so under rule 8.6 compose always uses
a token. The token is generated into the home volume on first start, and `docker compose
logs api` prints a one-time login URL, as Jupyter does. The host port is published on
`127.0.0.1` only. The `api` and `worker` images are the CPU image published to GHCR in
phase 11.

### 9.2 Levels mapped to surfaces

A level switch (Learn / Tinker / Research / Extend) changes which controls are visible,
never the request bodies. Hidden fields keep their level-0 defaults, and a test enforces
this.

| Level | Visible |
|---|---|
| Learn | lessons, the model graph (surface view), block palette with drag-and-drop and swaps (locked blocks shown with their unlock lesson), lesson templates, Train, live run page, baseline verdict, samples |
| Tinker | the code editor next to the graph, run form generated from specs, seeds, "duplicate and change one thing", compare, predictions |
| Research | study builder/files, budgets, live param matching, record mode (git-gated, one-click preregistration commit), queue and devices, forest plot, ablation cards, bench; offers "unlock all" once |
| Extend | the whole workspace tree, block registration and certification, curriculum authoring preview, schemas and `/docs` |

### 9.3 Key screens

1. **Lesson page**: `lesson.md` with surface/deep tabs, a "Start" button that copies the
   starter file and opens it in editor and graph, a check button with the result, and the
   compute estimate.
2. **Model page**: editor and graph side by side on one file. Depth dial. Shapes, params
   and FLOPs per node. Opaque and code-only regions are clearly marked with "edit in
   code". Certification badges. A palette to drag blocks onto layer slots, the stack and
   lesson-template slots; locked blocks are greyed out with the lesson that unlocks them
   (6.4).
3. **Run page**: state, ETA and device. Live curve with the baseline band. Per-block stats
   on the graph. Sample timeline. Stop, resume, generate, duplicate. A failed run shows its
   error from `status.json`.
4. **Compare page**: the verdict table as the library writes it, notes shown as warnings,
   a forest plot of deltas with CIs, per-seed curves, and a precision plan ("with 5 seeds
   the CI would be about ±0.008").
5. **Study builder and study page**: variant table with live matching (red when outside
   tolerance), "Save as file" (TOML), Run (explore); record mode only on a committed file
   in a clean tree. A "Commit preregistration" button shows the exact diff and message
   before committing (11.8). A variant x seed grid that fills in live, and a forest plot
   that updates as seeds complete.
8. **Components**: lock state of every block and feature, how each was unlocked, and
   "Unlock all".
6. **Inspect**: checkpoint scrubber with attention maps and logit lens.
7. **Hardware and queue**.

### 9.4 The editor (evaluating the proposed stance)

**Agreed.**

- The editor opens and saves real `.py` files in the mounted workspace. There is no copy
  in a database, so local editors, git, the CLI and CI keep working.
- It uses Monaco. Code is the source of truth, and the graph is a second view of the
  same file.
- It stays minimal: run, stop, live progress, inline errors, links to results. There is
  no terminal, no debugger and no git UI. The git elements are a read-only status line
  and the preregistration commit dialog (11.8), both of which record mode needs.

**Changes I recommend.**

1. **No full language server in the MVP.** The most useful inline errors are
   nanoscope's own, and pyright can't produce them:
   - `validate_run_request` problems (kwarg is neither a model parameter nor a preset
     field),
   - shape errors from the `describe` meta trace, mapped back to the source line,
   - failed equivalence checks,
   - "outside param-matching tolerance".

   The MVP pairs ruff diagnostics on save with completions generated from the block and
   preset registries, which are a static JSON fed to Monaco. Add basedpyright through
   `monaco-languageclient` in phase 16, as a separate `lsp` service. The WebSocket LSP
   bridge is the most fiddly piece and the least specific to nanoscope.
2. **Fall back per node, not per file.** Unknown calls become opaque nodes and the rest of
   the graph stays editable. Only statements outside the subset make a class code-only.
3. **Conflicts and external edits.** Saves use an ETag (mtime + hash), and a mismatch
   returns `409` with a diff. `GET /files/events` reloads the buffer when a host editor
   or `git checkout` changes the file.
4. **Saving never executes code.** After a save, `describe` is queued as a worker job, and
   inline shape errors arrive by SSE. This keeps rule 8.1.
5. **Notebooks stay out of the editor.** The in-stack notebook is marimo (decision 11.7),
   in the `notebook` profile, which stores notebooks as `.py` files in the workspace. The
   existing `.ipynb` notebooks may be retired once the marimo versions work (user, 2026-10-05: Kaggle/Colab compatibility is not a constraint).

### 9.5 Frontend stack, and what must not be built

The stack is React + TypeScript + Vite and TanStack Query, with a client generated from
`openapi.json`. Curves use **uPlot**, which handles live streams cheaply. The graph uses
**React Flow + elkjs** for automatic layout. The editor is **Monaco**. The forest plot is
plain SVG. Streamlit and Gradio are rejected: they run the library in-process and make
SSE-driven pages awkward, which blurs the thin-API boundary.

Do not build:

- client-side statistics or verdicts,
- study or model state that exists only in the browser or `queue.db`,
- bypasses of library refusals,
- a GUI-only feature with no CLI or function equivalent (each screen's help lists the
  equivalent command),
- free-form wiring outside the representable subset.

---

## 10. Roadmap

Effort is in focused engineering days for one person who knows the code. Every phase ends
shippable: tests green, the notebook test green, PR merged. The phases were renumbered on
2026-10-05 after the decisions in section 11:

- drag-and-drop graph editing moved into the MVP (11.5);
- lesson gating was added (6.4);
- `gpt2.py`/`modern.py` are rebuilt on blocks (11.4);
- a release-prep phase comes before compose (11.10);
- a CI baseline is added to phase 6, because the repo has no CI workflow today.

The item-level build list, with a done-check per item, is
[`checklist.md`](checklist.md).

### Phase 6: servable library + CI baseline (9-11 d)

Contents: section 4, plus a GitHub Actions workflow that runs lint, type check, the
offline tests and the first-notebook test (with cached data). **Done when:**

- A subprocess run goes `preparing → running`. `nanoscope stop` produces `cancelled` with
  a checkpoint. Resuming reaches `done`, with metrics equal to an uninterrupted run
  (extends `test_interrupted_run_resumes_exactly`).
- A model that raises ends `failed`, the error is on disk, and `nanoscope status` shows it.
- `load_run("tinystories-5min/gpt2/seed-0")` generates in a fresh process.
- `studies/m1_ablation.py` → `to_spec()` → TOML → `from_spec()` produces the same
  `jobs()`.
- All JSON from a toy study validates against the schemas, and the baselines load as v0.
- Notebooks are unchanged.
- CI is green on the phase's PR.

### Phase 7: queue and workers (5-6 d), after 6

**Done when:**

- The existing multi-device tests pass on the queue.
- A worker killed with SIGKILL mid-job has its job requeued and resumed, and the study
  completes.
- On a GPU box, `--workers-per-device 4` on `m1_ablation` is no slower than phase 5
  (manual run by the user, recorded in the PR).
- `--shard` is removed.

### Phase 8: blocks library, describe, rebuilt models (10-13 d), after 6, parallel with 7

Contents:

- the MVP blocks (bold in 5.1), including the primitives and the `Composite` base used by
  lesson templates;
- naive references moved into `nanoscope/reference/`, with the current `gpt2.py` and
  `modern.py` frozen there;
- `GPT2`/`Modern` rebuilt on blocks;
- `parse`/`emit` with set-argument and swap edits;
- `describe` + `nanoscope describe`, `BlockStats`, `checkpoint_steps`.

**Done when:**

- Every block passes its naive-reference test, and reference modules import nothing from
  `nanoscope.blocks` or `nanoscope.models`.
- The rebuilt `GPT2`/`Modern` match the frozen references' logits to 1e-5 and produce the
  same initial weights for a given seed. Run folder names and `config.json` are unchanged,
  or the change is deliberate and the baselines are re-exported.
- Round-trip property tests pass, and fixtures are byte-identical after an unedited
  round trip.
- `nanoscope describe` prints shapes, params and FLOPs whose totals equal
  `count_params`/`flops_per_token`.
- A composed model trains through `run()` unchanged.

### Phase 9: curriculum engine, lesson gating, two paths (14-17 d), after 8

Effort: 5 d engine, 3-4 d gating, 6-8 d content. Foundations now builds attention,
multi-head attention and the block from primitives. **Done when:**

- `nanoscope learn` runs Foundations and The modern block end to end on CPU.
- Each `equivalent` check fails on the starter and passes on the reference solution
  (tested in CI with the solution files).
- Foundations lesson 1 still meets the 2-minute CPU budget.
- Under policy `guided`:
  - `from nanoscope.blocks import Attention` raises with the unlock lesson and the
    escape command;
  - passing that lesson unlocks it, and `learn/unlocks.json` records the evidence;
  - `nanoscope learn unlock --all` opens everything;
  - `run(Modern)` works throughout.

### Phase 10: HTTP API (8-10 d), after 6, 7, 8, 9

**Done when:**

- An httpx test submits a Bigram run, streams it over SSE to `done`, and gets
  `POST /compare` rows equal to calling `compare()` directly.
- Graph parse and patch through `/files` round-trip a fixture.
- A CLI-started run appears in `/events`.
- A CI test asserts that the `api` process never imports workspace modules.
- The server refuses a non-loopback bind without a token.
- Under policy `guided`, a graph patch or run that uses a locked block gets a `422` that
  names the unlock lesson.

### Phase 11: release prep, PyPI + GHCR (2-3 d), after 10

Contents:

- pyproject metadata, a single-sourced version, a licence, and a test of the wheel's
  contents;
- a release workflow (PyPI trusted publishing);
- a CPU Dockerfile and a GHCR workflow.

The publish actions themselves are the user's. **Done when:**

- `uv build` produces a wheel containing the baselines, schemas, curricula and references.
- The CPU image builds in CI, and `nanoscope presets` runs in it.
- The user has published to TestPyPI, PyPI and GHCR (USER ACTION).

### Phase 12: docker-compose (3-4 d), after 11

Contents: section 9.1, including the marimo `notebook` profile. **Done when:**

- CI (CPU): `compose up`, a bigram run reaches `done` in under 2 minutes with cached data,
  and the run folder is on the home volume.
- Manual GPU run by the user: `--profile gpu` bench reports `cuda`.
- `down`/`up` keeps runs and resumes a run that was mid-training.
- Workers run non-root with the limits from 8.3.

### Phase 13: GUI MVP part 1: shell, Learn and Tinker screens (12-15 d), after 9, 10, 12

Contents: level switch, onboarding (guided or open), lesson page, run page, compare page
(verdict, forest plot, precision plan), predict-then-run, the run form generated from
specs, and the Components page. **Done when** a Playwright test on compose passes all of:

- Foundations lesson 1 goes start → train → check passed in at most 6 clicks.
- Learn and Tinker submit identical bodies when nothing was touched.
- A failed run shows its error from `status.json`.
- Unlock all on the Components page changes `learn/unlocks.json` to policy `open`.

### Phase 14: GUI MVP part 2: model page and drag-and-drop graph editing (14-17 d), after 13

Contents:

- Monaco without LSP: ruff on save, completions from the registries, nanoscope's own
  inline errors;
- the graph with a depth dial and inspector swaps;
- palette drag-and-drop onto layer slots, add or remove layers, layer patterns;
- lesson templates (slot filling with primitives);
- the locked palette;
- user blocks with certification.

**Done when** the property test covers every edit operation, and a Playwright test on
compose passes all of:

- In the modern-block lesson, swapping LayerNorm→RMSNorm in the inspector edits only that
  argument (the git diff is one line). The same swap by drag-and-drop gives the same diff.
- An external edit to the file updates the open graph.
- A locked block shows its unlock lesson and can't be dropped. After the lesson passes it
  can be dragged, without a restart.
- A certified user block passes its check job and appears in the palette without a
  restart.

### Phase 15: GUI Research (10-13 d), after 13 (14 for graph links)

Contents: study builder ↔ TOML, record gating, the one-click preregistration commit
(11.8), live study grid, ablation cards with an opt-in push to the user's Hub dataset
(11.9), report bundle, hardware/queue page, compute estimates. **Done when:**

- A builder-made `m1-ablation` spec produces the same `plan.json` as the `.py` file.
- Record mode is refused on a dirty tree, with the list of files.
- The preregistration commit shows its diff and message first, commits only the
  preregistration files, refuses when unrelated changes are present, and records its hash
  in `study.json`.
- Forest plot values equal `results.json`.

### Phase 16: Extend + depth (8-10 d), after 14

Contents: basedpyright LSP service, `optimizer=` factory (AdamW default, Muon example),
inspect page (attention maps, logit lens over `checkpoint_steps`), curriculum authoring
preview, more blocks (ALiBi, MoE) with their naive references and locks. **Done when:**

- `run(..., optimizer=...)` resumes exactly.
- The inspect page shows attention maps from the learner's own run at any archived step.

### Phase 17: remaining curriculum paths (18-26 d), after 15, 16

Contents: Honest ablations, Evaluation, Efficiency (the KV-cache lesson is the user's M4),
Interpretability, Scaling-lite and Extend. Every lesson that needs GPU hours ships a CPU
variant and a GPU variant, each with a compute estimate (11.6). **Done when:**

- Every path runs end to end in its CPU variants in CI.
- The induction-heads lesson shows a curve over checkpoints from the learner's own run.
- Every GPU variant has an estimate measured on the user's hardware (USER ACTION).

### Order, MVP slice, and risks

```
6 ─┬─> 7 ─────────────────┐
   └─> 8 ─> 9 ─────────────┴─> 10 ─> 11 ─> 12 ─> 13 ─> 14 ─┬─> 15 Research ──┐
                                                            └─> 16 Extend ────┴─> 17 paths
M4 KV cache: independent (the user's). M2 DDP: after 7. FineWeb ablation: on 7's workers.
```

**MVP slice = phases 6-14**, about **77-96 focused days** (it was 55-69 for 6-12 before
the decisions). It delivers:

- lessons with checks and gating;
- code plus a graph with drag-and-drop editing;
- live runs and verdict-first compare;
- compose on CPU or GPU, from a published package and image.

Where the extra 22-27 days come from:

| Change | Extra days |
|---|---|
| Drag-and-drop in the MVP (11.5) | 8-10 |
| Lesson gating across library, API and GUI | 5-7 |
| Rebuilding the models (11.4) | 2-3 |
| Release prep (11.10) | 2-3 |
| CI baseline | 1 |
| More Foundations content | 1-2 |

After the MVP, phases 15-17 take about 36-49 days.

| Risk | Mitigation |
|---|---|
| Scope: a visual editor and curriculum content are the two biggest sinks, and drag-and-drop is now in the MVP | Drag-and-drop only on slots, the stack and templates, all through `emit`, with a property test over every edit operation. The GUI is split into two phases. Two paths only in the MVP. Content is written as lesson files that also run from the CLI. |
| Level-0 path gets heavier | Server deps only in the extra. In-process `run()` untouched. No `unlocks.json` means no gating. The notebook test gates every phase. |
| Graph/code drift | `ast`/libcst only, property tests, code is the single artifact. |
| Queue and run folders disagree | Folders win. The queue only decides what runs next. |
| User code reachable over a network | Rules 8.1-8.9; loopback default; mandatory token. |
| Record mode inside containers (git `safe.directory`, ownership, commit identity) | Workspace is a host bind mount; covered by a phase 15 test. |
| Rebuilding `GPT2`/`Modern` changes `state_dict` keys or init order | Tests against the frozen references for keys, init and logits. If either changes: a key map in `load_run`, a schema bump, re-exported baselines. |
| Lesson gating read as security, or annoying to experts | Documented as a learning aid (6.4, rule 8.9). Default is `open` until a lesson starts. One command or button unlocks all. |
| A tool-made preregistration commit is weaker evidence | Diff shown first, pathspec-only commit, `committed_via` recorded (11.8). Treated as a risk, not a blocker. |
| GPU variants of lessons can't run in CI | CI runs CPU variants. GPU variants are checked by `gpu`-marked tests and the user's calibration runs. |
| One-person maintenance of a frontend | Few screens, generated client, Playwright in CI, no design system. |

---

## 11. Decisions (answered by the user 2026-10-05)

These replace the open questions and the recommendations that were here. Where an answer
differs from the earlier recommendation, the answer wins, and the sections above have
been updated to match.

1. **Same repo for the server and the web app: yes.** `nanoscope/server/` and `web/` live
   here, with one version and one CI.
2. **Deployment: model A first.** Local single-user compose comes first, keeping the seams
   for B and C (section 8).
3. **Persona: the learner first for the first screens** (phases 13-14). The plumbing
   (6-12) is still built for the remote-GPU researcher.
4. **`gpt2.py`/`modern.py`: rebuild them on blocks** (changed from "keep them").
   - Each block keeps a small, independent naive reference in `nanoscope/reference/`
     (importing only torch), so the equivalence tests still compare against something
     independent.
   - Today's hand-written models are frozen there as whole-model references.
   - The rebuilt classes keep their names, signatures and defaults. Tests check
     `state_dict` keys and initial weights; any change to either is deliberate and comes
     with re-exported baselines (5.1).
5. **Drag-and-drop is in the MVP** (changed). It is phase 14, and the MVP is now phases
   6-14 (section 10). Free-form DAGs remain a non-goal. Drag-and-drop works on layer
   slots, the stack and lesson templates.
6. **GPU-hour lessons are allowed, with two variants each**: a modest CPU variant and a
   bigger GPU variant, each with a compute estimate in `lesson.toml` (6.1). CI runs the
   CPU variants. The user calibrates the GPU variants.
7. **In-stack notebooks: marimo** (changed from JupyterLab). The compose profile is
   `notebook`. The existing `.ipynb` notebooks get no compatibility work. They stay as
   they are for Colab and Kaggle, and the first-notebook guard test keeps guarding
   `01-first-model.ipynb`.
   **Update 2026-10-05:** the user allows replacing the `.ipynb` notebooks with marimo; the
   guard test moves to the marimo version.
8. **Record-mode commits: the GUI may make a one-click preregistration commit**
   (changed). The design keeps the evidence sound:
   1. it shows the exact diff and commit message before committing, and the commit job
      refuses if the files changed since that preview (preview hash);
   2. it commits only the preregistration files (the study spec and any prediction
      files), by pathspec;
   3. it refuses if the tree has unrelated uncommitted changes, and lists them;
   4. it records the commit hash and `committed_via: "nanoscope"` in `study.json` and in
      every run's `config.json`;
   5. it runs as a worker job (git hooks are user code, rule 8.8), with the user's own git
      identity, and refuses if none is configured.

   **Risk, not blocker:** a tool-made commit is weaker evidence than one made by hand,
   because a reader can't tell how carefully the diff was reviewed. `committed_via`
   makes this visible in every report.
9. **Ablation cards: opt-in, in a public Hub dataset under the user's account,
   record-mode runs only.** Pushing is a job that needs `HF_TOKEN`. Creating the dataset
   is a USER ACTION.
10. **Publishing: prepare PyPI (`nanoscope-lab`) and a CPU image on GHCR in phase 11,
    before compose.** The publish actions are USER ACTIONs. **Licence: MIT** (user,
    2026-10-05).
11. **Schema policy:** readers accept N and N-1. Baselines are re-exported on every bump
    (`export_baseline`, `compare.py:299`).
12. **New requirement: lesson gating** (6.4).

**Update 2026-10-05:** the user allows breaking old checkpoints and baselines when
rebuilding `GPT2`/`Modern` (decision 4), so no key map is needed; a schema bump and
re-exported baselines are enough. The graph route for gating stays slot-filling templates.

## 12. What this plan does not change

- The shipped notebooks, the Kaggle flow, the CLI verbs and the `run()` signature, except
  for additive keywords (`optimizer=`, `checkpoint_steps=`, `block_stats=`) and the
  removal of the hidden `--shard` flag (phase 7).
- The names, constructor signatures and defaults of `Bigram`, `GPT2` and `Modern`. Their
  implementation moves onto blocks (11.4), but `run(GPT2)` and the run folders it writes
  stay the same.
- Level 0: without `learn/unlocks.json` there is no gating, so `run(Bigram)` in three
  cells stays exactly as it is.
- The deferred items keep their owners: DDP (M2) and the FineWeb-scale ablation (your
  compute call). KV cache (M4) is the Efficiency path's main lesson, and it is yours to
  build.

## 13. Reconciling the old docs

| Doc | Status | Action |
|---|---|---|
| `docs/project-nanoscope.md` | The M0-M7 research program. The milestones remain valid, but the "Repo layout" section (`configs/` YAML, `train/`, `infer/`, `eval/`, `interp/`) describes the deleted stack. | Keep as the research agenda. Rewrite "Repo layout" to the current modules, mark M0 and the M1 pipeline done (`experiments/m1-ablation/report.md`), and link each milestone to its curriculum path (6.2). Its acceptance criteria become `verdict`/`predicted` checks. |
| `docs/archive/roadmap-v2.md` (was `docs/roadmap-v2.md`) | A personal reading roadmap (B/R/S/K tiers). Not a tool doc. | Move to `docs/archive/`. Mine its tiered reading lists into `lesson.md` reading sections in phase 9. |
| `docs/m1-evaluator-plan.md` | Gitignored local file. Every module it cites (`nanoscope/model/registry.py`, `train/checkpoint.py`, YAML configs) is gone, and its goals are met by `compare.py`, `statistics.py` and `Study`. | Delete locally. It was never committed. |
| `docs/research.md` | Current. | Update in phases 6-7: `nanoscope stop`, status states, queue-based `--devices`, TOML studies. |
| `docs/dataset-card.md`, `notebooks/kaggle.ipynb` | Current. | Keep. |
| `README.md` | Describes two levels. | Phase 11: install from PyPI. After phase 12: describe four levels, add `nanoscope serve`, `docker compose up` and `nanoscope learn`. |
| This file + `plan-tool-landscape.md` | Proposal; decisions recorded in section 11. The landscape doc still uses the old phase numbers and calls questions "open". | Once phases start, fold the agreed parts into `docs/architecture.md` and keep these as the decision record. |
| `docs/checklist.md` | The build list for sections 4-10. | Tick items as they land; record scope changes there and in section 11. |
| Empty local `configs/` and `reports/` folders | Leftovers from the deleted stack (untracked). | Delete locally in phase 6. |
