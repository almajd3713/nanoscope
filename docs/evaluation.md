# Evaluate and compare model increments

New to the workflow? Start with the [hands-on walkthrough](evaluation-quickstart.md)
for a runnable example, commented templates, and how to interpret your report.

The evaluator scores saved checkpoints on an immutable held-out token corpus.
It reports next-token cross-entropy in nats/token and perplexity, excluding z-loss
and other auxiliary training penalties. The comparison command produces JSON,
CSV, Markdown, and optional PNG/SVG plots at a declared training budget.

Validation can run periodically inside training or independently on saved checkpoints.
Multi-seed reports include paired differences and 95% confidence intervals. One-
and two-seed studies remain explicitly exploratory.

## Try the complete workflow offline

Run from the repository root. Install plotting support if you want figures:

```bash
uv sync --extra dev --extra eval
uv run nanoscope prepare-eval --config configs/eval/local-corpus.yaml

for variant in base variant; do
  uv run nanoscope train --config "configs/eval/local-${variant}.yaml" --resume none
  for step in 00000002 00000004; do
    uv run nanoscope evaluate \
      --checkpoint "runs/eval-local-${variant}/checkpoints/step_${step}" \
      --eval-config configs/eval/local-validation.yaml
  done
done

uv run nanoscope compare --study configs/eval/local-study.yaml \
  --output runs/eval-comparison --plots
```

Open `runs/eval-comparison/report.md`, or inspect the CSV and figures alongside
it. These toy-model runs test the workflow; they are not M1 architecture results.
Training uses the existing unique-run-ID/resume rules. Repeating an identical
corpus preparation or checkpoint evaluation reuses its complete artifact.
Omit `--plots` to skip figures. Three-seed confidence intervals still need the
`eval` extra for SciPy.

## Prepare research data before training

Use `configs/eval/m1-corpus.yaml` as the FineWeb-Edu starting recipe. Choose the
context length first; the supplied 128-token context and 8,192 sequences score
1,048,576 prediction tokens. The dataset revision is pinned. Preparation makes one
unshuffled pass, bounded by `max_documents`, and fails if it cannot collect enough
unique held-out text. It never cycles a small corpus to fill the budget.

```bash
uv run nanoscope prepare-eval --config configs/eval/m1-corpus.yaml
```

The result contains `tokens.npy` and a checksummed `manifest.json`. Set
`corpus_hash` in `configs/eval/m1-validation.yaml` to the printed fingerprint to
pin the intended corpus explicitly. Changing corpus recipes requires a different
output directory; an existing artifact cannot be silently replaced.

Add the matching partition to every new M1 training config:

```yaml
data:
  # Keep the same dataset, revision, text field, and context as the corpus recipe.
  partition:
    salt: nanoscope-m1-v1
    buckets: 10000
    validation_buckets: 100
    test_buckets: 100
    split: train
```

Hashing exact text assigns documents to train, validation, or test independently
of the training seed. The training loader filters before shuffling. Its new
partitioned stream checkpoints the shuffle buffer, cursor, and RNG. Exact duplicate
text cannot cross partitions; this does not detect near duplicates or semantic
overlap. The partition specification is part of the training config digest, so
turning it on requires a new run. Existing configs without it retain their old
digests and loader path.

The corpus uses the same packing convention as training: concatenate documents
with EOS, take disjoint blocks of `context_length + 1`, and predict each token
after the first. Batch size changes grouping only, never tokens or context.
The manifest records discarded tail tokens and tokenization identity.

For a final test corpus, copy the corpus recipe, select `partition.split: test`,
and choose a different output directory. Reserve its evaluation for final models.

## Score a checkpoint

```bash
uv run nanoscope evaluate \
  --checkpoint runs/MY-RUN/checkpoints/step_00001000 \
  --eval-config configs/eval/m1-validation.yaml
```

New checkpoints embed their resolved training config. Legacy checkpoints use the
adjacent `resolved-config.yaml`, or an explicit `--training-config`; its digest
must match checkpoint metadata. Legacy runs without verified partition/tokenizer
provenance can be scored diagnostically but cannot enter strict comparisons.

Evaluation reconstructs the registered model and loads weights strictly, without
resuming the optimizer or dataloader. A DDP checkpoint can be evaluated on a single
device. Keep the model implementation available in the checkout. Scores record
checkpoint checksums, model/config/seed, dataset and tokenizer identity, evaluator
version and source hashes, training budget, and hardware/software provenance.

Use quoted `@/` paths in config files to start at the project root:

```yaml
corpus: "@/runs/eval-corpora/my-validation"
output: "@/runs/evaluations"
# In a study variant:
# results: ["@/runs/my-base-seed-*/evaluations/*.json"]
# In a training config:
# run:
#   output_dir: "@/runs"
```

The root is the nearest parent of the YAML containing `pyproject.toml` or `.git`.
This works in source exports without Git metadata. `"@"` alone means the root.
A config outside a marked project must use ordinary relative or absolute paths;
there is no fallback to the current working directory. Quote `@` values because
YAML reserves an unquoted leading `@`.

Ordinary evaluation/study paths remain relative to the YAML. Ordinary training
`run.output_dir` paths remain relative to the current working directory. Absolute
paths continue to work. The alias applies to these YAML path values, not shell
arguments, model names, dataset IDs, or HF repository IDs. Results go to
`<evaluation output>/<run ID>/<result ID>.json`. Copy that directory and the frozen
corpus off ephemeral machines explicitly when using standalone evaluation. Periodic
results are included in the checkpoint uploads described below. Resolved model
configs are embedded in checkpoint state.

FP32 forward evaluation is the reference profile. FP16 forward evaluation is an
explicit CUDA option; loss computation still uses FP32 logits and FP64 accumulation.
Lower `batch_size` for memory pressure. An error never silently changes precision
or truncates the corpus. Compare results using the same scoring profile.

## Validate during training

Pass the same evaluation config to training:

```bash
uv run nanoscope train --config configs/eval/local-base.yaml \
  --eval-config configs/eval/local-validation.yaml --resume auto
```

Evaluation settings are separate from the training recipe/digest:

```yaml
# Alongside corpus, output, batch_size, device, precision, and optional corpus_hash:
every_steps: 100
final: true
max_seconds: 300
```

`every_steps` counts completed optimizer attempts; `final` additionally scores the
configured final step. Periodic validation uses the training device and requires a
validation corpus with matching training exclusion, source, tokenizer and context.
Test corpora are rejected. Standalone evaluation ignores the cadence settings.
The time limit is checked between batches and after the last batch; it cannot
interrupt a blocked forward. It must be below 540 seconds to leave headroom within
the ten-minute distributed timeout. Use standalone scoring for longer evaluations.

Only rank zero evaluates, using the unwrapped model. Evaluation restores RNG,
module training flags and registered buffers. Each due evaluation saves a recovery
checkpoint first, then attaches complete results through an atomic manifest update.
Errors leave that checkpoint resumable; stop signals cancel evaluation between
batches without publishing partial results. Resuming retries a due score missing
from the restored checkpoint before taking another training step. An explicit
`--stop-after-step` saves immediately and defers validation until resume.

Periodic artifacts always live at `<run_dir>/evaluations/<result_id>.json` and
`<run_dir>/evaluations.jsonl`; the config's `output` applies to standalone scoring.
The complete result history and scheduling changes travel with checkpoint state
and checksummed sidecars, including existing HF uploads. Resume regenerates local
artifacts even when `--eval-config` is omitted (which disables new evaluations).
Rolling back to an older checkpoint moves abandoned results into `evaluations/orphaned`
and future checkpoints into hidden `.superseded-*` directories. Narrow study globs
to `evaluations/*.json` so abandoned outputs are excluded.

An evaluation interval creates a checkpoint even if normal checkpointing is less
frequent. Existing archive and retention settings still govern model-weight retention;
result history survives pruning. W&B receives separate `eval/*` metrics.

## Declare the comparison

Copy `configs/eval/local-study.yaml` and list the baseline and increments in their
intended order. Each `results` entry can be a JSON path, directory, or glob:

```yaml
name: M1 incremental architecture study
baseline: gpt2
axis: tokens
budget: 10000000
tolerance: 0.01
allow_changes:
  - model
variants:
  - name: gpt2
    results: ["@/runs/evaluations/gpt2-seed-*/*.json"]
  - name: rope
    results: ["@/runs/evaluations/rope-seed-*/*.json"]
```

For periodic results, use paths such as
`"@/runs/gpt2-seed-*/evaluations/*.json"`.

Use one run per seed per variant, with identical seed sets. Results for successive
checkpoints of one run form its curve. If multiple evaluations exist for one step
(for example, different precisions or evaluator versions), narrow the paths to
the intended result; the command rejects ambiguous inputs. Changes other than
those explicitly listed in `allow_changes` cause an error. Seeds within a single
variant must share the same model and training recipe.

The headline row uses the latest evaluated checkpoint at or below the budget,
within the relative tolerance. For example, a 10M-token budget with tolerance 0.01
accepts checkpoints from 9.9M through 10M tokens. The command does not select each
run's best validation score, interpolate missing endpoint scores, or extrapolate.
Keep the checkpoints needed for comparisons using the existing archive interval.

For matched estimated compute, set `axis: flops`, provide a FLOP budget, and set an
explicit `parameter_tolerance` (for example, 0.01 for 1% non-embedding parameter
count tolerance). The initial supported estimator is the existing approximate
`6 * non_embedding_parameters * training_tokens`. Unknown/custom estimator
identities cannot enter a strict FLOP comparison. This estimate omits important
architecture-dependent costs; use token comparisons while developing M1 and
add a validated architecture-aware estimator before claiming precise compute
matching. Both total and non-embedding parameter counts appear in the CSV.

Negative cross-entropy deltas mean improvement. `delta_baseline` compares the
same seed against the baseline; `delta_previous` compares against the preceding
variant in the manifest. Cumulative additions measure contributions conditional
on earlier additions. Leave-one-out studies can use the same interface with the
full modern model as baseline.

Throughput and memory columns describe training, not evaluation. Hardware, worker
count, precision, and timing mismatches mark speed comparisons incompatible.
New `training-step-v2` throughput measures the training step, including data loading
and gradient synchronization, while excluding evaluation, logging and checkpoint
work. GPU peaks are reset before each training step; evaluation has separate timing
and memory fields. CUDA reserved memory can include allocator cache left by validation;
evaluation peaks include resident training state. Legacy results retain their original
timing label. Cumulative
`train_seconds` is unknown when resuming a checkpoint without that measurement.

Reports retain individual seed rows and add `paired-summary.csv`. For each variant,
the report subtracts the same seed's baseline (and preceding increment) loss, then
reports the mean and sample standard deviation of those differences. With at least
three pairs and nonzero observed variation, the 95% interval is
`mean ± t(0.975, n−1) × sample_sd / sqrt(n)`, using
[SciPy's Student-t quantile](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).
Three seeds is a minimum reporting threshold, not a guarantee of a reliable interval.
The method assumes independent seed replicates and approximately normal paired
seed differences. Zero observed variation is marked degenerate and gets no interval.
An interval containing zero does not establish equivalence.

Plots show individual seeds, seed means and pointwise 95% bands only at exact
budgets measured for every seed in a variant. They do not fill missing seed scores
or treat checkpoints/tokens as independent replicates. Intervals describe training
seed variability on this fixed corpus, not uncertainty across new datasets, and
are not adjusted for multiple variants or repeated inspection of the curves.

## Validation

```bash
uv run pytest tests/test_evaluation.py tests/test_periodic_evaluation.py tests/test_eval_statistics.py
uv run pytest -m "not gpu and not network and not cloud"
uv run ruff check .
uv run pyright
```

Tests cover known probability distributions, target shifting, uneven batches,
auxiliary-loss independence, partition exclusion and buffered resume, checksum
failures, legacy checkpoints, fair-budget selection, and report generation.
