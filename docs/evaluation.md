# Evaluate and compare model increments

The evaluator scores saved checkpoints on an immutable held-out token corpus.
It reports next-token cross-entropy in nats/token and perplexity, excluding z-loss
and other auxiliary training penalties. The comparison command produces JSON,
CSV, Markdown, and optional PNG/SVG plots at a declared training budget.

This release provides standalone checkpoint evaluation. Periodic validation inside
the trainer and multi-seed confidence intervals are not yet implemented. Multiple
seeds can be compared as individual paired rows; the report does not invent an
uncertainty estimate from one run.

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
Omit `--plots` to produce tables without installing the plotting dependency.

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

Evaluation/study paths are relative to the YAML file; the existing training
`run.output_dir` remains relative to the current working directory. Results go to
`<evaluation output>/<run ID>/<result ID>.json`. Copy that directory and the frozen
corpus off ephemeral machines explicitly; this release does not upload evaluation
sidecars to HF automatically. Resolved model configs are included in the existing
checkpoint upload because they are embedded in checkpoint state.

FP32 forward evaluation is the reference profile. FP16 forward evaluation is an
explicit CUDA option; loss computation still uses FP32 logits and FP64 accumulation.
Lower `batch_size` for memory pressure. An error never silently changes precision
or truncates the corpus. Compare results using the same scoring profile.

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
    results: [../../runs/evaluations/gpt2-seed-*/*.json]
  - name: rope
    results: [../../runs/evaluations/rope-seed-*/*.json]
```

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
Existing throughput uses inter-step wall time and can include logging/checkpoint
overhead; treat it as a diagnostic measurement. Evaluation time is recorded
separately and never changes the saved training measurements.

## Validation

```bash
uv run pytest tests/test_evaluation.py
uv run pytest -m "not gpu and not network and not cloud"
uv run ruff check .
uv run pyright
```

Tests cover known probability distributions, target shifting, uneven batches,
auxiliary-loss independence, partition exclusion and buffered resume, checksum
failures, legacy checkpoints, fair-budget selection, and report generation.
