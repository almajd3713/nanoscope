# Use a permanent Kaggle notebook

Import [notebooks/kaggle-runner.ipynb](../notebooks/kaggle-runner.ipynb) into a Kaggle
notebook once and keep using that notebook. The CLI uploads workspace dataset
versions; you control the notebook's GPU, Internet, attached data and enabled
secrets through Kaggle. This replaces the stale instructions for a Python bootstrap
launcher. There is no CLI kernel submission or secret attachment in this workflow.

## First run

1. Locally, commit your code/configs and run `python3 kaggle_sync.py --dry-run`, then
   `python3 kaggle_sync.py -m "Update experiment workspace"`.
2. In Kaggle, import the example notebook, attach your private workspace dataset,
   select its current version, enable Internet, and choose your GPU(s).
3. Create and enable `HF_TOKEN`, `HF_REPO_ID`, and `WANDB_API_KEY` for this notebook.
   Use an existing private HF model repository with token write access.
   Optionally enable `WANDB_ENTITY`. Keep secret values out of cells and configs.
4. Run the example cells in order with their defaults. The four-step smoke model
   uses fixture text, all visible GPUs, HF persistence, W&B, and periodic evaluation.
5. Verify the HF recovery checkpoint and W&B run, then download the result ZIP
   produced by the notebook's last code cell.

The notebook uses Kaggle's
[`UserSecretsClient.get_secret`](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py)
to retrieve values already enabled for that notebook. It loads only the labels
needed by the selected config and prints no secret values. A disabled/missing
required secret stops setup before training.

The default parameters are:

```python
TRAIN_CONFIG = "configs/eval/kaggle-smoke.yaml"
CORPUS_CONFIG = "configs/eval/local-corpus.yaml"
EVAL_CONFIG = "configs/eval/kaggle-validation.yaml"
RESUME = "auto"
```

`WORKSPACE_INPUT = None` searches the extracted input directories under
`/kaggle/input` for a project containing `pyproject.toml`, `nanoscope/`, and `configs/`.
If more than one matches, set `WORKSPACE_INPUT` to the exact project directory,
including any nested `workspace/` folder. No archive extraction is needed.

The notebook copies the selected directory to
`/kaggle/working/nanoscope-workspaces/<tree-sha256>/` so package installation,
config edits and `@/runs` artifacts have a writable location. It leaves the input
dataset untouched. The hash covers relative filenames and file contents, not the
ZIP or mount location; it is printed and included in the result export. Updated
source gets a separate working copy, preventing stale files from a prior version.
The notebook itself is excluded from workspace sync; import it separately when
deliberately updating runner cells.

The environment installs the hashed Kaggle requirements and Nanoscope with
`--no-deps`, preserving the image's PyTorch. Training runs in a fresh subprocess.
Comparison libraries are installed locally, where you create the reports.

## Use your own model

Copy and edit the experiment/corpus/evaluation templates described in the
[evaluation walkthrough](evaluation-quickstart.md). Then change the notebook parameters:

```python
TRAIN_CONFIG = "configs/my-base-seed-1337.yaml"
CORPUS_CONFIG = "configs/my-corpus.yaml"
EVAL_CONFIG = "configs/my-eval.yaml"
RESUME = "none"
```

The research templates use FineWeb/GPT-2 tokenization, whereas the smoke uses byte
fixture text. Switch all three configs together. Give every variant/seed a unique
run ID; keep training settings and evaluation corpus/profile matched for comparisons.
Keep training `run.output_dir: /kaggle/working/nanoscope-runs` for a stable output root.

`@/` refers to the writable project copy. Corpus preparation can write beneath
that root, or you can attach an already-prepared corpus and set its absolute path
in the evaluation config. For an attachment set `CORPUS_CONFIG = None`. Pin
`corpus_hash` to the expected fingerprint for every research variant. Workspace
sync excludes local `runs/`; it does not transfer prepared corpora automatically.

Changed input source gets a different working-copy root. The notebook can rebuild the
corpus from its pinned recipe and verify the expected fingerprint. Preserve the
frozen corpus separately to avoid preparation work in every new session.

## Resume and stopping

- Same session: keep the training config and use `RESUME = "auto"`. You can disable
  dependency installation once it has succeeded in that session.
- Fresh session: select the same configs, set
  `RESUME = "hf://OWNER/REPOSITORY/runs/RUN_ID"`, and run the notebook in order.
  Restore/rebuild the matching corpus before training; keep worker count and
  training recipe compatible. Evaluation history is recovered with the checkpoint.
- Planned interruption: set `STOP_AFTER_STEP = 2` (or another step), then clear it
  on resume. This preserves the original `max_steps` and schedule. Evaluation at
  the stop step is deferred until resume.
- A caught notebook interrupt asks the CLI to stop and checkpoint. A forced kernel
  or session termination may prevent a final save; recover the last completed
  remote checkpoint. Do not assume the UI stop control always permits cleanup.

For an M0 config without held-out partitioning, set both `EVAL_CONFIG = None` and
`CORPUS_CONFIG = None`. Adding partitioning to an existing run requires a new run.

## Compare locally

The export cell writes one ZIP under `/kaggle/working/nanoscope-exports/` containing:

```text
runs/<run-id>/evaluations/*.json
runs/<run-id>/evaluations.jsonl
runs/<run-id>/metrics.jsonl
runs/<run-id>/resolved-config.yaml
runs/<run-id>/workspace-provenance.json
```

Download each run's ZIP through the notebook output file browser and extract into
your local project root. These archives contain scores/configs, not weights or
credentials. A run stopped before evaluation has no scores to compare yet.
Then run:

```bash
uv sync --extra eval
uv run nanoscope compare --study configs/my-study.yaml --output runs/my-comparison --plots
```

The study template's `@/runs/<variant>-seed-*/evaluations/*.json` paths now match the
exported layout. Adjust run-ID patterns and the training token budget. One smoke run
alone cannot form a comparison: train and export at least two variants. Use matched
seed sets for paired statistics. HF recovery history is separate from this manual
download flow; a CLI for collecting remote study results is not implemented.

Notebook schema, cell orchestration and CPU artifact flow are checked locally.
Live Kaggle GPU/HF/W&B acceptance still requires running the smoke notebook.
