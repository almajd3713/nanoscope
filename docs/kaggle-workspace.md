# Sync a workspace with the Kaggle CLI

Use the CLI to upload source, then run your permanent Kaggle notebook; no generated notebook or disposable GPU run is
needed. The notebook retains its own GPU, Internet, and secret settings.

## One-time setup

The helper needs Python 3.10+, Git, and the `kaggle` executable on PATH. It uses
only Python's standard library and does not need the training environment.
If the CLI is missing:

```bash
uv tool install kaggle
```

Configure your Kaggle API credentials using the installed CLI's authentication
options. For compatibility with the existing 1.7 CLI, create a legacy API key in
Kaggle account settings and store the downloaded `kaggle.json` in
`~/.kaggle/kaggle.json`. Verify access with this read-only command:

```bash
kaggle datasets files redhouanelazib/vscode-run-nanoscope
```

The existing workspace dataset is configured in `kaggle-sync.json`:

```json
{
  "dataset": "redhouanelazib/vscode-run-nanoscope"
}
```

Keep it private and attached to your permanent runner. The helper only versions
an existing dataset. It never creates or renames a dataset, changes its visibility,
or deletes old versions. An inaccessible dataset fails the command.

Kaggle API credentials authorize local uploads. HF/W&B credentials remain in
Kaggle notebook secrets and are loaded by `kaggle_run.py` during training.

## Daily workflow

Save edits and, for reproducible experiments, commit them. From the workspace:

```bash
# Offline preview: exact file list, compressed size, and ZIP hash.
python3 kaggle_sync.py --dry-run

# Upload a new version of the same dataset.
python3 kaggle_sync.py -m "Update model experiment"

# The upload command can return before Kaggle finishes processing.
kaggle datasets status redhouanelazib/vscode-run-nanoscope
```

Open the permanent runner, update its attached dataset to the new version, and
select **Save Version → Save & Run All**. Its existing `workspace.zip` extraction
cell continues to work. For DDP, select T4 x2 and use:

```python
!python kaggle_run.py --config configs/m0/kaggle-ddp.yaml --resume none
```

Replace the config path with your own experiment. After a session ends, use the
appropriate `hf://OWNER/REPOSITORY/runs/RUN_ID` resume URI. Dataset sync does not
upload checkpoints or training logs, and it does not start the notebook.

## What is uploaded

The helper selects saved working-tree files, including modified tracked files
and eligible untracked files. It uses Git's own exclude rules: standard Git
ignores (including nested `.gitignore`, global ignores, and `.git/info/exclude`)
plus root `.kaggleignore`. Kaggle exclusions apply even to tracked files.
Patterns use Git syntax, including `!` exceptions. Inspect the preview after changing these files.

Credential filenames (`.env*`, `kaggle.json`, `credentials*.json`, key/token files)
and `.git`/`.kaggle` paths are always excluded. Symlinks, submodules, and deleted
files are skipped. Merge conflicts must be resolved first. Other local files
can be excluded in `.kaggleignore`. The helper and its target config are excluded
from the training ZIP too.

The staging directory is temporary and contains only `workspace.zip` and the
existing dataset metadata downloaded by the CLI. ZIP ordering and timestamps
are stable, so its SHA-256 identifies the packaged content and executable bits.
An omitted `-m` uses that hash in the version notes. Uploading unchanged content
still requests a new dataset version; the helper does not deduplicate remotely.

Underneath, the two [official Kaggle CLI commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) are:

```bash
kaggle datasets metadata OWNER/SLUG -p STAGING_DIRECTORY
kaggle datasets version -p STAGING_DIRECTORY -m "Version notes"
```

Use a staging directory containing the filtered ZIP when running these commands
manually; passing the repository directly does not apply our `.kaggleignore`
workflow. A different existing dataset can be selected without editing the file:

```bash
python3 kaggle_sync.py --dataset OWNER/ANOTHER-SLUG --dry-run
```
