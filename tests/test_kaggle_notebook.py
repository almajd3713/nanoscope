from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import nbformat
import pytest

from kaggle_sync import package_workspace, workspace_files
from nanoscope.config import load_config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-runner.ipynb"


def sources():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}


def test_notebook_is_valid_clean_and_python_compiles():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            compile(cell.source, f"{NOTEBOOK}:{cell.id}", "exec")


def test_notebook_command_interrupt_requests_graceful_stop(monkeypatch):
    received = []

    class Child:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            if not received:
                received.append("interrupt")
                raise KeyboardInterrupt
            return 0

        def send_signal(self, value):
            received.append(value)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Child())
    namespace: dict[str, Any] = {"INSTALL_DEPENDENCIES": False}
    exec(sources()["run-command"], namespace)
    namespace["run_command"]("python", "-m", "nanoscope", "train")
    assert received == ["interrupt", signal.SIGTERM]


@pytest.mark.parametrize("missing", [False, True])
def test_notebook_secrets_are_loaded_without_printing_values(monkeypatch, capsys, missing):
    config = load_config(ROOT / "configs/eval/kaggle-smoke.yaml")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: json.dumps(config.to_dict()))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    labels = []

    class Secrets:
        def get_secret(self, label):
            labels.append(label)
            if missing:
                raise RuntimeError("PRIVATE-BACKEND-DETAIL")
            return f"PRIVATE-{label}"

    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=Secrets)
    )
    namespace = {
        "subprocess": subprocess,
        "sys": sys,
        "json": json,
        "os": os,
        "TRAIN_CONFIG": "unused.yaml",
        "RESUME": "none",
    }
    if missing:
        with pytest.raises(RuntimeError, match="Enable the HF_TOKEN") as error:
            exec(sources()["load-secrets"], namespace)
        assert error.value.__suppress_context__
        assert "PRIVATE" not in str(error.value)
    else:
        exec(sources()["load-secrets"], namespace)
        assert os.environ["HF_TOKEN"] == "PRIVATE-HF_TOKEN"
        assert {"HF_TOKEN", "HF_REPO_ID", "WANDB_API_KEY"} <= set(labels)
    assert "PRIVATE" not in capsys.readouterr().out


def test_notebook_cpu_flow_uses_real_workspace_and_exports_results(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    code = sources()
    files = workspace_files(ROOT)
    assert Path("notebooks/kaggle-runner.ipynb") not in files
    assert Path("configs/eval/kaggle-smoke.yaml") in files
    archive = tmp_path / "workspace.zip"
    package_workspace(ROOT, files, archive)
    namespace = {}
    exec(code["parameters"], namespace)
    namespace.update(
        WORKSPACE_ZIP=archive,
        PROJECT_BASE=tmp_path / "projects",
        EXPORT_DIR=tmp_path / "exports",
        TRAIN_CONFIG="configs/eval/local-base.yaml",
        EVAL_CONFIG="configs/eval/local-validation.yaml",
        INSTALL_DEPENDENCIES=False,
    )
    for cell in (
        "extract-workspace",
        "run-command",
        "load-secrets",
        "prepare-corpus",
        "train",
        "export-results",
    ):
        exec(code[cell], namespace)
    result_zip = tmp_path / "exports/eval-local-base-results.zip"
    with zipfile.ZipFile(result_zip) as result:
        names = result.namelist()
        evaluations = [name for name in names if "/evaluations/" in name]
        assert len(evaluations) == 2
        assert {json.loads(result.read(name))["checkpoint"]["step"] for name in evaluations} == {
            2,
            4,
        }
        assert all(json.loads(result.read(name))["complete"] for name in evaluations)
        assert "runs/eval-local-base/workspace-provenance.json" in names
        assert not any(name.endswith(".pt") or ".env" in name for name in names)
    # Re-running extraction reuses the same complete archive rather than overlaying source.
    project = namespace["PROJECT"]
    exec(code["extract-workspace"], namespace)
    assert namespace["PROJECT"] == project


def test_notebook_rejects_archive_paths_outside_project(tmp_path):
    code = sources()
    namespace = {}
    exec(code["parameters"], namespace)
    archive = tmp_path / "workspace.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "bad")
    namespace.update(WORKSPACE_ZIP=archive, PROJECT_BASE=tmp_path / "projects")
    with pytest.raises(ValueError, match="outside the project"):
        exec(code["extract-workspace"], namespace)
    assert not (tmp_path / "projects/outside.txt").exists()
