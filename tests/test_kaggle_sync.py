import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import kaggle_sync


def repository(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".venv/\n*.log\n")
    (tmp_path / ".kaggleignore").write_text("/docs/\n*.pt\n!keep.pt\n")
    return tmp_path.resolve()


def test_filters_tracked_untracked_and_nested_ignores(tmp_path):
    root = repository(tmp_path)
    contents = {
        "main.py": "print('ready')",
        "new config.yaml": "seed: 1337",
        "docs/guide.md": "development only",
        "weights.pt": "not uploaded",
        "keep.pt": "explicit exception",
        "nested/.gitignore": "private.txt\n",
        "nested/private.txt": "ignored by nested rules",
        "nested/public.txt": "included",
        ".venv/large.bin": "not uploaded",
        "debug.log": "not uploaded",
        ".env": "dummy secret",
        ".kaggle/access_token": "dummy secret",
        "kaggle.json": "dummy secret",
        "credentials-test.json": "dummy secret",
        "client.pem": "dummy secret",
        "deleted.py": "removed from working tree",
    }
    for name, value in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    subprocess.run(["git", "-C", str(root), "add", "-f", "."], check=True)
    (root / "deleted.py").unlink()
    (root / "untracked.py").write_text("new model")
    (root / "link.py").symlink_to(root / ".env")
    names = {path.as_posix() for path in kaggle_sync.workspace_files(root)}
    assert names == {
        ".gitignore",
        ".kaggleignore",
        "main.py",
        "new config.yaml",
        "keep.pt",
        "nested/.gitignore",
        "nested/public.txt",
        "untracked.py",
    }


def test_archive_contains_current_contents_and_is_deterministic(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "main.py").write_text("old content")
    subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
    (root / "main.py").write_text("current content")
    (root / "odd\nname.txt").write_text("newline name")
    files = kaggle_sync.workspace_files(root)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    assert kaggle_sync.package_workspace(root, files, first) == kaggle_sync.package_workspace(
        root, files, second
    )
    with zipfile.ZipFile(first) as archive:
        assert archive.read("main.py") == b"current content"
        assert archive.read("odd\nname.txt") == b"newline name"
        assert set(archive.namelist()) == {path.as_posix() for path in files}


def test_preview_makes_no_upload_call(tmp_path, monkeypatch, capsys):
    root = repository(tmp_path)

    def unexpected(*args):
        pytest.fail("preview must not contact Kaggle")

    monkeypatch.setattr(kaggle_sync, "upload_dataset", unexpected)
    kaggle_sync.main(["--root", str(root), "--dataset", "owner/workspace", "--dry-run"])
    assert "no Kaggle requests" in capsys.readouterr().out


def test_upload_fetches_metadata_and_only_versions_existing_dataset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(kaggle_sync.shutil, "which", lambda _: "/bin/kaggle")

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["check"]
        if arguments[2] == "metadata":
            (tmp_path / "dataset-metadata.json").write_text(json.dumps({"id": "owner/workspace"}))

    monkeypatch.setattr(kaggle_sync.subprocess, "run", run)
    kaggle_sync.upload_dataset("owner/workspace", tmp_path, "updated model")
    assert calls == [
        ["/bin/kaggle", "datasets", "metadata", "owner/workspace", "-p", str(tmp_path)],
        ["/bin/kaggle", "datasets", "version", "-p", str(tmp_path), "-m", "updated model"],
    ]


def test_mismatched_metadata_blocks_upload(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(kaggle_sync.shutil, "which", lambda _: "/bin/kaggle")

    def run(arguments, **kwargs):
        calls.append(arguments)
        (tmp_path / "dataset-metadata.json").write_text(json.dumps({"id": "wrong/dataset"}))

    monkeypatch.setattr(kaggle_sync.subprocess, "run", run)
    with pytest.raises(ValueError, match="does not match"):
        kaggle_sync.upload_dataset("owner/workspace", tmp_path, "updated model")
    assert len(calls) == 1
