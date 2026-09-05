"""Package a Git workspace and version its existing Kaggle dataset using the CLI."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=True
    ).stdout


def excluded_path(path: Path) -> bool:
    """Credential and Git metadata exclusions cannot be negated by ignore rules."""
    patterns = (".env*", "kaggle.json", "credentials*.json", "*.key", "*.pem", "*.p12", "*.token")
    return any(part in {".git", ".kaggle"} for part in path.parts) or any(
        fnmatch.fnmatchcase(path.name.lower(), pattern) for pattern in patterns
    )


def workspace_files(root: Path) -> list[Path]:
    if Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root:
        raise ValueError("--root must be the Git repository root")
    if git(root, "ls-files", "--unmerged", "-z"):
        raise ValueError("resolve Git merge conflicts before uploading")
    exclusions = ["--exclude-standard"]
    if (root / ".kaggleignore").exists():
        exclusions.append("--exclude-from=.kaggleignore")

    def paths(*arguments: str) -> set[Path]:
        return {
            Path(os.fsdecode(name))
            for name in git(root, "ls-files", "-z", *arguments, *exclusions).split(b"\0")
            if name
        }

    # --cached includes tracked files even when ignored: explicitly subtract them.
    candidates = paths("--cached", "--others") - paths("--cached", "--ignored")
    selected = []
    for relative in sorted(candidates):
        if relative.is_absolute() or ".." in relative.parts or excluded_path(relative):
            continue
        full = root / relative
        # Do not follow links, including tracked files under a replaced directory.
        if any(item.is_symlink() for item in (full, *full.parents) if item != root):
            continue
        if full.is_file():
            selected.append(relative)
    if not selected:
        raise ValueError("no workspace files remain after applying exclusions")
    return selected


def package_workspace(root: Path, files: list[Path], archive: Path) -> str:
    """Stable ordering/timestamps make the SHA identify file content and executable bits."""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for relative in files:
            source = root / relative
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError(f"workspace file became a link or escaped the root: {relative}")
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100755 if source.stat().st_mode & 0o111 else 0o100644) << 16
            with source.open("rb") as reader, output.open(info, "w", force_zip64=True) as writer:
                shutil.copyfileobj(reader, writer)
    digest = hashlib.sha256()
    with archive.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def upload_dataset(dataset: str, directory: Path, message: str = "") -> None:
    cli = shutil.which("kaggle")
    if cli is None:
        raise RuntimeError("Kaggle CLI not found; install it with: uv tool install kaggle")
    # Fetch existing metadata: no guessed title/license, no implicit dataset creation.
    subprocess.run([cli, "datasets", "metadata", dataset, "-p", str(directory)], check=True)
    metadata = json.loads((directory / "dataset-metadata.json").read_text(encoding="utf-8"))
    if str(metadata.get("id", "")).lower() != dataset.lower():
        raise ValueError("downloaded dataset metadata does not match the configured dataset ID")
    subprocess.run([cli, "datasets", "version", "-p", str(directory), "-m", message], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", help="owner/slug; defaults to kaggle-sync.json in the root")
    parser.add_argument("-m", "--message", help="version notes; defaults to the workspace SHA-256")
    parser.add_argument("--dry-run", action="store_true", help="preview files/ZIP without network")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        dataset = args.dataset
        if dataset is None:
            dataset = json.loads((root / "kaggle-sync.json").read_text(encoding="utf-8"))["dataset"]
        if not isinstance(dataset, str) or not re.fullmatch(r"[\w-]+/[\w-]+", dataset, re.ASCII):
            raise ValueError("dataset must be an owner/slug identifier")
        files = workspace_files(root)
        with tempfile.TemporaryDirectory(prefix="kaggle-workspace-") as temporary:
            directory = Path(temporary)
            archive = directory / "workspace.zip"
            digest = package_workspace(root, files, archive)
            print(f"Dataset: {dataset}", flush=True)
            if args.dry_run:
                for path in files:
                    print(f"  {path.as_posix()!r}")
            print(
                f"workspace.zip: {len(files)} files, {archive.stat().st_size:,} bytes", flush=True
            )
            print(f"SHA-256: {digest}", flush=True)
            if args.dry_run:
                print("Preview only; no Kaggle requests made.")
                return
            upload_dataset(dataset, directory, args.message or f"Workspace sha256:{digest}")
        print(f"Version submitted: https://www.kaggle.com/datasets/{dataset}")
        print(f"Check processing status: kaggle datasets status {dataset}")
        print("Update the permanent notebook's attached dataset version, then Save & Run All.")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Workspace sync failed: {exc}") from exc


if __name__ == "__main__":
    main()
