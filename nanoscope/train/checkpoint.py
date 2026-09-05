from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest["schema_version"]) != SCHEMA_VERSION:
            return False
        return all(
            (path / name).is_file() and _sha256(path / name) == expected
            for name, expected in manifest["files"].items()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


class CheckpointManager:
    def __init__(self, run_dir: Path, keep_last: int) -> None:
        self.run_dir = run_dir
        self.root = run_dir / "checkpoints"
        self.keep_last = keep_last
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self, step: int, state: dict[str, Any], metadata: dict[str, Any], archive: bool = False
    ) -> Path:
        destination = self.root / f"step_{step:08d}"
        if destination.exists() and validate_checkpoint(destination):
            self._write_latest(destination)
            return destination
        if destination.exists():
            quarantine = self.root / f".corrupt-{destination.name}-{uuid.uuid4().hex}"
            os.replace(destination, quarantine)

        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.root))
        try:
            torch.save(state, temporary / "state.pt")
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            files = {
                name: _sha256(temporary / name) for name in ("state.pt", "metadata.json")
            }
            (temporary / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "step": step,
                        "archive": archive,
                        "files": files,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._write_latest(destination)
        self._prune()
        return destination

    def _write_latest(self, path: Path) -> None:
        temporary = self.run_dir / ".latest.json.tmp"
        temporary.write_text(
            json.dumps({"path": str(path.relative_to(self.run_dir))}), encoding="utf-8"
        )
        os.replace(temporary, self.run_dir / "latest.json")

    def _prune(self) -> None:
        recovery = []
        for path in self.valid_checkpoints():
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if not manifest.get("archive", False):
                recovery.append(path)
        for path in recovery[: -self.keep_last]:
            shutil.rmtree(path)

    def valid_checkpoints(self) -> list[Path]:
        return sorted(
            (path for path in self.root.glob("step_*" ) if validate_checkpoint(path)),
            key=lambda path: path.name,
        )

    def latest(self) -> Path | None:
        valid = self.valid_checkpoints()
        return valid[-1] if valid else None

    def load(self, path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
        if not validate_checkpoint(path):
            raise CheckpointError(f"checkpoint is incomplete or corrupt: {path}")
        state = torch.load(path / "state.pt", map_location=device, weights_only=False)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return state, metadata
