from __future__ import annotations

import json
import os
import time
from pathlib import Path


class HubCheckpointStore:
    def __init__(self, repo_id: str | None, run_id: str, policy: str) -> None:
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")
        self.run_id = run_id
        self.policy = policy
        if policy == "required" and not self.repo_id:
            raise RuntimeError("required Hub persistence needs hub_repo or HF_REPO_ID")

    @property
    def enabled(self) -> bool:
        return self.policy != "disabled" and self.repo_id is not None

    def upload(self, checkpoint: Path, step: int, attempts: int = 3) -> str | None:
        if not self.enabled:
            return None
        repo_id = self.repo_id
        assert repo_id is not None
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ImportError as exc:
            if self.policy == "required":
                raise RuntimeError("Hub persistence requires huggingface-hub") from exc
            return None

        latest = checkpoint.parent.parent / ".hub-latest.json"
        latest.write_text(
            json.dumps({"step": step, "checkpoint": f"step_{step:08d}"}), encoding="utf-8"
        )
        prefix = f"runs/{self.run_id}/checkpoints/step_{step:08d}"
        error: Exception | None = None
        for attempt in range(attempts):
            try:
                # CommitOperation instances are mutated by the SDK and cannot be retried.
                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}/{name}", path_or_fileobj=checkpoint / name
                    )
                    for name in ("state.pt", "metadata.json", "manifest.json")
                ]
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"runs/{self.run_id}/latest.json", path_or_fileobj=latest
                    )
                )
                result = HfApi().create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=operations,
                    commit_message=f"nanoscope {self.run_id} recovery checkpoint step {step}",
                )
                return result.commit_url
            except Exception as exc:  # SDK exposes several transport exception types.
                error = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        if self.policy == "required":
            raise RuntimeError(f"failed to persist required Hub checkpoint: {error}") from error
        return None


def download_hub_checkpoint(uri: str, destination: Path) -> Path:
    """Download hf://owner/repo/runs/<run-id> using its latest pointer."""
    if not uri.startswith("hf://"):
        raise ValueError("Hub checkpoint URI must start with hf://")
    parts = uri.removeprefix("hf://").strip("/").split("/")
    if len(parts) < 4:
        raise ValueError("expected hf://owner/repo/runs/<run-id>")
    repo_id = "/".join(parts[:2])
    base = "/".join(parts[2:])
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hub resume requires huggingface-hub") from exc

    latest_file = hf_hub_download(repo_id=repo_id, filename=f"{base}/latest.json")
    latest = json.loads(Path(latest_file).read_text(encoding="utf-8"))
    remote_prefix = f"{base}/checkpoints/{latest['checkpoint']}"
    local_root = Path(
        snapshot_download(repo_id=repo_id, allow_patterns=f"{remote_prefix}/*")
    )
    source = local_root / remote_prefix
    target = destination / latest["checkpoint"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        import shutil

        shutil.copytree(source, target)
    return target
