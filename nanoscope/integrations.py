"""Optional outside services: Weights & Biases for live curves, the HF Hub for keeping runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanoscope.log import info

StepHook = Callable[[int, dict[str, Any]], None]


def wandb_hook(
    project: str, name: str, config: dict[str, Any],
) -> tuple[StepHook, Callable[[], None]]:
    """Log every step row to W&B. Returns (on_step, finish)."""
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb=... needs: pip install 'nanoscope-lab[wandb]'") from exc

    wb_run = wandb.init(project=project, name=name, config=config, resume="allow", id=name)

    def on_step(step: int, row: dict[str, Any]) -> None:
        wb_run.log({k: v for k, v in row.items() if isinstance(v, int | float)}, step=step)

    return on_step, wb_run.finish


def chain(*hooks: StepHook | None) -> StepHook | None:
    active = [h for h in hooks if h is not None]
    if not active:
        return None

    def on_step(step: int, row: dict[str, Any]) -> None:
        for hook in active:
            hook(step, row)

    return on_step


class HubSync:
    """Keep a run folder mirrored in a private Hub model repo, so a new machine or Kaggle
    session resumes where the last one stopped. Uploads at most every `every` seconds,
    plus the final checkpoint; older remote checkpoints are replaced."""

    def __init__(self, repo_id: str, run_dir: Path, path_in_repo: str, every: float = 1200):
        self.repo_id, self.run_dir, self.path = repo_id, run_dir, path_in_repo
        self.every = every
        self.last_push = time.monotonic()
        self.created = False

    def pull(self) -> bool:
        """Fetch the run from the Hub if it's there and we have no local checkpoint."""
        if (self.run_dir / "latest.json").exists():
            return False
        from huggingface_hub import HfApi, snapshot_download

        try:
            files = HfApi().list_repo_files(self.repo_id)
        except Exception:
            return False
        if f"{self.path}/latest.json" not in files:
            return False
        info(f"resuming {self.path} from hf.co/{self.repo_id}")
        root = self.run_dir
        for _ in Path(self.path).parts:
            root = root.parent
        snapshot_download(self.repo_id, local_dir=root, allow_patterns=[f"{self.path}/*"])
        return True

    def __call__(self, step: int, final: bool) -> None:
        if final or time.monotonic() - self.last_push >= self.every:
            self.push(step)

    def push(self, step: int, attempts: int = 3) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        if not self.created:
            api.create_repo(self.repo_id, exist_ok=True, private=True)
            self.created = True
        info(f"uploading {self.path} at step {step} to hf.co/{self.repo_id}")
        for attempt in range(attempts):
            try:
                api.upload_folder(
                    repo_id=self.repo_id, folder_path=str(self.run_dir), path_in_repo=self.path,
                    delete_patterns=["checkpoints/*"],
                    commit_message=f"nanoscope: {self.path} step {step}",
                )
                self.last_push = time.monotonic()
                return
            except Exception as exc:
                if attempt == attempts - 1:  # training goes on; the next checkpoint retries
                    info(f"upload failed ({type(exc).__name__}); will retry at "
                          "the next checkpoint")
                    return
                time.sleep(2**attempt)
