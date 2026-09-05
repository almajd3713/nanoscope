from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


class JsonlLogger:
    def __init__(self, path: Path, resume_step: int = 0) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            kept = [row for row in read_metrics(path) if int(row["step"]) <= resume_step]
            payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept)
            self.path.write_text(payload, encoding="utf-8")

    def log(self, values: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")
            handle.flush()


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class WandbLogger:
    def __init__(
        self,
        mode: Literal["disabled", "offline", "online"],
        run_id: str,
        project: str,
        entity: str | None,
        directory: Path,
        config: dict[str, Any],
        resumed: bool,
    ) -> None:
        self._run = None
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B logging was requested but wandb is not installed") from exc
        self._run = wandb.init(
            id=run_id,
            project=project,
            entity=entity,
            dir=str(directory),
            mode=mode,
            resume="allow" if resumed else "never",
            config=config,
        )
        self._run.define_metric("step")
        self._run.define_metric("*", step_metric="step")

    def log(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(values)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
