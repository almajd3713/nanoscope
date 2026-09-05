"""Launch configured workers using the current Python environment."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from nanoscope.config import Config
from nanoscope.train.distributed import worker_count


def launch_training(config: Config, arguments: list[str]) -> int | None:
    """Return a child exit status, or None when the caller should train in-process."""
    count = worker_count(config)
    if "RANK" in os.environ or count == 1:
        return None
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={count}",
        "--max-restarts=0",
        "--module",
        "nanoscope",
        *arguments,
    ]
    print(
        f"Launching DDP with {count} workers (global batch {config.train.batch_size})", flush=True
    )
    with tempfile.TemporaryDirectory(prefix="nanoscope-launch-") as temporary:
        stop_file = Path(temporary) / "stop"

        def request_stop(_signum: int, _frame: object) -> None:
            # Let workers save at a shared step boundary before torchrun exits.
            stop_file.touch()

        old_int = signal.signal(signal.SIGINT, request_stop)
        old_term = signal.signal(signal.SIGTERM, request_stop)
        try:
            child = subprocess.Popen(
                command,
                env={**os.environ, "NANOSCOPE_STOP_FILE": str(stop_file)},
                start_new_session=True,
            )
            return child.wait()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
