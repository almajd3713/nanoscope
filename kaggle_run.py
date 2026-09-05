"""Run the strict M0 training profile through the Run with Kaggle extension."""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Protocol, cast

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "m0" / "kaggle-acceptance.yaml"
REQUIRED_SECRETS = ("HF_TOKEN", "HF_REPO_ID", "WANDB_API_KEY")
OPTIONAL_SECRETS = ("WANDB_ENTITY",)


class _SecretsClient(Protocol):
    def get_secret(self, label: str) -> str: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the Kaggle environment, validate it, and run M0 training."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="training configuration (default: the strict M0 Kaggle profile)",
    )
    parser.add_argument(
        "--resume",
        default="none",
        help="none, auto, a checkpoint directory, or an hf:// checkpoint URI",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="skip dependency installation when rerunning in the same Kaggle session",
    )
    return parser


def _load_secrets() -> None:
    try:
        module = importlib.import_module("kaggle_secrets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kaggle_run.py must run inside a Kaggle kernel; kaggle_secrets is unavailable"
        ) from exc

    client = cast(_SecretsClient, module.UserSecretsClient())
    missing: list[str] = []
    for name in REQUIRED_SECRETS:
        try:
            value = client.get_secret(name)
        except Exception:
            missing.append(name)
            continue
        if not value:
            missing.append(name)
            continue
        os.environ[name] = value

    if missing:
        labels = ", ".join(missing)
        raise RuntimeError(
            f"missing Kaggle secrets: {labels}. Add and enable them for this notebook."
        )

    for name in OPTIONAL_SECRETS:
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ[name] = value


def _run(*args: str) -> None:
    # Forward notebook/launcher stop signals to the CLI, which coordinates a
    # checkpoint boundary across its DDP workers.
    with subprocess.Popen(args, cwd=ROOT) as child:

        def forward(signum: int, _frame: object) -> None:
            if child.poll() is None:
                child.send_signal(signum)

        old_int = signal.signal(signal.SIGINT, forward)
        old_term = signal.signal(signal.SIGTERM, forward)
        try:
            status = child.wait()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
        if status:
            raise subprocess.CalledProcessError(status, args)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _load_secrets()

    if not args.skip_install:
        _run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(ROOT / "requirements-kaggle.lock"),
        )
        _run(sys.executable, "-m", "pip", "install", "-e", str(ROOT), "--no-deps")

    _run(sys.executable, "-m", "nanoscope", "doctor", "--config", args.config)
    _run(
        sys.executable,
        "-m",
        "nanoscope",
        "train",
        "--config",
        args.config,
        "--resume",
        args.resume,
    )


if __name__ == "__main__":
    main()
