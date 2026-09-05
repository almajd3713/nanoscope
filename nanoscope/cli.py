from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from nanoscope.config import load_config
from nanoscope.train.acceptance import run_acceptance
from nanoscope.train.checkpoint import validate_checkpoint
from nanoscope.train.doctor import print_doctor
from nanoscope.train.launcher import launch_training
from nanoscope.train.trainer import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanoscope")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate the runtime and configuration")
    doctor.add_argument("--config", required=True)

    training = commands.add_parser("train", help="run or resume training")
    training.add_argument("--config", required=True)
    training.add_argument("--resume", default="auto")
    training.add_argument("--stop-after-step", type=int, help="checkpoint and stop at this step")

    inspect = commands.add_parser("inspect-checkpoint", help="inspect checkpoint metadata")
    inspect.add_argument("path")

    acceptance = commands.add_parser("m0-acceptance", help="run the interruption comparison")
    acceptance.add_argument("--config", required=True)
    acceptance.add_argument("--work-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        raise SystemExit(print_doctor(load_config(args.config)))
    if args.command == "train":
        config = load_config(args.config)
        exit_code = launch_training(config, list(argv if argv is not None else sys.argv[1:]))
        if exit_code is not None:
            raise SystemExit(exit_code)
        result = train(config, resume=args.resume, stop_after_step=args.stop_after_step)
        if int(os.getenv("RANK", "0")) == 0:
            print(json.dumps({"step": result.final_step, "checkpoint": str(result.checkpoint)}))
        return
    if args.command == "inspect-checkpoint":
        path = Path(args.path)
        if not validate_checkpoint(path):
            raise SystemExit(f"invalid checkpoint: {path}")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        print(json.dumps({"metadata": metadata, "state_keys": sorted(state)}, indent=2))
        return
    if args.command == "m0-acceptance":
        report = run_acceptance(load_config(args.config), args.work_dir)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)
