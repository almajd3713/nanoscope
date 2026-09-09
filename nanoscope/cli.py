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

    corpus = commands.add_parser("prepare-eval", help="freeze a held-out token corpus")
    corpus.add_argument("--config", required=True)

    evaluation = commands.add_parser("evaluate", help="score a saved checkpoint on a frozen corpus")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--eval-config", required=True)
    evaluation.add_argument("--training-config", help="original config for a legacy checkpoint")

    comparison = commands.add_parser(
        "compare", help="compare evaluations at a fixed training budget"
    )
    comparison.add_argument("--study", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--plots", action="store_true", help="write PNG/SVG plots (eval extra)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command in {"prepare-eval", "evaluate", "compare"}:
        from nanoscope.eval.compare import build_comparison
        from nanoscope.eval.config import load_corpus_config, load_eval_config
        from nanoscope.eval.corpus import prepare_corpus
        from nanoscope.eval.report import write_report
        from nanoscope.eval.runner import evaluate_checkpoint

        try:
            if args.command == "prepare-eval":
                corpus = prepare_corpus(load_corpus_config(args.config))
                print(json.dumps({"corpus": str(corpus.path), "fingerprint": corpus.fingerprint}))
            elif args.command == "evaluate":
                path = evaluate_checkpoint(
                    args.checkpoint, load_eval_config(args.eval_config), args.training_config
                )
                print(json.dumps({"result": str(path)}))
            else:
                path = write_report(
                    build_comparison(args.study), Path(args.output), plots=args.plots
                )
                print(json.dumps({"report": str(path)}))
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise SystemExit(str(exc)) from exc
        return
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
