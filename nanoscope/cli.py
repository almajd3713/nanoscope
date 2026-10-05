"""nanoscope run my_model.py:MyLM --preset tinystories-5min --set d_model=256 --seeds 3
nanoscope compare runs/tinystories-5min/mylm gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --devices cuda:0,cuda:1
nanoscope report studies/m1_ablation.py"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from nanoscope.compare import compare
from nanoscope.presets import list_presets
from nanoscope.run import RunGroup, run


def _load_model_class(spec: str):
    if ":" not in spec:
        raise ValueError(f"model spec must be file.py:ClassName, got {spec!r}")
    file_part, class_name = spec.rsplit(":", 1)
    path = Path(file_part).resolve()
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    module_spec = importlib.util.spec_from_file_location("_user_model", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"{class_name} not found in {path}")
    return getattr(module, class_name)


def _parse_set(items: list[str]) -> dict:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        for convert in (int, float):
            try:
                value = convert(value)
                break
            except ValueError:
                pass
        result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanoscope", description="Nanoscope CLI")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Train a model")
    run_parser.add_argument("model", help="path/to/model.py:ClassName")
    run_parser.add_argument("--preset", default="tinystories-5min")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--seeds", type=int, default=None, help="train seeds 0..N-1")
    run_parser.add_argument("--device", default=None)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    run_parser.add_argument("--no-resume", action="store_true")

    sub.add_parser("presets", help="List available presets")

    prep_parser = sub.add_parser("prepare-data", help="Download or tokenize a preset's data now")
    prep_parser.add_argument("preset")

    pub_parser = sub.add_parser("publish-data", help="Upload a preset's tokens to a Hub dataset")
    pub_parser.add_argument("preset")
    pub_parser.add_argument("repo", help="user/name of the Hub dataset repo")
    pub_parser.add_argument("--private", action="store_true")

    status_parser = sub.add_parser("status", help="Show the state of every run under a folder")
    status_parser.add_argument("path", nargs="?", default="runs")

    study_parser = sub.add_parser("study", help="Train every run of a study file")
    study_parser.add_argument("file", help="path/to/study.py")
    study_parser.add_argument("--name", default=None, help="which Study, if the file has several")
    study_parser.add_argument("--devices", default=None, help="comma-separated, e.g. cuda:0,cuda:1")
    study_parser.add_argument("--push-to-hub", default=None, metavar="REPO",
                              help="mirror runs to this Hub repo and resume from it")
    study_parser.add_argument("--shard", default=None, help=argparse.SUPPRESS)

    report_parser = sub.add_parser("report", help="Write a study's report to experiments/")
    report_parser.add_argument("file", help="path/to/study.py")
    report_parser.add_argument("--name", default=None)

    cmp_parser = sub.add_parser("compare", help="Compare runs against a baseline (the last one)")
    cmp_parser.add_argument("runs", nargs="+", help="run folders, or run names with --preset")
    cmp_parser.add_argument("--preset", default=None)
    cmp_parser.add_argument("--metric", default="val_bpb", choices=["val_bpb", "val_loss"])

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    if args.command == "presets":
        for name in list_presets():
            print(name)
        return

    if args.command in ("prepare-data", "publish-data"):
        from nanoscope.dataset import load_data, publish_data
        from nanoscope.presets import get_preset

        preset = get_preset(args.preset)
        if args.command == "prepare-data":
            data = load_data(preset)
            print(f"{preset.name}: {data.train.meta['tokens']:,} training tokens in "
                  f"{len(data.train.shards)} shard(s), {len(data.val):,} validation tokens")
        else:
            print(publish_data(preset, args.repo, private=args.private))
        return

    if args.command == "status":
        from nanoscope.progress import format_snapshot, snapshot

        print(format_snapshot(snapshot(args.path), args.path))
        return

    if args.command in ("study", "report"):
        from nanoscope.study import REPORTS_DIR, load_study

        study = load_study(args.file, args.name)
        if args.command == "study":
            devices = args.devices.split(",") if args.devices else None
            shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
            if args.push_to_hub:
                study.push_to_hub = args.push_to_hub
            study.run(devices=devices, shard=shard)
            if shard is not None:
                return
        print(study.report())
        print(f"Written to {REPORTS_DIR / study.name}/")
        return

    if args.command == "compare":
        print(compare(*args.runs, preset=args.preset, metric=args.metric))
        return

    if args.command == "run":
        model_cls = _load_model_class(args.model)
        kwargs = _parse_set(args.overrides)

        result = run(
            model_cls,
            preset=args.preset,
            seed=args.seed,
            seeds=args.seeds,
            device=args.device,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            **kwargs,
        )

        if isinstance(result, RunGroup):
            summary = result.summary()
            bpb = summary["val_bpb"]
            ci = (f" (95% CI {bpb['ci95_low']:.3f} to {bpb['ci95_high']:.3f})"
                  if bpb["ci95_low"] is not None else "")
            print(f"\nDone. {len(result)} seeds. Mean val bits per byte: {bpb['mean']:.3f}{ci}")
            print(f"Run dir: {summary['run_dir']}")
            return

        print(f"\nDone. {result.final_step} steps.")
        summary = result.summary()
        if summary["final_val_loss"] is not None:
            print(f"Final val loss: {summary['final_val_loss']:.4f} "
                  f"({summary['final_val_bpb']:.3f} bits per byte)")
        print(f"Run dir: {summary['run_dir']}")


if __name__ == "__main__":
    main()
