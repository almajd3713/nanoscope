"""nanoscope run my_model.py:MyLM --preset tinystories-5min --set d_model=256 --seeds 3
nanoscope compare runs/tinystories-5min/mylm gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --devices cuda:0,cuda:1
nanoscope report studies/m1_ablation.py"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from nanoscope import paths
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
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load a model from {path}")
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


def _parse_compile(value: str | None) -> bool | str:
    return False if value is None else (True if value == "true" else value)


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
    run_parser.add_argument("--compile", nargs="?", const="true", default=None,
                            help="torch.compile; or --compile reduce-overhead for CUDA graphs")
    run_parser.add_argument("--no-resume", action="store_true")

    sub.add_parser("presets", help="List available presets")

    bench_parser = sub.add_parser("bench", help="Measure training speed and find the bottleneck")
    bench_parser.add_argument("model", help="bigram, gpt2, modern, or path/to/model.py:ClassName")
    bench_parser.add_argument("--preset", default="tinystories-5min")
    bench_parser.add_argument("--device", default=None)
    bench_parser.add_argument("--steps", type=int, default=60)
    bench_parser.add_argument("--compile", nargs="?", const="true", default=None,
                              help="torch.compile; or --compile reduce-overhead for CUDA graphs")
    bench_parser.add_argument("--set", nargs="*", default=[], dest="overrides")

    prep_parser = sub.add_parser("prepare-data", help="Download or tokenize a preset's data now")
    prep_parser.add_argument("preset")

    pub_parser = sub.add_parser("publish-data", help="Upload a preset's tokens to a Hub dataset")
    pub_parser.add_argument("preset")
    pub_parser.add_argument("repo", help="user/name of the Hub dataset repo")
    pub_parser.add_argument("--private", action="store_true")

    status_parser = sub.add_parser("status", help="Show the state of every run under a folder")
    status_parser.add_argument("path", nargs="?", default=None,
                               help="runs folder (default: $NANOSCOPE_HOME/runs, or ./runs)")

    study_parser = sub.add_parser("study", help="Train every run of a study file")
    study_parser.add_argument("file", help="path/to/study.py")
    study_parser.add_argument("--name", default=None, help="which Study, if the file has several")
    study_parser.add_argument("--devices", default=None, help="comma-separated, e.g. cuda:0,cuda:1")
    study_parser.add_argument("--workers-per-device", type=int, default=1,
                              help="runs at once on each device; more keeps a GPU busier")
    study_parser.add_argument("--threads", type=int, default=None,
                              help="CPU threads per worker (default: cores split between workers)")
    study_parser.add_argument("--compile", nargs="?", const="true", default=None,
                              help="torch.compile; or --compile reduce-overhead for CUDA graphs")
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

    if args.command == "bench":
        from nanoscope import models
        from nanoscope.bench import bench

        named = {"bigram": models.Bigram, "gpt2": models.GPT2, "modern": models.Modern}
        model_cls = named.get(args.model) or _load_model_class(args.model)
        print(bench(model_cls, args.preset, steps=args.steps, device=args.device,
                    compile=_parse_compile(args.compile), **_parse_set(args.overrides)))
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

        root = args.path or paths.runs_dir()
        print(format_snapshot(snapshot(root), root))
        return

    if args.command in ("study", "report"):
        from nanoscope.study import load_study

        study = load_study(args.file, args.name)
        if args.command == "study":
            devices = args.devices.split(",") if args.devices else None
            shard = None
            if args.shard:
                index, count = (int(x) for x in args.shard.split("/"))
                shard = (index, count)
            if args.push_to_hub:
                study.push_to_hub = args.push_to_hub
            if args.compile:
                study.compile = _parse_compile(args.compile)
            study.run(devices=devices, shard=shard, workers_per_device=args.workers_per_device,
                      threads=args.threads)
            if shard is not None:
                return
        print(study.report())
        print(f"Written to {paths.reports_dir() / study.name}/")
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
            compile=_parse_compile(args.compile),
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
