"""nanoscope run my_model.py:MyLM --preset tinystories-5min --set d_model=256 --seeds 3
nanoscope compare runs/tinystories-5min/mylm gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --devices cuda:0,cuda:1
nanoscope report studies/m1_ablation.py"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanoscope import paths
from nanoscope.compare import compare
from nanoscope.modelref import load_class
from nanoscope.presets import list_presets
from nanoscope.run import RunGroup, run


def _load_model_class(spec: str):
    if ":" not in spec:
        raise ValueError(f"model spec must be file.py:ClassName, got {spec!r}")
    file_part, class_name = spec.rsplit(":", 1)
    path = Path(file_part).resolve()
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    return load_class(f"{path}:{class_name}")


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

    job_parser = sub.add_parser("run-job", help="Run one queued job in this process (workers do)")
    job_parser.add_argument("id", type=int)

    bench_parser = sub.add_parser("bench", help="Measure training speed and find the bottleneck")
    bench_parser.add_argument("model", help="bigram, gpt2, modern, or path/to/model.py:ClassName")
    bench_parser.add_argument("--preset", default="tinystories-5min")
    bench_parser.add_argument("--device", default=None)
    bench_parser.add_argument("--steps", type=int, default=60)
    bench_parser.add_argument("--save", action="store_true",
                              help="append the result to <home>/hardware/bench.jsonl")
    bench_parser.add_argument("--compile", nargs="?", const="true", default=None,
                              help="torch.compile; or --compile reduce-overhead for CUDA graphs")
    bench_parser.add_argument("--set", nargs="*", default=[], dest="overrides")

    worker_parser = sub.add_parser("worker", help="Run queued jobs on one device")
    worker_parser.add_argument("--device", default="cpu", help="cpu, cuda:0, ...")
    worker_parser.add_argument("--slots", type=int, default=1, help="jobs at once on the device")
    worker_parser.add_argument("--lanes", default="interactive,batch")
    worker_parser.add_argument("--timeout", type=float, default=None,
                               help="seconds a job may run before it is stopped and fails")
    worker_parser.add_argument("--exit-when-idle", action="store_true",
                               help="exit once nothing is queued or running")

    prep_parser = sub.add_parser("prepare-data", help="Download or tokenize a preset's data now")
    prep_parser.add_argument("preset")

    pub_parser = sub.add_parser("publish-data", help="Upload a preset's tokens to a Hub dataset")
    pub_parser.add_argument("preset")
    pub_parser.add_argument("repo", help="user/name of the Hub dataset repo")
    pub_parser.add_argument("--private", action="store_true")

    status_parser = sub.add_parser("status", help="Show the state of every run under a folder")
    status_parser.add_argument("--data", action="store_true",
                               help="show data preparation (downloads, tokenizing) instead")
    status_parser.add_argument("--workers", action="store_true",
                               help="show the running workers and their jobs instead")
    status_parser.add_argument("path", nargs="?", default=None,
                               help="runs folder (default: $NANOSCOPE_HOME/runs, or ./runs)")

    study_parser = sub.add_parser("study", help="Train every run of a study file")
    study_parser.add_argument("file", help="path/to/study.py or study.toml")
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
    report_parser.add_argument("file", help="path/to/study.py or study.toml")
    report_parser.add_argument("--name", default=None)

    spec_parser = sub.add_parser("spec", help="Print a study as TOML (a spec you can commit)")
    spec_parser.add_argument("file", help="path/to/study.py")
    spec_parser.add_argument("--name", default=None, help="which Study, if the file has several")

    stop_parser = sub.add_parser(
        "stop", help="Ask running runs to stop: they save a checkpoint and can be resumed")
    stop_parser.add_argument("target", help="a run ref, a study name, or a folder")

    cmp_parser = sub.add_parser("compare", help="Compare runs against a baseline (the last one)")
    cmp_parser.add_argument(
        "runs", nargs="+",
        help="run refs (see `nanoscope status`), run folders, or run names with --preset")
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

    if args.command == "run-job":
        from nanoscope.jobs.execute import main as run_job

        run_job(args.id)
        return

    if args.command == "worker":
        from nanoscope.jobs.worker import Worker

        Worker(args.device, args.slots, lanes=tuple(args.lanes.split(",")),
               timeout=args.timeout, exit_when_idle=args.exit_when_idle).run()
        return

    if args.command == "bench":
        from nanoscope import models
        from nanoscope.bench import bench

        named = {"bigram": models.Bigram, "gpt2": models.GPT2, "modern": models.Modern}
        model_cls = named.get(args.model) or _load_model_class(args.model)
        print(bench(model_cls, args.preset, steps=args.steps, device=args.device,
                    compile=_parse_compile(args.compile), save=args.save,
                    **_parse_set(args.overrides)))
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

        if args.workers:
            from nanoscope.progress import format_workers, read_workers

            print(format_workers(read_workers()))
            return
        if args.data:
            from nanoscope.prepare import format_prepare, read_all

            print(format_prepare(read_all()))
            return
        root = args.path or paths.runs_dir()
        print(format_snapshot(snapshot(root), root))
        return

    if args.command == "spec":
        from nanoscope.study import load_study

        print(load_study(args.file, args.name).to_spec().to_toml(), end="")
        return

    if args.command == "stop":
        from nanoscope.store import request_stop

        refs = request_stop(args.target)
        for ref in refs:
            print(f"stop requested: {ref}")
        if not refs:
            print(f"nothing is running under {args.target}")
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
