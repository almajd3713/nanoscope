"""nanoscope run my_model.py:MyLM --preset tinystories-5min --set d_model=256 --seeds 3
nanoscope compare runs/tinystories-5min/mylm gpt2 --preset tinystories-5min"""

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


def main(argv: list[str] | None = None) -> None:
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

    cmp_parser = sub.add_parser("compare", help="Compare runs against a baseline (the last one)")
    cmp_parser.add_argument("runs", nargs="+", help="run folders, or run names with --preset")
    cmp_parser.add_argument("--preset", default=None)
    cmp_parser.add_argument("--metric", default="val_bpb", choices=["val_bpb", "val_loss"])

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    if args.command == "presets":
        for name in list_presets():
            print(name)
        return

    if args.command == "compare":
        print(compare(*args.runs, preset=args.preset, metric=args.metric))
        return

    if args.command == "run":
        model_cls = _load_model_class(args.model)
        kwargs = _parse_set(args.overrides)

        def on_step(step, row):
            loss_str = f"loss={row['loss']:.4f}"
            val_str = (f" val={row['val_loss']:.4f} ({row['val_bpb']:.3f} bpb)"
                       if "val_loss" in row else "")
            lr_str = f" lr={row['lr']:.2e}"
            tps = f" tok/s={row['tokens_per_sec']:.0f}"
            print(f"  step {step:>6d}  {loss_str}{val_str}{lr_str}{tps}")

        result = run(
            model_cls,
            preset=args.preset,
            seed=args.seed,
            seeds=args.seeds,
            device=args.device,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            on_step=on_step,
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
