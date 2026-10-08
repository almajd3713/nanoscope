"""`nanoscope run-job <id>`: do one queued job in this (fresh) process.

The worker starts it through a JobRunner. It reads the job from the queue, dispatches on
the kind, and writes the result (or the error) back to the queue. The worker decides what
the exit means for the job's state.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

from nanoscope import queue, store
from nanoscope.hardware import configure_device
from nanoscope.jobs.payload import build_preset
from nanoscope.modelref import load_class
from nanoscope.schemas.upgrade import read_json

NAMED_MODELS = ("bigram", "gpt2", "modern")


def model_class(ref: str) -> type:
    if ref in NAMED_MODELS:
        from nanoscope import models

        return {"bigram": models.Bigram, "gpt2": models.GPT2, "modern": models.Modern}[ref]
    return load_class(ref)


def _run(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from nanoscope.run import RunResult, run

    out = store.resolve(job["ref"], must_exist=False) if job["ref"] else None
    result = run(
        model_class(payload["model"]), build_preset(payload), seed=payload.get("seed", 0),
        device=job["device"], output_dir=out, push_to_hub=payload.get("push_to_hub"),
        wandb=payload.get("wandb", False), compile=payload.get("compile", False),
        study=payload.get("study"), progress=False, **payload.get("kwargs", {}))
    assert isinstance(result, RunResult)
    state = read_json(result.run_dir / "status.json", "status")["state"]
    summary = result.summary()
    return {"state": state, "ref": summary["ref"], "final_step": summary["final_step"],
            "final_val_bpb": summary["final_val_bpb"]}


def _prepare_data(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from nanoscope.dataset import load_data
    from nanoscope.presets import get_preset

    data = load_data(get_preset(payload["preset"]))
    return {"preset": payload["preset"], "train_tokens": data.train.meta["tokens"],
            "val_tokens": len(data.val)}


def _bench(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from nanoscope.bench import bench

    result = bench(model_class(payload["model"]), payload.get("preset", "tinystories-5min"),
                   steps=payload.get("steps", 60), device=job["device"])
    return {"report": str(result), **result.to_dict()}


def _check(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from nanoscope.learn.checks import run_lesson_checks
    from nanoscope.learn.loader import load_lesson

    doc = run_lesson_checks(load_lesson(payload["lesson"]), owner=job["owner"] or "local",
                            variant=payload.get("variant", "cpu"))
    return {"passed": doc["passed"], "check_id": doc["id"], "lesson": doc["lesson"],
            "checks": [{"id": c["id"], "passed": c["passed"], "reason": c["reason"]}
                       for c in doc["checks"]]}


def _describe(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from nanoscope.inspect import describe

    return describe(model_class(payload["model"]), payload.get("preset", "tinystories-5min"),
                    **payload.get("kwargs", {}))


def _certify(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from nanoscope.blocks.certify import certify

    return certify(payload["file"], payload["block"])


def _study(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Load a study spec (this imports its models: it runs in a worker, not the API) and put
    every unfinished run on the batch lane."""
    from pathlib import Path

    from nanoscope import paths
    from nanoscope.study import Study
    from nanoscope.studyspec import StudySpec

    spec = StudySpec.load(payload["spec"])
    for variant in spec.variants:  # a relative file ref is relative to the workspace
        file, sep, name = variant.model.rpartition(":")
        if sep and file.endswith(".py") and not Path(file).is_absolute():
            candidate = paths.workspace_dir() / file
            if candidate.exists():
                variant.model = f"{candidate.resolve()}:{name}"
    study = Study.from_spec(spec, source=Path(payload["spec"]))
    ids = study.enqueue(lane="batch")
    return {"study": spec.name, "runs": len(study.jobs()), "jobs": ids}


def _load_spec_study(payload: dict[str, Any]) -> Any:
    from nanoscope.study import load_study

    return load_study(payload["spec"])


def _prereg_preview(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """What a preregistration commit of this study's spec would hold (changes nothing)."""
    from nanoscope.prereg import preview

    return preview(_load_spec_study(payload)).to_dict()


def _commit(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Make the preregistration commit the user previewed; refused if the preview changed."""
    from nanoscope.prereg import commit

    return {"commit": commit(_load_spec_study(payload), payload["preview_hash"])}


def _card_push(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Upload an exported ablation card to a Hub dataset. Only ever queued on request."""
    from nanoscope import cards

    return cards.push(cards.read_card(payload["card"]), payload["repo"])


def _sync_hub(job: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Pull a run that was trained elsewhere (Kaggle, Colab) from its private Hub repo, so
    it can be listed, compared and resumed here."""
    from nanoscope.integrations import HubSync

    ref = payload["ref"]
    run_dir = store.resolve(ref, must_exist=False)
    pulled = HubSync(payload["repo"], run_dir, payload.get("path", ref)).pull()
    return {"ref": ref, "repo": payload["repo"], "pulled": pulled,
            "present": (run_dir / "latest.json").exists()}


HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "run": _run, "prepare-data": _prepare_data, "bench": _bench, "check": _check,
    "describe": _describe, "study": _study, "sync-hub": _sync_hub, "certify": _certify,
    "prereg-preview": _prereg_preview, "commit": _commit, "card-push": _card_push,
}


def offline_refusal(kind: str, payload: dict[str, Any]) -> str | None:
    """Why a job cannot run under NANOSCOPE_JOBS_OFFLINE=1, or None if it can."""
    if kind == "sync-hub":
        return "it pulls a run from the Hugging Face Hub"
    if kind == "card-push":
        return f"it uploads a card to the Hub dataset {payload['repo']}"
    if kind == "run" and payload.get("push_to_hub"):
        return f"it pushes to the Hub repo {payload['push_to_hub']}"
    if kind == "run" and payload.get("wandb"):
        return "it logs to Weights & Biases"
    return None


def execute(job_id: int) -> int:
    """Run one job; returns the process exit code (0 unless the job raised)."""
    os.environ["NANOSCOPE_JOB_ID"] = str(job_id)
    job = queue.get(job_id)
    payload = json.loads(job["payload"])
    try:
        if os.environ.get("NANOSCOPE_JOBS_OFFLINE") == "1":
            why = offline_refusal(job["kind"], payload)
            if why:
                raise RuntimeError(f"NANOSCOPE_JOBS_OFFLINE=1: this {job['kind']} job needs the "
                                   f"network ({why}). Unset it to allow the job.")
            # cached data and tokenizers still load; anything that would download fails fast
            os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="offline")
        configure_device(job["device"], int(os.environ.get("NANOSCOPE_WORKER_SLOTS", "1")))
        result = HANDLERS[job["kind"]](job, payload)
    except BaseException as exc:  # a KeyboardInterrupt too: the queue must learn why
        traceback.print_exc()
        queue.record(job_id, error=f"{type(exc).__name__}: {exc}")
        return 1
    queue.record(job_id, result=result)
    return 0


def main(job_id: int) -> None:
    sys.exit(execute(job_id))


