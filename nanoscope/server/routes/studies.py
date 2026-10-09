from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from nanoscope import cards, paths, queue, store, studyfiles
from nanoscope.schemas.upgrade import read_json
from nanoscope.server import workspace
from nanoscope.server.errors import problem
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import JobDoc
from nanoscope.server.routes.git import git_status
from nanoscope.server.routes.models import find_model, resolve_ref
from nanoscope.server.routes.validate import StudyValidation, validate_study
from nanoscope.studyspec import StudySpec

router = APIRouter(prefix="/api", tags=["studies"])

SPEC_DIR = "studies"  # workspace/studies/<name>.toml


class VariantInfo(BaseModel):
    name: str
    model: str


class StudyEntry(BaseModel):
    name: str
    spec: str | None  # the workspace file the spec lives in
    preset: str | None = None
    mode: str | None = None
    seeds: list[int] = []
    variants: list[VariantInfo] = []
    runs_total: int = 0
    runs_by_state: dict[str, int] = {}  # of the runs that exist on disk


def spec_path(name: str) -> str:
    return f"{SPEC_DIR}/{name}.toml"


def study_dir(name: str):
    return paths.runs_dir() / "studies" / name


def entry(name: str) -> StudyEntry:
    path = workspace.root() / spec_path(name)
    spec = StudySpec.load(path) if path.exists() else None
    folder = study_dir(name)
    states: dict[str, int] = {}
    if folder.exists():
        for run in store.list_runs(f"studies/{name}"):
            states[run.state] = states.get(run.state, 0) + 1
    total = 0
    plan = folder / "plan.json"
    if plan.exists():
        total = len(read_json(plan, "plan")["runs"])
    elif spec is not None:
        total = len(spec.variants) * len(spec.seeds)
    return StudyEntry(
        name=name, spec=spec_path(name) if spec else None,
        preset=spec.preset if spec else None, mode=spec.mode if spec else None,
        seeds=list(spec.seeds) if spec else [],
        variants=[VariantInfo(name=v.name, model=v.model) for v in spec.variants] if spec else [],
        runs_total=total, runs_by_state=states)


def names() -> list[str]:
    found = {p.stem for p in (workspace.root() / SPEC_DIR).glob("*.toml")} if (
        workspace.root() / SPEC_DIR).exists() else set()
    runs_root = paths.runs_dir() / "studies"
    if runs_root.exists():
        found |= {p.name for p in runs_root.iterdir() if p.is_dir()}
    return sorted(found)


@router.get("/studies")
def list_studies() -> list[StudyEntry]:
    """Studies in the workspace (`studies/<name>.toml`) and studies that have been run."""
    return [entry(n) for n in names()]


class SpecDoc(BaseModel):
    name: str
    path: str  # the workspace file
    spec: dict[str, Any]
    toml: str


@router.get("/studies/{name}/spec", response_model=SpecDoc)
def study_spec(name: str) -> Any:
    """A saved study spec, as data and as the file text."""
    spec = _spec_or_404(name)
    return SpecDoc(name=name, path=spec_path(name), spec=spec.to_dict(), toml=spec.to_toml())


class StudyUpload(StudyValidation):
    overwrite: bool = False


@router.post("/studies", status_code=201, response_model=StudyEntry)
def save_study(body: StudyUpload, request: Request) -> Any:
    """Save a study spec as `studies/<name>.toml` in the workspace, after the same checks as
    `/validate/study`. It is the committable file: nothing runs yet."""
    verdict = validate_study(StudyValidation(toml=body.toml, spec=body.spec))
    if not verdict.ok:
        first = verdict.problems[0]
        return problem(422, first.message, request,
                       problems=[p.model_dump() for p in verdict.problems])
    spec = StudySpec.from_toml(body.toml) if body.toml is not None else StudySpec.from_dict(
        body.spec or {})
    if not spec.name.replace("-", "").replace("_", "").isalnum():
        return problem(422, f"study name {spec.name!r} may use letters, digits, - and _ only",
                       request)
    for variant in spec.variants:  # write refs that mean the same anywhere: no short names
        variant.model = find_model(variant.model)[0].ref
    target = workspace.safe_path(spec_path(spec.name), must_exist=False)
    if target.exists() and not body.overwrite:
        return problem(409, f"{spec_path(spec.name)} exists: send overwrite=true to replace it",
                       request)
    workspace.write_atomic(target, spec.to_toml())
    return entry(spec.name)


@router.post("/studies/sizes", status_code=202)
def study_sizes(body: StudyValidation, request: Request) -> Any:
    """Queue a sizes job for a spec that need not be saved: each variant's non-embedding
    parameters and FLOPs per token with `match_knob` resolved, and its distance from the
    reference variant. The job's result is the table (a worker builds the models)."""
    verdict = validate_study(StudyValidation(toml=body.toml, spec=body.spec))
    if not verdict.ok:
        return problem(422, verdict.problems[0].message, request,
                       problems=[p.model_dump() for p in verdict.problems])
    spec = StudySpec.from_toml(body.toml) if body.toml is not None else StudySpec.from_dict(
        body.spec or {})
    for variant in spec.variants:
        variant.model = resolve_ref(variant.model)
    job_id = queue.enqueue("sizes", {"spec": spec.to_dict()}, lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))


class FromFile(BaseModel):
    file: str  # a workspace .py file that defines a Study
    name: str | None = None  # which Study, when it defines several


@router.post("/studies/from-file", status_code=202)
def study_from_file(body: FromFile, request: Request) -> Any:
    """Queue reading a `.py` study as a spec (what `nanoscope spec` prints). A worker imports the
    file, so this process never does; the job's result is `{spec, toml}`."""
    if not body.file.endswith(".py"):
        return problem(422, f"{body.file!r} is not a .py file; a .toml spec opens by its name",
                       request)
    target = workspace.safe_path(body.file)
    job_id = queue.enqueue("study-spec", {"file": str(target), "name": body.name},
                           lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))


class StudyRun(BaseModel):
    study: str
    job: JobDoc


@router.post("/studies/{name}/run", status_code=202, response_model=StudyRun)
def run_study(name: str, request: Request) -> Any:
    """Queue a study. Record mode needs the spec committed and a clean tree. A worker loads the spec, which imports its models, and puts
    every unfinished run on the batch lane; watch them with `/jobs` or `/events?prefix=`."""
    path = workspace.root() / spec_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no study spec {spec_path(name)} in the workspace")
    spec = StudySpec.load(path)
    if spec.mode == "record":
        # The worker checks again when it queues the runs; this refuses early, with the files.
        git = git_status(spec_path(name))
        if not git.repo:
            return problem(422, "record mode needs the workspace inside a git repository", request)
        if not git.path_committed:
            return problem(422, f"commit {spec_path(name)} first: record mode runs from a "
                           "committed spec (Commit preregistration)", request)
        if not git.clean:
            return problem(422, "record mode needs a clean git tree, so every result maps to one "
                           "commit. Uncommitted changes:\n" + "\n".join(git.changed[:10]),
                           request, changed=git.changed)
    verdict = validate_study(StudyValidation(spec=spec.to_dict()))
    if not verdict.ok:
        return problem(422, verdict.problems[0].message, request,
                       problems=[p.model_dump() for p in verdict.problems])
    job_id = queue.enqueue("study", {"spec": str(path.resolve())}, lane="interactive",
                           ref=f"studies/{name}")
    assert job_id is not None
    return StudyRun(study=name, job=job_doc(queue.get(job_id)))


class StopResult(BaseModel):
    study: str
    stopping: list[str]


@router.post("/studies/{name}/stop")
def stop_study(name: str) -> StopResult:
    """Stop a study: queued runs are cancelled and running ones checkpoint and stop."""
    if not study_dir(name).exists() and not any(
            j["ref"] and j["ref"].startswith(f"studies/{name}") for j in queue.list_jobs()):
        raise FileNotFoundError(f"no study {name!r} has run or been queued")
    if paths.queue_db().exists():
        queue.cancel_prefix(f"studies/{name}")
    stopping = store.request_stop(f"studies/{name}") if study_dir(name).exists() else []
    return StopResult(study=name, stopping=stopping)


@router.get("/studies/{name}/report")
def study_report(name: str, request: Request) -> Any:
    """The study's results as data (results.v1): the comparison of its variants against the
    baseline, and any preregistered predictions scored. Computed from the run files, so it works
    for a study that is still running, and for one started from the command line."""
    if not study_dir(name).exists():
        raise FileNotFoundError(f"no study {name!r} has run")
    spec_file = workspace.root() / spec_path(name)
    spec = StudySpec.load(spec_file) if spec_file.exists() else None
    try:
        comparison, predictions, manifest = studyfiles.results(name, spec)
    except ValueError as exc:
        return problem(422, str(exc), request)
    doc = studyfiles.results_doc(name, spec.mode if spec else "explore", comparison, predictions,
                                 manifest)
    return {**doc, "comparison": json.loads(json.dumps(comparison.to_dict(), default=str))}


def _report_markdown(name: str, spec: StudySpec | None, comparison: Any,
                     predictions: list[dict[str, Any]], manifest: dict[str, Any] | None) -> str:
    budget = None
    if spec and spec.budget:
        (kind, n), = spec.budget.items()
        budget = ("Tokens" if kind == "tokens" else "FLOPs", n)
    return studyfiles.render_report(
        name, spec.mode if spec else "explore", spec.preset if spec else "?",
        list(spec.seeds) if spec else [], budget, manifest, comparison, predictions)


@router.get("/studies/{name}/report.md")
def study_report_markdown(name: str, request: Request) -> Any:
    """The Markdown report that goes in the bundle, as text."""
    if not study_dir(name).exists():
        raise FileNotFoundError(f"no study {name!r} has run")
    spec_file = workspace.root() / spec_path(name)
    spec = StudySpec.load(spec_file) if spec_file.exists() else None
    try:
        comparison, predictions, manifest = studyfiles.results(name, spec)
    except ValueError as exc:
        return problem(422, str(exc), request)
    return Response(_report_markdown(name, spec, comparison, predictions, manifest),
                    media_type="text/markdown; charset=utf-8")


@router.get("/studies/{name}/bundle.zip")
def study_bundle(name: str, request: Request) -> Any:
    """Everything needed to check or re-run the study, as one zip: report.md, results.json, the
    spec, study.json, plan.json and the per-seed finals. Built from the files on disk."""
    if not study_dir(name).exists():
        raise FileNotFoundError(f"no study {name!r} has run")
    spec_file = workspace.root() / spec_path(name)
    spec = StudySpec.load(spec_file) if spec_file.exists() else None
    try:
        comparison, predictions, manifest = studyfiles.results(name, spec)
    except ValueError as exc:
        return problem(422, str(exc), request)
    mode = spec.mode if spec else "explore"
    report = _report_markdown(name, spec, comparison, predictions, manifest)
    with tempfile.TemporaryDirectory() as tmp:
        zipped = studyfiles.write_bundle(
            Path(tmp) / f"{name}-bundle.zip", name, report,
            studyfiles.results_doc(name, mode, comparison, predictions, manifest),
            "spec.toml" if spec else "spec-missing.txt",
            spec.to_toml() if spec else "the workspace has no spec file for this study\n")
        data = zipped.read_bytes()
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{name}-bundle.zip"'})


def _spec_or_404(name: str) -> StudySpec:
    path = workspace.root() / spec_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no study spec {spec_path(name)} in the workspace")
    return StudySpec.load(path)


@router.get("/studies/{name}/card")
def study_card(name: str, request: Request) -> Any:
    """The ablation card (card.v1) of a finished record-mode study. Nothing is uploaded."""
    try:
        return cards.card_from_files(_spec_or_404(name))
    except ValueError as exc:
        return problem(422, str(exc), request)


class CardPlan(BaseModel):
    repo: str
    repo_type: str
    path_in_repo: str
    content: str  # the file's text, byte for byte
    commit_message: str
    exported_at: str  # send it back to push this exact card


@router.get("/studies/{name}/card/upload", response_model=CardPlan)
def card_upload_plan(name: str, request: Request, repo: str,
                     exported_at: str | None = None) -> Any:
    """Exactly what a push to `repo` would upload: the file, its text and the commit message.
    Nothing is sent. Push with the same `exported_at` to upload these very bytes; ask with it to
    see the same card again."""
    if repo.count("/") != 1 or not all(repo.split("/")):
        return problem(422, f"{repo!r} is not a Hub dataset id; use user/name", request)
    try:
        card = cards.card_from_files(_spec_or_404(name), exported_at)
    except ValueError as exc:
        return problem(422, str(exc), request)
    return {**cards.push_plan(card, repo), "exported_at": card["exported_at"]}


class CardPush(BaseModel):
    repo: str  # the public Hub dataset, user/name
    exported_at: str | None = None  # the stamp of the plan the user read; the card keeps it


class CardPushed(BaseModel):
    study: str
    repo: str
    path_in_repo: str  # the one file the job uploads
    job: JobDoc


@router.post("/studies/{name}/card/push", status_code=202, response_model=CardPushed)
def push_card(name: str, body: CardPush, request: Request) -> Any:
    """Queue the upload of this study's card to a public Hub dataset (needs HF_TOKEN in the
    worker). Opt-in: nothing calls this on its own. The response names the one file uploaded;
    `GET .../card` is its content."""
    if body.repo.count("/") != 1 or not all(body.repo.split("/")):
        return problem(422, f"{body.repo!r} is not a Hub dataset id; use user/name", request)
    try:
        card = cards.card_from_files(_spec_or_404(name), body.exported_at)
    except ValueError as exc:
        return problem(422, str(exc), request)
    path = cards.write_card(card, paths.reports_dir() / name / "card.json")
    job_id = queue.enqueue("card-push", {"card": str(path), "repo": body.repo},
                           lane="interactive")
    assert job_id is not None
    return CardPushed(study=name, repo=body.repo,
                      path_in_repo=cards.push_plan(card, body.repo)["path_in_repo"],
                      job=job_doc(queue.get(job_id)))


class PreregJob(BaseModel):
    study: str
    job: JobDoc


@router.post("/studies/{name}/preregister/preview", status_code=202, response_model=PreregJob)
def preregister_preview(name: str) -> Any:
    """Queue a preview of the preregistration commit: the files, the exact diff, the message,
    other uncommitted files and a hash. A worker runs git, so nothing here touches the
    repository. The job's result is the preview."""
    _spec_or_404(name)
    job_id = queue.enqueue("prereg-preview", {"spec": str(workspace.root() / spec_path(name))},
                           lane="interactive")
    assert job_id is not None
    return PreregJob(study=name, job=job_doc(queue.get(job_id)))


class PreregCommit(BaseModel):
    preview_hash: str  # the hash of the preview the user confirmed


@router.post("/studies/{name}/preregister/commit", status_code=202, response_model=PreregJob)
def preregister_commit(name: str, body: PreregCommit) -> Any:
    """Queue the preregistration commit. The worker recomputes the preview and refuses unless
    its hash equals `preview_hash`; the job's result is the commit hash."""
    _spec_or_404(name)
    job_id = queue.enqueue(
        "commit", {"spec": str(workspace.root() / spec_path(name)),
                   "preview_hash": body.preview_hash}, lane="interactive")
    assert job_id is not None
    return PreregJob(study=name, job=job_doc(queue.get(job_id)))
