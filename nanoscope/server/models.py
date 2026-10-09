"""Response models. Where the library already writes a versioned JSON file (`status.json`,
`comparison`, `describe`, ...), the model has exactly that schema's required fields, so the
OpenAPI document, the JSON Schemas and the files on disk describe one thing. Extra keys are kept
(`extra="allow"`): a newer library can add a field without breaking an older client.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Doc(BaseModel):
    """A versioned document: `schema` is the integer version."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Health(BaseModel):
    status: str


class Version(BaseModel):
    nanoscope: str
    schemas: dict[str, int]


class StatusDoc(Doc):
    """A run's status.json (status.v1)."""

    schema_version: int = Field(alias="schema")
    state: str
    updated_at: str
    step: int | None = None
    max_steps: int | None = None
    error: dict[str, Any] | None = None


class ConfigDoc(Doc):
    """A run's config.json (config.v1)."""

    schema_version: int = Field(alias="schema")
    model: dict[str, Any]
    preset: dict[str, Any]
    tokenizer: str
    seed: int


class ComparisonDoc(Doc):
    """`compare().to_dict()` (comparison.v1)."""

    schema_version: int = Field(alias="schema")
    metric: str
    baseline: str
    rows: list[dict[str, Any]]
    notes: list[str]
    curves: list[dict[str, Any]] | None = None
    precision_plan: dict[str, Any] | None = None
    noise_floor: dict[str, Any] | None = None


class DescribeDoc(Doc):
    """`describe()` (describe.v1)."""

    schema_version: int = Field(alias="schema")
    nanoscope: str
    model: str
    kwargs: dict[str, Any]
    preset: str
    context_length: int
    batch_size: int
    params: dict[str, int]
    flops_per_token: int
    flops_source: str
    memory: dict[str, int]
    modules: list[dict[str, Any]]


class GraphDoc(Doc):
    """`graph.parse()` (graph.v1)."""

    schema_version: int = Field(alias="schema")
    nanoscope: str
    path: str
    classes: list[dict[str, Any]]


class BlocksDoc(Doc):
    """The palette catalog (blocks.v1), with the lock state added per block."""

    schema_version: int = Field(alias="schema")
    nanoscope: str
    blocks: list[dict[str, Any]]
    errors: list[dict[str, Any]]


class CheckDoc(Doc):
    """A lesson check result (check.v1)."""

    schema_version: int = Field(alias="schema")
    nanoscope: str
    id: str
    lesson: str
    at: str
    variant: str
    passed: bool
    checks: list[dict[str, Any]]


class ProgressDoc(Doc):
    """learn/progress.json (progress.v1)."""

    schema_version: int = Field(alias="schema")
    nanoscope: str
    lessons: dict[str, dict[str, Any]]


class UnlocksDoc(Doc):
    """learn/unlocks.json (unlocks.v1)."""

    schema_version: int = Field(alias="schema")
    nanoscope: str
    policy: str
    unlocks: dict[str, dict[str, Any]]


class BenchDoc(Doc):
    """One bench result (bench.v1)."""

    schema_version: int = Field(alias="schema")
    at: str
    model: str
    device: str
    step_ms: float
    tokens_per_sec: float
    tflops: float
    verdict: str


class PrepareDoc(Doc):
    """Data preparation state (prepare.v1)."""

    schema_version: int = Field(alias="schema")
    preset: str
    stage: str
    started_at: str
    updated_at: str


class WorkerDoc(Doc):
    """A worker's file (worker.v1)."""

    schema_version: int = Field(alias="schema")
    worker_id: str
    device: str
    slots: int
    jobs: list[Any]
    started_at: str
    heartbeat_at: str


class ProblemItem(BaseModel):
    """One thing wrong with a request (problem.v1)."""

    code: str
    message: str
    field: str | None = None
    hint: str | None = None


class ProblemDetails(BaseModel):
    """The body of every error response (RFC 9457)."""

    model_config = ConfigDict(extra="allow")

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None
    problems: list[ProblemItem] | None = None


class FieldSpecDoc(BaseModel):
    name: str
    type: str
    default: Any
    required: bool
    help: str


class PresetSpecDoc(BaseModel):
    """A preset and every field that can be overridden (`nanoscope.specs.PresetSpec`)."""

    name: str
    fields: list[FieldSpecDoc]


class ParamSpecDoc(BaseModel):
    name: str
    annotation: str | None
    default: Any
    required: bool
    from_data: bool


class ModelSpecDoc(BaseModel):
    """A model class and its constructor parameters (`nanoscope.specs.ModelSpec`)."""

    name: str
    ref: str
    doc: str
    params: list[ParamSpecDoc]
    shipped: bool = False
    source: str | None = None  # the workspace file a model was found in, if any


class JobDoc(BaseModel):
    """A queue job: what to do, where it stands, and what came of it."""

    id: int
    kind: str
    lane: str
    state: str
    ref: str | None = None
    owner: str = "local"
    device: str | None = None
    worker_id: str | None = None
    attempts: int = 0
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


# model -> the JSON Schema file it must agree with
SCHEMA_OF = {
    StatusDoc: "status", ConfigDoc: "config", ComparisonDoc: "comparison",
    DescribeDoc: "describe", GraphDoc: "graph", BlocksDoc: "blocks", CheckDoc: "check",
    ProgressDoc: "progress", UnlocksDoc: "unlocks", BenchDoc: "bench", PrepareDoc: "prepare",
    WorkerDoc: "worker", ProblemItem: "problem",
}
