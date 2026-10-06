"""The response models agree with the JSON Schema files the library writes."""

import pytest

from nanoscope import schemas
from nanoscope.server.app import create_app
from nanoscope.server.models import SCHEMA_OF, ProblemDetails


@pytest.mark.parametrize(("model", "name"), [(m, n) for m, n in SCHEMA_OF.items()],
                         ids=lambda x: getattr(x, "__name__", x))
def test_model_required_fields_equal_the_schema_files(model, name):
    file_required = set(schemas.get(name)["required"])
    component = model.model_json_schema(by_alias=True)
    assert set(component.get("required", [])) == file_required, (
        f"{model.__name__} disagrees with {name}.v{schemas.CURRENT[name]}.json")


def test_openapi_components_come_from_these_models():
    spec = create_app().openapi()
    components = spec["components"]["schemas"]
    assert set(components["Version"]["required"]) == {"nanoscope", "schemas"}
    assert components["Health"]["required"] == ["status"]


def test_documents_keep_unknown_fields_and_dump_by_alias():
    from nanoscope.server.models import StatusDoc

    doc = StatusDoc.model_validate({"schema": 1, "state": "done", "updated_at": "t", "later": 7})
    dumped = doc.model_dump(by_alias=True)
    assert dumped["schema"] == 1 and dumped["later"] == 7 and "schema_version" not in dumped


def test_problem_details_shape():
    assert set(ProblemDetails.model_json_schema()["required"]) == {
        "type", "title", "status", "detail"}


def test_level0_defaults():
    """A POST body has the defaults of the library function behind it, so an empty body (just
    the model) does what `run(Model)` does."""
    import inspect

    from nanoscope import compare, run
    from nanoscope.bench import bench
    from nanoscope.inspect import describe
    from nanoscope.learn.checks import run_lesson_checks
    from nanoscope.server.routes.compare import CompareRequest
    from nanoscope.server.routes.hardware import BenchRequest
    from nanoscope.server.routes.learn import CheckRequest, StartRequest, UnlockRequest
    from nanoscope.server.routes.models import DescribeRequest
    from nanoscope.server.routes.runs import RunRequest

    def defaults(fn):
        return {k: p.default for k, p in inspect.signature(fn).parameters.items()
                if p.default is not inspect.Parameter.empty}

    def field_defaults(cls):
        return {k: f.default for k, f in cls.model_fields.items() if not f.is_required()}

    pairs = [(RunRequest, run, ["preset", "seed", "compile", "push_to_hub", "wandb"]),
             (BenchRequest, bench, ["preset", "steps"]),
             (DescribeRequest, describe, ["preset"]),
             (CompareRequest, compare, ["baseline", "metric", "preset"]),
             (CheckRequest, run_lesson_checks, ["variant"])]
    for body, fn, names in pairs:
        library, api = defaults(fn), field_defaults(body)
        for name in names:
            assert api[name] == library[name], f"{body.__name__}.{name} differs from {fn.__name__}"
    # the bodies that may be left out entirely build from nothing
    assert RunRequest(model="bigram").kwargs == {}
    for body in (DescribeRequest, CheckRequest, StartRequest, UnlockRequest):
        body()
