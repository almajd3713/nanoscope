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
