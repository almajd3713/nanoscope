import jsonschema

from nanoscope.schemas import get


def assert_valid(kind, obj):
    """Fail with the first schema violation, naming where in the document it is."""
    try:
        jsonschema.validate(obj, get(kind), cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise AssertionError(f"{kind}.v1 violated at {where}: {exc.message}") from exc
