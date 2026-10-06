"""What the API process may import. It reads files and the queue and never runs user code, so it
must not reach for the parts of the library that import models or train them."""

import ast
from pathlib import Path

import nanoscope
import nanoscope.server as server_package

SERVER = Path(server_package.__file__).parent

# The documented library surface the server builds on. Anything else is a design question:
# add it here on purpose, with a reason, or call it through a job.
PUBLIC_MODULES = {
    "nanoscope.paths", "nanoscope.store", "nanoscope.queue", "nanoscope.schemas",
    "nanoscope.specs", "nanoscope.presets", "nanoscope.status", "nanoscope.progress",
    "nanoscope.prepare", "nanoscope.statistics", "nanoscope.studyspec", "nanoscope.study",
    "nanoscope.hardware", "nanoscope.estimate", "nanoscope.compare", "nanoscope.log",
    "nanoscope.fsutil", "nanoscope.learn", "nanoscope.blocks.catalog",
    "nanoscope.blocks.discover", "nanoscope.blocks.graph", "nanoscope.blocks.registry",
    "nanoscope.jobs.payload", "nanoscope.models",  # the shipped models: our code, not a learner's
}
# These execute or import user code: the API process never touches them (plan 8.1).
FORBIDDEN = {"nanoscope.inspect", "nanoscope.modelref", "nanoscope.run", "nanoscope.train_loop",
             "nanoscope.blockstats", "nanoscope.bench", "nanoscope.jobs.execute",
             "nanoscope.jobs.runner", "nanoscope.jobs.worker"}


def imported_names(path: Path):
    """(module, names) for every import of nanoscope in a file, including lazy ones."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == (
                "nanoscope"):
            yield node.module, [a.name for a in node.names], node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "nanoscope":
                    yield alias.name, [], node.lineno


def allowed(module: str) -> bool:
    return module == "nanoscope.server" or module.startswith("nanoscope.server.") or any(
        module == m or module.startswith(m + ".") for m in PUBLIC_MODULES)


def test_public_only():
    public_names = set(nanoscope.__all__) | {"__version__"}
    offenders = []
    for file in sorted(SERVER.rglob("*.py")):
        for module, names, line in imported_names(file):
            where = f"{file.relative_to(SERVER.parent)}:{line}"
            if module == "nanoscope":
                offenders += [f"{where} imports nanoscope.{n}" for n in names
                              if n not in public_names and n not in _submodules()]
            elif not allowed(module) or any(
                    module == f or module.startswith(f + ".") for f in FORBIDDEN):
                offenders.append(f"{where} imports {module}")
    assert not offenders, "the server may import only public library names:\n" + "\n".join(
        offenders)


def _submodules() -> set[str]:
    """`from nanoscope import paths` names a submodule: fine if the module is allowed."""
    return {m.split(".", 1)[1] for m in PUBLIC_MODULES if m.count(".") == 1}


def test_the_lists_do_not_overlap_and_name_real_modules():
    import importlib

    assert not {m for m in PUBLIC_MODULES for f in FORBIDDEN if m == f or m.startswith(f + ".")}
    for module in PUBLIC_MODULES | FORBIDDEN:
        importlib.import_module(module)  # a typo here would silently allow or forbid nothing
