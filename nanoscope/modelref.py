"""Naming a model class so a run can be rebuilt later, in another process.

A ref is `module:Qualname` for importable classes (`nanoscope.models.bigram:Bigram`) and
`/path/to/file.py:Qualname` for classes loaded from a file. Classes defined in a notebook or
`__main__` can't be found again: their source is saved next to the run instead.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

FILE_MODULES = ("_user_model", "_nanoscope_study_", "_nanoscope_ref_")


def source_file(cls: Any) -> Path | None:
    module = sys.modules.get(cls.__module__)
    file = getattr(module, "__file__", None)
    return Path(file).resolve() if file else None


def model_ref(cls: Any) -> tuple[str, bool]:
    """(ref, rebuildable). Classes from `__main__` or a notebook get a ref but can't be rebuilt."""
    if cls.__module__ == "__main__" or source_file(cls) is None:
        return f"__main__:{cls.__qualname__}", False
    if cls.__module__.startswith(FILE_MODULES):
        return f"{source_file(cls)}:{cls.__qualname__}", True
    try:
        importable = importlib.util.find_spec(cls.__module__) is not None
    except (ImportError, ValueError):
        importable = False
    if importable:
        return f"{cls.__module__}:{cls.__qualname__}", True
    return f"{source_file(cls)}:{cls.__qualname__}", True


def source_sha256(cls: type) -> str | None:
    """Hash of the file that defines the class (or of the class's own source), if there is one."""
    file = source_file(cls)
    if file is not None and file.exists():
        return hashlib.sha256(file.read_bytes()).hexdigest()
    try:
        return hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
    except (OSError, TypeError):
        return None


def class_source(cls: type) -> str | None:
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        return None


def load_class(ref: str) -> type:
    """The class a ref names. Importing a file runs it, like `nanoscope run file.py:Class`."""
    if ":" not in ref:
        raise ValueError(f"model ref must be module:Class or file.py:Class, got {ref!r}")
    where, qualname = ref.rsplit(":", 1)
    if where == "__main__":
        raise ValueError(
            f"{ref} was defined in a notebook or script, so it can't be found again. "
            "Import the class yourself and pass it in."
        )
    if where.endswith(".py"):
        path = Path(where)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        name = f"_nanoscope_ref_{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load a model from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # No __pycache__ next to the file: a record-mode study needs a clean git tree.
        write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        finally:
            sys.dont_write_bytecode = write_bytecode
    else:
        module = importlib.import_module(where)
    obj: Any = module
    for part in qualname.split("."):
        if not hasattr(obj, part):
            raise AttributeError(f"{qualname} not found in {where}")
        obj = getattr(obj, part)
    return obj
