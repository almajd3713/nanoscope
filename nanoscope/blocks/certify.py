"""Certification of a user block: does it match the reference it was registered with?

    certify("models/gate.py", "Gate")   # runs the file (a worker does this), writes a cert

`nanoscope.blocks.certs` reads the certs without running anything (the API uses it).

The reference is the plain function given to `register_block(reference=fn)`. It is called as
`fn(x, ...)`: the first parameter receives the random input, every other parameter is looked
up by name among the block's parameters and buffers, then among its attributes (an option
such as `scale`). The block is built with random weights, so a zero-initialised parameter
cannot hide a wrong formula.

A cert is stored in `home()/certs/<sha256 of the file's source>.json`. It belongs to that
exact source: change the file and the hash changes, so the badge is `stale` until the block
is certified again.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nanoscope.blocks.registry as registry
from nanoscope import __version__
from nanoscope.blocks.certs import cert_path, read_cert, source_sha256
from nanoscope.fsutil import write_json_atomic

SHAPES = ((2, 8), (1, 3))  # (batch, tokens) of the random inputs
TRIALS = 3
TOLERANCE = 1e-5
D_MODEL = 16


def _load_module(file: Path) -> Any:
    name = f"_nanoscope_certify_{hashlib.sha1(str(file.resolve()).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _bind(reference: Any, module: Any, x: Any) -> list[Any]:
    """The reference's arguments: x, then each other parameter by name."""
    params = dict(module.named_parameters()) | dict(module.named_buffers())
    names = list(inspect.signature(reference).parameters.values())
    if not names:
        raise ValueError("the reference takes no arguments; it must take the input first")
    call: list[Any] = [x]
    for p in names[1:]:
        if p.name in params:
            call.append(params[p.name].detach())
        elif hasattr(module, p.name):
            call.append(getattr(module, p.name))
        elif p.default is not inspect.Parameter.empty:
            call.append(p.default)
        else:
            have = ", ".join(sorted({*params, "(options by name)"}))
            raise ValueError(f"the reference takes {p.name!r}, but the block has no parameter, "
                             f"buffer or attribute with that name (it has: {have})")
    return call


def run_check(cls: Any, reference: Any, d_model: int = D_MODEL) -> dict[str, Any]:
    """`{"passed", "message", "max_abs_diff", "trials", ...}` for a built-in-process check."""
    import torch

    torch.manual_seed(0)
    context = max(b * t for b, t in SHAPES)
    module = cls(d_model=d_model, context_length=context).eval()
    scrambler = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=scrambler) * 0.5)
    generator = torch.Generator().manual_seed(1234)
    worst, trials = 0.0, 0
    for batch, tokens in SHAPES:
        for _ in range(TRIALS):
            x = torch.randn(batch, tokens, d_model, generator=generator)
            shape = list(x.shape)
            try:
                with torch.no_grad():
                    got = module(x)
                    got = got[0] if isinstance(got, tuple) else got
                    want = reference(*_bind(reference, module, x))
            except ValueError as exc:
                return {"passed": False, "message": str(exc), "max_abs_diff": None,
                        "trials": trials}
            except Exception as exc:
                return {"passed": False, "max_abs_diff": None, "trials": trials,
                        "message": f"running on an input of shape {shape} raised "
                                   f"{type(exc).__name__}: {exc}"}
            if got.shape != want.shape:
                return {"passed": False, "max_abs_diff": None, "trials": trials,
                        "message": f"output shape {list(got.shape)} but the reference gives "
                                   f"{list(want.shape)} for an input of shape {shape}"}
            diff = float((got.float() - want.float()).abs().max())
            worst, trials = max(worst, diff), trials + 1
            if not diff <= TOLERANCE:
                return {"passed": False, "max_abs_diff": diff, "trials": trials,
                        "message": f"output differs from the reference: max abs diff "
                                   f"{diff:.3g} on an input of shape {shape} "
                                   f"(tolerance {TOLERANCE:g})"}
    return {"passed": True, "max_abs_diff": worst, "trials": trials,
            "message": f"matches the reference: max abs diff {worst:.3g} over {trials} random "
                       f"inputs (tolerance {TOLERANCE:g})"}


def certify(file: str | Path, name: str) -> dict[str, Any]:
    """Import `file`, check block `name` against its reference, and store the cert. This runs
    the user's code: call it from a worker, never from the API process."""
    file = Path(file)
    module = _load_module(file)
    cls = getattr(module, name, None)
    if cls is None or name not in {b.name for b in registry.all_blocks() if b.user}:
        raise ValueError(f"{file} registers no block named {name!r}")
    reference = registry.reference_for(name)
    if reference is None:
        result: dict[str, Any] = {
            "passed": False, "max_abs_diff": None, "trials": 0,
            "message": "this block has no reference: register it with "
                       "register_block(reference=fn) to certify it"}
        ref_name = None
    else:
        result = run_check(cls, reference)
        ref_name = reference.__qualname__
    sha = source_sha256(file)
    path = cert_path(sha)
    doc = read_cert(path) or {"schema": 1, "nanoscope": __version__, "source_sha256": sha,
                          "file": str(file.resolve()), "results": {}}
    result = {"reference": ref_name, "tolerance": TOLERANCE, "at": _now(), **result}
    doc["results"][name] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, doc)
    return {"block": name, "source_sha256": sha, **result}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
