"""Reading certs: the certification state of a user block, without running any user code.

    state("models/gate.py", "Gate")   # certified | failed | stale | uncertified

`nanoscope.blocks.certify` writes them (it imports the user's file, so only a worker runs it).
A cert is stored in `home()/certs/<sha256 of the file's source>.json` and belongs to that exact
source: change the file and the hash changes, so the badge is `stale` until it is certified again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nanoscope import paths
from nanoscope.schemas.upgrade import read_json


def source_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certs_dir() -> Path:
    base = paths.home()
    return base / "certs" if base else Path.home() / ".nanoscope" / "certs"


def cert_path(sha: str) -> Path:
    return certs_dir() / f"{sha}.json"


def read_cert(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path, "cert")
    except (OSError, ValueError):
        return None


def state(file: str | Path, name: str) -> dict[str, Any]:
    """`{"state": ..., ...}` for one block. certified/failed carry the stored result; stale means
    the block was certified (or failed) for an older version of the file."""
    sha = source_sha256(file)
    doc = read_cert(cert_path(sha))
    if doc and name in doc["results"]:
        result = doc["results"][name]
        return {"state": "certified" if result["passed"] else "failed", **result}
    resolved = str(Path(file).resolve())
    for other in sorted(certs_dir().glob("*.json")) if certs_dir().exists() else []:
        old = read_cert(other)
        if old and old["file"] == resolved and name in old["results"] \
                and old["source_sha256"] != sha:
            return {"state": "stale", **old["results"][name]}
    return {"state": "uncertified"}
