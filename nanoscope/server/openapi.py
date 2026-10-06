"""The OpenAPI document, written to a file the web client's types are generated from.

    python -m nanoscope.server.openapi docs/openapi.json     (or: make openapi)

It is committed, and a test fails when it is stale, so an API change always shows up in review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from nanoscope.server.app import create_app
from nanoscope.server.settings import Settings


def document() -> dict[str, Any]:
    # loopback settings and no built app: the document must not depend on where it is made
    return create_app(Settings(static_dir=Path("/nonexistent"))).openapi()


def render() -> str:
    return json.dumps(document(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0]) if args else Path("docs/openapi.json")
    target.write_text(render(), encoding="utf-8")
    print(f"wrote {target} ({len(document()['paths'])} paths)")


if __name__ == "__main__":
    main()
