"""One logger for everything nanoscope says: `logging.getLogger("nanoscope")`.

By default it prints `[nanoscope] message` to stdout, as before. To route it elsewhere,
add your own handler to the logger (it does not propagate to the root logger, so you won't
see each message twice if you also configure logging).
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("nanoscope")


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is when a message is emitted (notebooks, test capture)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


if not logger.handlers:
    _handler = _StdoutHandler()
    _handler.setFormatter(logging.Formatter("[nanoscope] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def info(message: str) -> None:
    logger.info(message)
