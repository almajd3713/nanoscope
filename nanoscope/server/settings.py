"""How the server was started."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path

LOOPBACK_NAMES = ("localhost",)


def is_loopback(host: str) -> bool:
    """True for 127.0.0.0/8, ::1 and localhost: addresses only this machine can reach."""
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    token: str | None = None  # required to bind beyond loopback (see auth)
    worker: str | None = None  # "cpu": start a local worker subprocess with the server
    static_dir: Path | None = None  # a built SPA to serve at /

    @property
    def loopback(self) -> bool:
        return is_loopback(self.host)
