"""How the server was started."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
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


def token_path() -> Path:
    from nanoscope import paths

    return paths.server_dir() / "token"


def ensure_token(environ: Mapping[str, str] | None = None) -> str:
    """The access token for a server that listens beyond this machine: $NANOSCOPE_TOKEN if set,
    else the one saved in `server/token` (made on first use, readable only by you)."""
    import os
    import secrets

    env: Mapping[str, str] = os.environ if environ is None else environ
    if env.get("NANOSCOPE_TOKEN"):
        return env["NANOSCOPE_TOKEN"]
    path = token_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return token


def login_url(settings: Settings) -> str:
    host = "localhost" if settings.host in ("0.0.0.0", "::") else settings.host
    return f"http://{host}:{settings.port}/login?token={settings.token}"


def check(settings: Settings) -> None:
    """Refuse to listen beyond loopback without a token."""
    if not settings.loopback and not settings.token:
        raise ValueError(
            f"refusing to listen on {settings.host} without a token: anyone who can reach "
            "this port could run training jobs on your machine. Use --host 127.0.0.1, or "
            "start with a token (nanoscope serve creates one for you).")
