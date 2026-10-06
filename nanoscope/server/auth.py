"""Access control. On loopback there is none: only this machine can reach the server. Beyond
loopback every request needs the token, as a bearer header or as the HttpOnly cookie that
`GET /login?token=...` sets. This keeps a stray port scan from starting training jobs; it is
not a user system (a hosted, multi-user deployment is a different plan)."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from nanoscope.server.settings import Settings

COOKIE = "nanoscope_token"
OPEN_PATHS = ("/api/health", "/login")  # a load balancer's probe and the sign-in itself


def _same(given: str | None, token: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), token.encode())  # type: ignore[union-attr]


def install(app: FastAPI, settings: Settings) -> None:
    if settings.loopback:
        return
    token = settings.token or ""

    @app.get("/login", include_in_schema=False)
    def login(token: str = "") -> Response:  # noqa: B008
        if not _same(token, settings.token or ""):
            return JSONResponse({"detail": "wrong token"}, status_code=401)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=30 * 86400)
        return response

    @app.middleware("http")
    async def require_token(request: Request, call_next: Any) -> Response:
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        bearer = header[7:] if header.lower().startswith("bearer ") else None
        if _same(bearer, token) or _same(request.cookies.get(COOKIE), token):
            return await call_next(request)
        return JSONResponse({"detail": "sign in with the token (see `nanoscope serve`)"},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
