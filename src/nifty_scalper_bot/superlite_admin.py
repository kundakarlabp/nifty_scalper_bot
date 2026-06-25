"""Independent low-overhead host for the existing Nifty admin controls."""
from __future__ import annotations

import html

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from nifty_scalper_bot import admin_dashboard as dashboard
from nifty_scalper_bot.superlite_admin_core import (
    ENGINE_SERVICE,
    bounded_logs,
    read_env,
    restart,
    same_origin,
    status_snapshot,
    write_env,
)
from nifty_scalper_bot.superlite_admin_style import STYLE

_ORIGINAL_FLASH = dashboard._flash


def _guard(request: Request) -> None:
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        same_origin(request)


def _validated_update() -> tuple[bool, str]:
    return True, "automatic validated updater checks every two minutes"


def _flash(request: Request) -> str:
    if request.query_params.get("upd") == "ok":
        return (
            '<div class="flash ok">Update check queued. The validated updater '
            "checks the main branch within two minutes.</div>"
        )
    return _ORIGINAL_FLASH(request)


def _logs(
    lines: int,
    since: str = "",
    until: str = "",
    contains: str = "",
    clean: bool = True,
) -> str:
    del since, until, clean
    return bounded_logs(lines, contains)


# Keep the familiar dashboard and forms, but run them outside the engine and
# replace blocking helpers with bounded control-plane implementations.
dashboard._check_auth = _guard
dashboard._read_env = read_env
dashboard._write_env = write_env
dashboard._git_update = _validated_update
dashboard._gather_logs = _logs
dashboard._restart_service = lambda: restart(ENGINE_SERVICE)
dashboard._flash = _flash
dashboard._CSS = STYLE
dashboard.FIELDS = [(label, key, True) for label, key, _secret in dashboard.FIELDS]

app = FastAPI(title="Nifty Bot Admin", docs_url=None, redoc_url=None)
app.include_router(dashboard.router)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/admin")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/api/status")
def status() -> JSONResponse:
    return JSONResponse(status_snapshot(), headers={"Cache-Control": "no-store"})
