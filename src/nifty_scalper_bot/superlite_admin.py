"""Independent low-overhead host for the existing Nifty admin controls."""
from __future__ import annotations

import os
import subprocess
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from nifty_scalper_bot import admin_dashboard as dashboard
from nifty_scalper_bot.ops.service_control import memory_snapshot
from nifty_scalper_bot.superlite_admin_core import (
    APP_DIR,
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


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(APP_DIR), "rev-parse", "HEAD"],
            text=True,
            timeout=1.5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


_INITIAL_REVISION = _revision()


def _watch_revision() -> None:
    try:
        interval = max(60, int(os.getenv("BOT_ADMIN_REVISION_CHECK_SECONDS", "120")))
    except ValueError:
        interval = 120
    while True:
        time.sleep(interval)
        current = _revision()
        if _INITIAL_REVISION and current and current != _INITIAL_REVISION:
            os._exit(0)


threading.Thread(target=_watch_revision, daemon=True, name="admin-revision-watch").start()


def _guard(request: Request) -> None:
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        same_origin(request)


def _validated_update() -> tuple[bool, str]:
    subprocess.Popen(
        ["git", "-C", str(APP_DIR), "fetch", "--quiet", "origin", "main"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True, "remote revision refreshed; validated updater runs within two minutes"


def _flash(request: Request) -> str:
    if request.query_params.get("upd") == "ok":
        return (
            '<div class="flash ok">Remote revision refreshed. The validated '
            "updater deploys eligible changes within two minutes.</div>"
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
    data = status_snapshot()
    data["host_memory"] = memory_snapshot()
    return JSONResponse(data, headers={"Cache-Control": "no-store"})
