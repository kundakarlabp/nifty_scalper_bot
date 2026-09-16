"""Low-resource operating-mode control for the production ASGI wrapper.

AUTO keeps the full trading engine loaded only around the NSE cash session on
weekdays. QUIET keeps only HTTP/admin controls alive. ACTIVE keeps the engine
loaded continuously. This reduces off-hours CPU/network work; it does not by
itself change the fixed Lightsail bundle price.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from nifty_scalper_bot.admin_dashboard import (
    _check_auth,
    _read_env,
    _restart_service,
    _write_env,
)

IST = ZoneInfo("Asia/Kolkata")
VALID_MODES = {"AUTO", "ACTIVE", "QUIET"}
DEFAULT_START = dt_time(8, 55)
DEFAULT_STOP = dt_time(15, 40)
router = APIRouter()


def operating_mode() -> str:
    """Return canonical configured operating mode."""
    value = (
        _read_env().get("BOT_OPERATING_MODE")
        or os.getenv("BOT_OPERATING_MODE")
        or "AUTO"
    ).strip().upper()
    return value if value in VALID_MODES else "AUTO"


def _clock(name: str, default: dt_time) -> dt_time:
    raw = (
        _read_env().get(name)
        or os.getenv(name)
        or default.strftime("%H:%M")
    ).strip()
    try:
        hour, minute = (int(part) for part in raw.split(":", 1))
        return dt_time(hour, minute)
    except (ValueError, TypeError):
        return default


def auto_window_active(now: datetime | None = None) -> bool:
    """Return whether the weekday engine warm/run window is active."""
    now = now or datetime.now(IST)
    local = now.astimezone(IST)
    if local.weekday() >= 5:
        return False
    clock = local.time().replace(tzinfo=None)
    start = _clock("BOT_AUTO_START_IST", DEFAULT_START)
    stop = _clock("BOT_AUTO_STOP_IST", DEFAULT_STOP)
    return start <= clock < stop


def engine_should_run(now: datetime | None = None) -> bool:
    """Return whether the full trading engine should be loaded."""
    mode = operating_mode()
    return mode == "ACTIVE" or (mode == "AUTO" and auto_window_active(now))


def start_transition_watchdog(
    initial_should_run: bool,
    interval_seconds: int = 30,
) -> threading.Thread:
    """Restart the service only when AUTO crosses an active/quiet boundary."""

    def watch() -> None:
        expected = initial_should_run
        while True:
            time.sleep(max(10, interval_seconds))
            if operating_mode() != "AUTO":
                continue
            current = engine_should_run()
            if current != expected:
                _restart_service()
                return

    thread = threading.Thread(
        target=watch,
        name="operating-mode-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _mode_page(mode: str, running: bool) -> str:
    state = "FULL ENGINE ACTIVE" if running else "QUIET — ADMIN ONLY"
    buttons = []
    choices = (
        ("AUTO", "AUTO (recommended)"),
        ("ACTIVE", "ACTIVE"),
        ("QUIET", "QUIET"),
    )
    for value, label in choices:
        disabled = " disabled" if mode == value else ""
        buttons.append(
            '<form method="post" action="/power/mode">'
            f'<input type="hidden" name="mode" value="{value}">'
            f"<button{disabled}>{label}</button></form>"
        )
    controls = "".join(buttons)
    return f"""<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nifty Bot Power</title>
<style>
body{{font-family:system-ui;background:#0a0e14;color:#e6edf3;
max-width:720px;margin:40px auto;padding:16px}}
.card{{background:#141b24;border:1px solid #263241;border-radius:14px;padding:22px}}
button{{padding:12px 16px;margin:6px;border:0;border-radius:9px;
background:#1f6feb;color:white;font-weight:700}}
button:disabled{{background:#2ea043}}form{{display:inline}}
a{{color:#58a6ff}}.muted{{color:#8b97a6}}
</style>
</head>
<body><div class="card">
<h2>Bot Operating Mode</h2><h3>{state}</h3>
<p><b>Configured:</b> {mode}</p>
<p class="muted">AUTO: full engine 08:55–15:40 IST Monday–Friday; outside that
window only the lightweight API/admin process remains. ACTIVE overrides the
schedule. QUIET immediately unloads trading activity after restart.</p>
{controls}
<p><a href="/admin">Admin dashboard</a> · <a href="/livez">Health</a></p>
<p class="muted">Quiet mode reduces CPU/network work but does not remove the
fixed Lightsail instance charge.</p>
</div></body></html>"""


@router.get("/power", response_class=HTMLResponse)
def power_page(request: Request) -> HTMLResponse:
    """Render the operating-mode control page."""
    _check_auth(request)
    return HTMLResponse(_mode_page(operating_mode(), engine_should_run()))


@router.post("/power/mode")
def set_operating_mode(
    request: Request,
    mode: str = Form(...),
) -> RedirectResponse:
    """Persist a validated operating mode and restart the service."""
    _check_auth(request)
    selected = mode.strip().upper()
    if selected not in VALID_MODES:
        return RedirectResponse("/power", status_code=303)
    _write_env(
        {
            "BOT_OPERATING_MODE": selected,
            "BOT_AUTO_START_IST": "08:55",
            "BOT_AUTO_STOP_IST": "15:40",
        }
    )
    _restart_service()
    return RedirectResponse("/power", status_code=303)
