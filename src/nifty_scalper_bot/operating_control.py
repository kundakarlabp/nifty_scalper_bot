"""Low-resource operating-mode control for the production ASGI wrapper.

AUTO keeps the full trading engine loaded only around the NSE cash session on
weekdays. QUIET keeps only HTTP/admin controls alive. ACTIVE keeps the engine
loaded continuously. This reduces off-hours CPU/network work; it does not by
itself change the fixed Lightsail bundle price.
"""

from __future__ import annotations

import html
import os
import threading
import time
from datetime import datetime, timedelta
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
    raw = (_read_env().get(name) or os.getenv(name) or default.strftime("%H:%M")).strip()
    try:
        hour, minute = (int(part) for part in raw.split(":", 1))
        return dt_time(hour, minute)
    except (ValueError, TypeError):
        return default


def auto_window_active(now: datetime | None = None) -> bool:
    """Return whether the weekday engine warm/run window is active."""
    local = (now or datetime.now(IST)).astimezone(IST)
    if local.weekday() >= 5:
        return False
    clock = local.time().replace(tzinfo=None)
    return _clock("BOT_AUTO_START_IST", DEFAULT_START) <= clock < _clock(
        "BOT_AUTO_STOP_IST", DEFAULT_STOP
    )


def engine_should_run(now: datetime | None = None) -> bool:
    """Return whether the full trading engine should be loaded."""
    mode = operating_mode()
    return mode == "ACTIVE" or (mode == "AUTO" and auto_window_active(now))


def next_auto_transition(now: datetime | None = None) -> str:
    """Return a concise IST label for the next AUTO start/stop transition."""
    local = (now or datetime.now(IST)).astimezone(IST)
    start = _clock("BOT_AUTO_START_IST", DEFAULT_START)
    stop = _clock("BOT_AUTO_STOP_IST", DEFAULT_STOP)
    if local.weekday() < 5 and auto_window_active(local):
        return f"Today {stop.strftime('%H:%M')} IST → QUIET"
    candidate = local
    for offset in range(0, 8):
        day = (local + timedelta(days=offset)).date()
        if day.weekday() >= 5:
            continue
        candidate = datetime.combine(day, start, tzinfo=IST)
        if candidate > local:
            label = "Today" if offset == 0 else "Tomorrow" if offset == 1 else candidate.strftime("%a %d %b")
            return f"{label} {start.strftime('%H:%M')} IST → ACTIVE"
    return candidate.strftime("%a %d %b %H:%M IST")


def start_transition_watchdog(initial_should_run: bool, interval_seconds: int = 30) -> threading.Thread:
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

    thread = threading.Thread(target=watch, name="operating-mode-watchdog", daemon=True)
    thread.start()
    return thread


def _mode_controls(mode: str, *, return_to: str = "/admin") -> str:
    buttons: list[str] = []
    choices = (("AUTO", "AUTO"), ("ACTIVE", "ACTIVE"), ("QUIET", "QUIET"))
    for value, label in choices:
        disabled = " disabled" if mode == value else ""
        css = "gray" if value == "QUIET" else "blu" if value == "ACTIVE" else ""
        buttons.append(
            '<form method="post" action="/power/mode" style="display:inline">'
            f'<input type="hidden" name="mode" value="{value}">'
            f'<input type="hidden" name="return_to" value="{html.escape(return_to, quote=True)}">'
            f'<button class="{css}" type="submit"{disabled}>{label}</button></form>'
        )
    return "".join(buttons)


def admin_power_card() -> str:
    """Render the canonical operating-mode card for the existing admin dashboard."""
    mode = operating_mode()
    running = engine_should_run()
    state = "FULL ENGINE ACTIVE" if running else "QUIET — ADMIN ONLY"
    state_class = "on" if running else "off"
    start = _clock("BOT_AUTO_START_IST", DEFAULT_START).strftime("%H:%M")
    stop = _clock("BOT_AUTO_STOP_IST", DEFAULT_STOP).strftime("%H:%M")
    return f"""<div class="card session"><h2>Operating Mode &amp; Off-hours Power</h2>
    <p><span class="pill {state_class}">● {state}</span> &nbsp; Configured: <b>{mode}</b></p>
    <p><b>AUTO (recommended)</b> runs the full trading engine {start}–{stop} IST on weekdays and keeps only the lightweight admin/API process outside that window.</p>
    <div class="row">{_mode_controls(mode)}</div>
    <p class="muted" style="margin-top:12px">Next AUTO transition: <b>{next_auto_transition()}</b>. ACTIVE is a manual always-on override; QUIET is a manual sleep override. Changing mode restarts the bot service once.</p>
    </div>"""


def install_admin_power_card() -> None:
    """Inject the operating card into the legacy admin page without duplicating its routes."""
    from nifty_scalper_bot import admin_dashboard

    if getattr(admin_dashboard, "_OPERATING_CARD_INSTALLED", False):
        return
    original_page = admin_dashboard._page

    def page_with_operating_card(body: str) -> str:
        marker = '<div class=wrap>'
        if marker in body:
            body = body.replace(marker, marker + admin_power_card(), 1)
        return original_page(body)

    admin_dashboard._page = page_with_operating_card
    admin_dashboard._OPERATING_CARD_INSTALLED = True


def _mode_page(mode: str, running: bool) -> str:
    state = "FULL ENGINE ACTIVE" if running else "QUIET — ADMIN ONLY"
    return f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nifty Bot Power</title><style>body{{font-family:system-ui;background:#0a0e14;color:#e6edf3;max-width:720px;margin:40px auto;padding:16px}}.card{{background:#141b24;border:1px solid #263241;border-radius:14px;padding:22px}}button{{padding:12px 16px;margin:6px;border:0;border-radius:9px;background:#1f6feb;color:white;font-weight:700}}button:disabled{{background:#2ea043}}form{{display:inline}}a{{color:#58a6ff}}.muted{{color:#8b97a6}}</style></head>
<body><div class="card"><h2>Bot Operating Mode</h2><h3>{state}</h3><p><b>Configured:</b> {mode}</p>
<p class="muted">AUTO: full engine 08:55–15:40 IST Monday–Friday; ACTIVE overrides the schedule; QUIET keeps only admin/API alive.</p>
{_mode_controls(mode, return_to='/power')}<p><b>Next AUTO transition:</b> {next_auto_transition()}</p>
<p><a href="/admin">Admin dashboard</a> · <a href="/livez">Health</a></p></div></body></html>"""


@router.get("/power", response_class=HTMLResponse)
def power_page(request: Request) -> HTMLResponse:
    """Render the operating-mode control page."""
    _check_auth(request)
    return HTMLResponse(_mode_page(operating_mode(), engine_should_run()))


@router.post("/power/mode")
def set_operating_mode(
    request: Request,
    mode: str = Form(...),
    return_to: str = Form("/admin"),
) -> RedirectResponse:
    """Persist a validated operating mode and restart the service."""
    _check_auth(request)
    selected = mode.strip().upper()
    target = return_to if return_to in {"/admin", "/power"} else "/admin"
    if selected not in VALID_MODES:
        return RedirectResponse(target, status_code=303)
    _write_env({
        "BOT_OPERATING_MODE": selected,
        "BOT_AUTO_START_IST": "08:55",
        "BOT_AUTO_STOP_IST": "15:40",
    })
    _restart_service()
    return RedirectResponse(target, status_code=303)
