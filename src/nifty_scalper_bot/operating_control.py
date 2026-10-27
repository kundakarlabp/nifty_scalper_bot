"""Purpose: Bound production AUTO/QUIET/ACTIVE operating modes and transitions.

Responsibilities:
- Compute AUTO schedule-target state (08:55–15:40 IST weekdays) while keeping
  operating-mode selection separate from LIVE/SHADOW order authorization.
- Render authenticated AUTO, ACTIVE and QUIET controls on the independent admin
  process and accept changes at /admin/power/mode (owner cookie path /admin).
- Persist changes through the canonical protected environment writer and queue
  a guarded restart; display whether it was queued, never claim activation.
- Queue/retry AUTO boundary transitions through the validated release runner
  without relaxing live position, exit-order, or broker-authentication checks.

Safety: Never restart trading directly or bypass broker-flat release gates.
Off-hours unloading reduces CPU work, not the fixed Lightsail bundle price.
"""

from __future__ import annotations

import html
import logging
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
_LOGGER = logging.getLogger(__name__)


def operating_mode() -> str:
    """Return canonical configured operating mode."""
    value = (
        (
            _read_env().get("BOT_OPERATING_MODE")
            or os.getenv("BOT_OPERATING_MODE")
            or "AUTO"
        )
        .strip()
        .upper()
    )
    return value if value in VALID_MODES else "AUTO"


def _clock(name: str, default: dt_time) -> dt_time:
    raw = (
        _read_env().get(name) or os.getenv(name) or default.strftime("%H:%M")
    ).strip()
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
    return (
        _clock("BOT_AUTO_START_IST", DEFAULT_START)
        <= clock
        < _clock("BOT_AUTO_STOP_IST", DEFAULT_STOP)
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
            if offset == 0:
                label = "Today"
            elif offset == 1:
                label = "Tomorrow"
            else:
                label = candidate.strftime("%a %d %b")
            return f"{label} {start.strftime('%H:%M')} IST → ACTIVE"
    return candidate.strftime("%a %d %b %H:%M IST")


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
                # A rejected queue request must not strand AUTO in the
                # wrong process mode for the entire market session.
                # The existing guarded deployer still owns broker-flat
                # restart authorization.
                result = _restart_service()
                if result.ok:
                    return
                _LOGGER.warning(
                    "AUTO mode transition restart request rejected; "
                    "will retry on next watchdog tick"
                )

    thread = threading.Thread(
        target=watch,
        name="operating-mode-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _mode_controls(
    mode: str, *, return_to: str = "/admin", editable: bool = True
) -> str:
    buttons: list[str] = []
    choices = (("AUTO", "AUTO"), ("ACTIVE", "ACTIVE"), ("QUIET", "QUIET"))
    safe_return = html.escape(return_to, quote=True)
    for value, label in choices:
        disabled = " disabled" if mode == value or not editable else ""
        css = "gray" if value == "QUIET" else "blu" if value == "ACTIVE" else ""
        buttons.append(
            '<form method="post" action="/admin/power/mode" style="display:inline">'
            f'<input type="hidden" name="mode" value="{value}">'
            f'<input type="hidden" name="return_to" value="{safe_return}">'
            f'<button class="{css}" type="submit"{disabled}>'
            f"{label}</button></form>"
        )
    return "".join(buttons)


def admin_power_card(*, editable: bool = True) -> str:
    """Render operating mode controls in the independently hosted admin page.

    The schedule is a requested target, never evidence of a loaded engine.
    Token login unlocks writes; rendering this panel does not contact Kite.
    """
    mode = operating_mode()
    scheduled_active = engine_should_run()
    target = "ACTIVE" if scheduled_active else "QUIET"
    start = _clock("BOT_AUTO_START_IST", DEFAULT_START).strftime("%H:%M")
    stop = _clock("BOT_AUTO_STOP_IST", DEFAULT_STOP).strftime("%H:%M")
    transition = next_auto_transition()
    controls = _mode_controls(mode, editable=editable)
    auth_note = (
        ""
        if editable
        else '<p class="muted">Verify your Zerodha owner session using '
        "the request-token form above to enable mode changes.</p>"
    )
    return (
        '<section class="card" id="operating-mode">'
        "<h2>Operating mode</h2>"
        f"<p>Configured: <b>{html.escape(mode)}</b> · "
        f"Schedule target now: <b>{target}</b></p>"
        f"<p><b>AUTO</b>: engine requested {start}–{stop} IST "
        "Monday–Friday; QUIET outside the window. "
        "AUTO's weekday window is not a holiday calendar; "
        "broker readiness remains a separate requirement. "
        "<b>ACTIVE</b>: manual always-on override. "
        "<b>QUIET</b>: manual control-only override.</p>"
        f'<div class="row">{controls}</div>'
        f"{auth_note}"
        '<p class="muted">Changing mode queues a guarded release. '
        "The live Engine and Zerodha indicators above determine whether "
        "the new state actually became ready.</p>"
        f'<p class="muted">Next AUTO transition: '
        f"{html.escape(transition)}</p></section>"
    )


def _mode_page(mode: str, running: bool) -> str:
    state = "FULL ENGINE ACTIVE" if running else "QUIET — ADMIN ONLY"
    controls = _mode_controls(mode, return_to="/admin/power")
    transition = next_auto_transition()
    return (
        "<!doctype html><html><head>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Nifty Bot Power</title><style>"
        "body{font-family:system-ui;background:#0a0e14;color:#e6edf3;"
        "max-width:720px;margin:40px auto;padding:16px}"
        ".card{background:#141b24;border:1px solid #263241;"
        "border-radius:14px;padding:22px}"
        "button{padding:12px 16px;margin:6px;border:0;border-radius:9px;"
        "background:#1f6feb;color:white;font-weight:700}"
        "button:disabled{background:#2ea043}form{display:inline}"
        "a{color:#58a6ff}.muted{color:#8b97a6}</style></head>"
        '<body><div class="card"><h2>Bot Operating Mode</h2>'
        f"<h3>{state}</h3><p><b>Configured:</b> {mode}</p>"
        '<p class="muted">AUTO: full engine 08:55–15:40 IST '
        "Monday–Friday; ACTIVE overrides the schedule; QUIET keeps only "
        "admin/API alive.</p>"
        f"{controls}<p><b>Next AUTO transition:</b> {transition}</p>"
        '<p><a href="/admin">Admin dashboard</a> · '
        '<a href="/livez">Health</a></p></div></body></html>'
    )


@router.get("/admin/power", response_class=HTMLResponse)
def power_page(request: Request) -> HTMLResponse:
    """Render the operating-mode control page."""
    _check_auth(request)
    return HTMLResponse(_mode_page(operating_mode(), engine_should_run()))


@router.post("/admin/power/mode")
def set_operating_mode(
    request: Request,
    mode: str = Form(...),
    return_to: str = Form("/admin"),
) -> RedirectResponse:
    """Persist a validated operating mode and restart the service."""
    _check_auth(request)
    selected = mode.strip().upper()
    target = return_to if return_to in {"/admin", "/admin/power"} else "/admin"
    if selected not in VALID_MODES:
        return RedirectResponse(target, status_code=303)
    if selected == operating_mode():
        return RedirectResponse(target + "?power=unchanged", status_code=303)
    _write_env(
        {
            "BOT_OPERATING_MODE": selected,
            "BOT_AUTO_START_IST": "08:55",
            "BOT_AUTO_STOP_IST": "15:40",
        }
    )
    restart = _restart_service()
    if restart.ok:
        return RedirectResponse(target + "?power=queued", status_code=303)
    return RedirectResponse(target + "?power=saved_not_applied", status_code=303)
