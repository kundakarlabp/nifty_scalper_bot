"""Browser admin dashboard for non-technical operation on a plain VM (Lightsail).

Provides, over HTTP (password-protected):
  - a form to view/update credentials (API key/secret/access token, Telegram, flags)
    stored in the .env file the service loads at boot
  - a daily "update access token" quick field
  - a live log viewer (tail of the service log)
  - a one-click "restart bot" button (re-execs the service via systemd)

This removes the need to touch the terminal after the one-time setup.

Security: every route requires the ADMIN_PASSWORD (set in .env). Served only
over the instance's own port; put it behind the firewall / a single allowed
port. Never commit real secrets — they live only in the .env on the box.
"""
from __future__ import annotations

import os
import re
import secrets
import subprocess
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse

router = APIRouter()

# .env the service sources at boot (same file the Procfile/systemd unit loads).
ENV_PATH = Path(os.getenv("BOT_ENV_FILE", "/home/ubuntu/nifty_scalper_bot/.env"))
LOG_PATH = Path(os.getenv("BOT_LOG_FILE", "/home/ubuntu/nifty_scalper_bot/bot.log"))
SERVICE_NAME = os.getenv("BOT_SERVICE_NAME", "niftybot")

# Fields shown in the dashboard. (label, env_key, is_secret)
FIELDS: list[tuple[str, str, bool]] = [
    ("Zerodha API Key", "KITE_API_KEY", False),
    ("Zerodha API Secret", "KITE_API_SECRET", True),
    ("Zerodha Access Token (update daily)", "KITE_ACCESS_TOKEN", True),
    ("Telegram Bot Token", "TELEGRAM_BOT_TOKEN", True),
    ("Telegram Chat ID", "TELEGRAM_CHAT_ID", False),
    ("Telegram Allowed ID", "TELEGRAM_ALLOWED_ID", False),
    ("Live Trading (true/false)", "ENABLE_LIVE", False),
    ("Execution Mode (LIVE/SHADOW)", "EXECUTION_MODE", False),
]

_SESSIONS: set[str] = set()


def _read_env() -> dict[str, str]:
    data: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def _write_env(updates: dict[str, str]) -> None:
    """Merge *updates* into the .env file, preserving other keys/comments."""
    existing_lines: list[str] = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text("\n".join(out) + "\n")
    os.chmod(ENV_PATH, 0o600)


def _check_auth(request: Request) -> None:
    token = request.cookies.get("admin_session", "")
    if token not in _SESSIONS:
        raise HTTPException(status_code=303, headers={"Location": "/admin/login"})


def _admin_password() -> str:
    pw = os.getenv("ADMIN_PASSWORD", "").strip()
    if not pw:
        # Fall back to the value persisted in .env so it survives restarts.
        pw = _read_env().get("ADMIN_PASSWORD", "").strip()
    return pw


_PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Nifty Bot Admin</title><style>
body{{font-family:system-ui,Arial;background:#0d1117;color:#e6edf3;margin:0;padding:16px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:0 auto 16px;max-width:720px}}
h2{{margin:0 0 12px}} label{{display:block;font-size:13px;color:#9da7b3;margin:10px 0 4px}}
input{{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:15px}}
button{{margin-top:14px;padding:11px 16px;border:0;border-radius:8px;background:#238636;color:#fff;font-size:15px;font-weight:600;cursor:pointer}}
button.alt{{background:#1f6feb}} button.warn{{background:#9e6a03}}
a{{color:#58a6ff}} pre{{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px;max-height:480px;overflow:auto;font-size:12px;white-space:pre-wrap}}
.row{{display:flex;gap:10px;flex-wrap:wrap}} .row form{{flex:1}}
.ok{{color:#3fb950}} .badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px}}
</style></head><body>{body}</body></html>"""


@router.get("/admin/login", response_class=HTMLResponse)
def login_page() -> HTMLResponse:
    body = """<div class=card><h2>Nifty Bot Admin</h2>
    <form method=post action="/admin/login">
    <label>Password</label><input type=password name=password autofocus>
    <button type=submit>Sign in</button></form></div>"""
    return HTMLResponse(_PAGE.format(body=body))


@router.post("/admin/login")
def login(password: str = Form(...)) -> RedirectResponse:
    expected = _admin_password()
    if not expected or not secrets.compare_digest(password, expected):
        return RedirectResponse("/admin/login?e=1", status_code=303)
    token = secrets.token_urlsafe(32)
    _SESSIONS.add(token)
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie("admin_session", token, httponly=True, max_age=86400, samesite="lax")
    return resp


@router.get("/admin", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    _check_auth(request)
    env = _read_env()
    rows = ""
    for label, key, is_secret in FIELDS:
        val = env.get(key, "")
        shown = ("•" * 8 if val else "") if is_secret else val
        rows += f"<label>{label} <span class=badge>{key}</span></label>"
        rows += f'<input name="{key}" value="{shown}" placeholder="(unchanged)" {"type=text" if not is_secret else ""}>'
    body = f"""
    <div class=card><h2>Credentials & Settings</h2>
    <p style="font-size:13px;color:#9da7b3">Leave a secret field showing dots to keep it unchanged. Type a new value to replace it. Save, then Restart Bot.</p>
    <form method=post action="/admin/save">{rows}<button type=submit>Save settings</button></form></div>

    <div class=card><h2>Daily Access Token</h2>
    <p style="font-size:13px;color:#9da7b3">Paste the fresh Zerodha access token each morning and click. This updates it and restarts the bot in one step.</p>
    <form method=post action="/admin/token">
    <label>New Access Token</label><input name="KITE_ACCESS_TOKEN" autofocus>
    <button class=alt type=submit>Update token &amp; restart</button></form></div>

    <div class=card><h2>Controls</h2><div class=row>
    <form method=post action="/admin/restart"><button class=warn type=submit>Restart Bot</button></form>
    <form method=get action="/admin/logs"><button class=alt type=submit>View Logs</button></form>
    <form method=get action="/health"><button type=submit>Health</button></form>
    </div></div>"""
    return HTMLResponse(_PAGE.format(body=body))


async def _collect_form(request: Request) -> dict[str, str]:
    form = await request.form()
    updates: dict[str, str] = {}
    for _, key, is_secret in FIELDS:
        if key not in form:
            continue
        val = str(form[key]).strip()
        # Skip masked/unchanged secret placeholders.
        if is_secret and (val == "" or set(val) <= {"•"}):
            continue
        if val == "":
            continue
        updates[key] = val
    return updates


@router.post("/admin/save")
async def save_async(request: Request) -> RedirectResponse:
    _check_auth(request)
    updates = await _collect_form(request)
    if updates:
        _write_env(updates)
    return RedirectResponse("/admin?saved=1", status_code=303)


@router.post("/admin/token")
async def update_token(request: Request, KITE_ACCESS_TOKEN: str = Form(...)) -> RedirectResponse:
    _check_auth(request)
    tok = KITE_ACCESS_TOKEN.strip()
    if tok:
        _write_env({"KITE_ACCESS_TOKEN": tok, "ZERODHA_ACCESS_TOKEN": tok, "BROKER_ACCESS_TOKEN": tok})
        _restart_service()
    return RedirectResponse("/admin?token=1", status_code=303)


@router.post("/admin/restart")
def restart(request: Request) -> RedirectResponse:
    _check_auth(request)
    _restart_service()
    return RedirectResponse("/admin?restart=1", status_code=303)


@router.get("/admin/logs", response_class=PlainTextResponse)
def logs(request: Request, lines: int = 400) -> PlainTextResponse:
    _check_auth(request)
    lines = max(50, min(int(lines or 400), 5000))
    # Prefer journald (systemd); fall back to a log file.
    try:
        out = subprocess.run(
            ["journalctl", "-u", SERVICE_NAME, "-n", str(lines), "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return PlainTextResponse(out.stdout)
    except Exception:
        pass
    if LOG_PATH.exists():
        try:
            content = LOG_PATH.read_text(errors="replace").splitlines()[-lines:]
            return PlainTextResponse("\n".join(content))
        except Exception as exc:  # noqa: BLE001
            return PlainTextResponse(f"log read error: {exc}")
    return PlainTextResponse("no logs found yet")


def _restart_service() -> None:
    """Restart the systemd service so new .env values take effect."""
    try:
        subprocess.run(["sudo", "systemctl", "restart", SERVICE_NAME], timeout=15, check=False)
    except Exception:
        # Fallback: exit so the process manager relaunches us with fresh env.
        os._exit(0)


