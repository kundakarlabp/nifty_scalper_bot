"""Super-light server operations dashboard.

The route intentionally exposes only fixed, allow-listed operations.  It is a
browser control plane for a small Lightsail VM, not a general terminal.
"""
from __future__ import annotations

import html
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

try:
    from nifty_scalper_bot.admin_dashboard import _check_auth, _page, _topbar
except Exception:  # pragma: no cover - fallback during partial imports
    def _check_auth(request: Request) -> None:  # noqa: ANN001
        return None

    def _page(body: str) -> str:
        return f"<!doctype html><html><body>{body}</body></html>"

    def _topbar(live_on: bool) -> str:
        return '<div><a href="/admin">Dashboard</a></div>'


router = APIRouter()
APP_DIR = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))
ENV_PATH = Path(os.getenv("BOT_ENV_FILE", str(APP_DIR / ".env")))
SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
FORCE_SYNC_SERVICE = os.getenv("BOT_FORCE_SYNC_SERVICE", "niftybot-force-sync.service")
STREAMLIT_SERVICE = os.getenv("BOT_STREAMLIT_SERVICE_NAME", "niftybot-streamlit")
MAX_OUTPUT = int(os.getenv("OPS_MAX_OUTPUT_CHARS", "60000") or "60000")
_ALLOWED_SERVICES = {SERVICE, FORCE_SYNC_SERVICE, STREAMLIT_SERVICE, "caddy"}


def _run(cmd: Sequence[str], *, timeout: int = 20) -> tuple[int, str]:
    try:
        out = subprocess.run(
            list(cmd),
            cwd=str(APP_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        text = (out.stdout or "") + (out.stderr or "")
        text = text.strip() or "Command completed with no output."
        if len(text) > MAX_OUTPUT:
            text = text[-MAX_OUTPUT:]
        return out.returncode, text
    except subprocess.TimeoutExpired as exc:
        return 124, f"Timed out after {timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"Failed before execution: {exc}"


def _status_summary() -> str:
    commands = [
        ("uptime", ["uptime"]),
        ("memory", ["free", "-h"]),
        ("disk", ["df", "-h", "/"]),
        ("bot", ["systemctl", "is-active", SERVICE]),
        ("auto-update timer", ["systemctl", "is-active", "niftybot-force-sync.timer"]),
        ("commit", ["git", "-C", str(APP_DIR), "log", "--oneline", "-n", "1"]),
    ]
    blocks: list[str] = []
    for title, cmd in commands:
        code, out = _run(cmd, timeout=6)
        blocks.append(f"## {title} (exit={code})\n{out}")
    return "\n\n".join(blocks)


def _render(title: str, output: str) -> HTMLResponse:
    body = (
        f"{_topbar(False)}<div class=wrap>"
        f"<div class=card><h2>{html.escape(title)}</h2>"
        f"<p class=muted><a href='/admin/ops'>Back to Ops</a></p>"
        f"<pre>{html.escape(output)}</pre></div></div>"
    )
    return HTMLResponse(_page(body))


def _button(action: str, label: str, css: str = "blu", confirm: str = "") -> str:
    c = f" onclick=\"return confirm('{html.escape(confirm)}')\"" if confirm else ""
    return (
        "<form method=post action='/admin/ops/action'>"
        f"<input type=hidden name=action value='{html.escape(action)}'>"
        f"<button class={css} type=submit{c}>{html.escape(label)}</button></form>"
    )


@router.get("/admin/ops", response_class=HTMLResponse)
def ops_home(request: Request) -> HTMLResponse:
    _check_auth(request)
    body = f"""{_topbar(False)}<div class=wrap>
    <div class=card><h2>Ops Console</h2>
    <p>Super-light controls for a 2 GB Lightsail host. No live polling and no raw shell.</p>
    <p class=muted>Service: <code>{html.escape(SERVICE)}</code> · Env: <code>{html.escape(str(ENV_PATH))}</code></p>
    </div>

    <div class=card><h2>Force Update / Deployment</h2><div class=row>
    {_button('force_update', 'Force Update from GitHub', 'blu', 'Run force-sync now?')}
    {_button('deploy_status', 'Deploy Status', 'gray')}
    </div><p class=muted>Runs <code>sudo systemctl start {html.escape(FORCE_SYNC_SERVICE)}</code>.</p></div>

    <div class=card><h2>Bot Service</h2><div class=row>
    {_button('bot_start', 'Start Bot', 'blu')}
    {_button('bot_stop', 'Stop Bot', 'red', 'Stop the trading bot service?')}
    {_button('bot_restart', 'Restart Bot', 'amb')}
    {_button('bot_status', 'Bot Status', 'gray')}
    </div></div>

    <div class=card><h2>Server / Aux Services</h2><div class=row>
    {_button('restart_streamlit', 'Restart Streamlit', 'amb')}
    {_button('restart_caddy', 'Restart Caddy', 'amb')}
    {_button('server_reboot', 'Reboot Server', 'red', 'Reboot the server now?')}
    {_button('server_shutdown', 'Shutdown Server OS', 'red', 'This halts Ubuntu but may not stop AWS billing. Continue?')}
    </div></div>

    <div class=card><h2>Resource Checks</h2><div class=row>
    {_button('summary', 'Summary', 'blu')}
    {_button('memory', 'Memory', 'gray')}
    {_button('disk', 'Disk', 'gray')}
    {_button('cpu', 'CPU', 'gray')}
    </div></div>

    <div class=card><h2>Logs</h2>
    <form method=get action='/admin/ops/logs'><div class=grid>
    <div><label>Service</label><input name=service value='{html.escape(SERVICE)}'></div>
    <div><label>Lines</label><input name=lines value='200'></div>
    <div><label>Contains</label><input name=contains placeholder='optional'></div>
    </div><button class=blu type=submit>View/Search Logs</button>
    <a class='btn gray' href='/admin/ops/logs/download?service={html.escape(SERVICE)}&lines=1000'>Download Logs</a></form>
    </div></div>"""
    return HTMLResponse(_page(body))


@router.post("/admin/ops/action", response_class=HTMLResponse)
def ops_action(request: Request, action: str = Form(...)) -> HTMLResponse:
    _check_auth(request)
    action = action.strip()
    actions: dict[str, tuple[str, Sequence[str], int]] = {
        "force_update": ("Force Update", ["sudo", "systemctl", "start", FORCE_SYNC_SERVICE], 12),
        "deploy_status": ("Deploy Status", ["systemctl", "status", FORCE_SYNC_SERVICE, "--no-pager"], 12),
        "bot_start": ("Start Bot", ["sudo", "systemctl", "start", SERVICE], 12),
        "bot_stop": ("Stop Bot", ["sudo", "systemctl", "stop", SERVICE], 12),
        "bot_restart": ("Restart Bot", ["sudo", "systemctl", "restart", SERVICE], 12),
        "bot_status": ("Bot Status", ["systemctl", "status", SERVICE, "--no-pager", "-l"], 12),
        "restart_streamlit": ("Restart Streamlit", ["sudo", "systemctl", "restart", STREAMLIT_SERVICE], 12),
        "restart_caddy": ("Restart Caddy", ["sudo", "systemctl", "restart", "caddy"], 12),
        "server_reboot": ("Reboot Server", ["sudo", "systemctl", "reboot"], 5),
        "server_shutdown": ("Shutdown Server OS", ["sudo", "systemctl", "poweroff"], 5),
        "memory": ("Memory", ["free", "-h"], 8),
        "disk": ("Disk", ["df", "-h"], 8),
        "cpu": ("CPU", ["sh", "-c", "uptime && ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -20"], 8),
    }
    if action == "summary":
        return _render("Summary", _status_summary())
    spec = actions.get(action)
    if spec is None:
        return _render("Rejected", f"Unknown or non-allowlisted action: {action}")
    title, cmd, timeout = spec
    code, out = _run(cmd, timeout=timeout)
    return _render(title, f"exit={code}\n{out}")


@router.get("/admin/ops/logs", response_class=HTMLResponse)
def logs(request: Request, service: str = SERVICE, lines: int = 200, contains: str = "") -> HTMLResponse:
    _check_auth(request)
    service = service if service in _ALLOWED_SERVICES else SERVICE
    lines = max(50, min(int(lines or 200), 3000))
    code, out = _run(["journalctl", "-u", service, "-n", str(lines), "--no-pager", "-o", "short-iso"], timeout=12)
    if contains:
        needle = contains.lower()
        out = "\n".join(line for line in out.splitlines() if needle in line.lower()) or "No matching log lines."
    return _render(f"Logs: {service}", f"exit={code}\n{out}")


@router.get("/admin/ops/logs/download", response_class=PlainTextResponse)
def logs_download(request: Request, service: str = SERVICE, lines: int = 1000, contains: str = "") -> PlainTextResponse:
    _check_auth(request)
    service = service if service in _ALLOWED_SERVICES else SERVICE
    lines = max(50, min(int(lines or 1000), 5000))
    _, out = _run(["journalctl", "-u", service, "-n", str(lines), "--no-pager", "-o", "short-iso"], timeout=15)
    if contains:
        needle = contains.lower()
        out = "\n".join(line for line in out.splitlines() if needle in line.lower()) or "No matching log lines."
    ts = time.strftime("%Y%m%d-%H%M%S")
    return PlainTextResponse(
        out,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{service}-{ts}.log"'},
    )


__all__ = ["router"]
