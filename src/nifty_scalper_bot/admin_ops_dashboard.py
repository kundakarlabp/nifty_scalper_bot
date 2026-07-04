"""Admin operations dashboard for controlled no-SSH Lightsail maintenance.

This module exposes a restricted browser-accessible operations panel for a
single-owner Lightsail deployment. It intentionally uses an allow-list of fixed
commands instead of exposing a raw shell terminal.
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

try:  # Reuse the existing dashboard auth boundary and styling helpers.
    from nifty_scalper_bot.admin_dashboard import _check_auth, _page, _topbar
except Exception:  # pragma: no cover - defensive fallback for partial imports
    def _check_auth(request: Request) -> None:  # noqa: ANN001
        return None

    def _page(body: str) -> str:
        return f"<!doctype html><html><body>{body}</body></html>"

    def _topbar(live_on: bool) -> str:
        return '<div><a href="/admin">Dashboard</a></div>'


router = APIRouter()

APP_DIR = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))
ENV_PATH = Path(os.getenv("BOT_ENV_FILE", str(APP_DIR / ".env")))
SERVICE_NAME = os.getenv("BOT_SERVICE_NAME", "niftybot")
STREAMLIT_SERVICE = os.getenv("BOT_STREAMLIT_SERVICE_NAME", "niftybot-streamlit")
LOG_SERVICE_NAMES = (SERVICE_NAME, STREAMLIT_SERVICE, "caddy")
_MAX_OUTPUT = 80_000
_ALLOWED_CONFIG_FILES = {"env": ENV_PATH, "env_example": APP_DIR / ".env.example"}


def _run(command: Sequence[str], *, timeout: int = 25) -> tuple[int, str]:
    """Run a fixed command and return returncode plus bounded combined output."""
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(APP_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if len(output) > _MAX_OUTPUT:
            output = output[-_MAX_OUTPUT:]
        return completed.returncode, output.strip() or "Command completed with no output."
    except subprocess.TimeoutExpired as exc:
        return 124, f"Command timed out after {timeout}s. Partial output:\n{exc.stdout or ''}\n{exc.stderr or ''}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"Command failed before execution: {exc}"


def _restart_unit(unit: str) -> tuple[int, str]:
    return _run(["sudo", "systemctl", "restart", "--no-block", unit], timeout=10)


def _journal(service: str, *, lines: int = 300, contains: str = "") -> str:
    lines = max(50, min(int(lines or 300), 5000))
    code, out = _run(["journalctl", "-u", service, "-n", str(lines), "--no-pager", "-o", "short-iso"], timeout=15)
    if contains:
        needle = contains.lower()
        out = "\n".join(line for line in out.splitlines() if needle in line.lower()) or "No matching log lines."
    return out if code == 0 else f"journalctl exited {code}\n{out}"


def _git_last_commit() -> str:
    _, out = _run(["git", "-C", str(APP_DIR), "log", "--oneline", "-n", "5"], timeout=10)
    return out


def _render_output(title: str, output: str) -> HTMLResponse:
    body = (
        f"{_topbar(False)}<div class=wrap>"
        f"<div class=card><h2>{html.escape(title)}</h2>"
        f"<p class=muted><a href='/admin/ops'>Back to Server Operations</a></p>"
        f"<pre>{html.escape(output)}</pre></div></div>"
    )
    return HTMLResponse(_page(body))


def _button(action: str, label: str, css: str = "blu", confirm: str = "") -> str:
    confirm_attr = f" onclick=\"return confirm('{html.escape(confirm)}')\"" if confirm else ""
    return (
        f"<form method=post action='/admin/ops/action'>"
        f"<input type=hidden name=action value='{html.escape(action)}'>"
        f"<button class={css} type=submit{confirm_attr}>{html.escape(label)}</button></form>"
    )


@router.get("/admin/ops", response_class=HTMLResponse)
def ops_dashboard(request: Request) -> HTMLResponse:
    _check_auth(request)
    env_hint = html.escape(str(ENV_PATH))
    current = html.escape(_git_last_commit())
    body = f"""{_topbar(False)}<div class=wrap>
    <div class=card><h2>Server Operations</h2>
    <p>Restricted no-SSH controls for this Lightsail Ubuntu instance. No arbitrary shell is exposed.</p>
    <p class=muted>Env file: <code>{env_hint}</code></p></div>

    <div class=card><h2>System</h2><div class=row>
    {_button('restart_server', 'Restart Server', 'amb', 'Reboot the whole server now?')}
    {_button('shutdown_server', 'Shutdown Server OS', 'red', 'This halts Ubuntu but may not stop AWS billing. Continue?')}
    {_button('reboot_bot', 'Reboot Bot', 'amb')}
    {_button('restart_streamlit', 'Restart Streamlit', 'amb')}
    {_button('restart_caddy', 'Restart Caddy', 'amb')}
    </div></div>

    <div class=card><h2>Disk / Memory / CPU</h2><div class=row>
    {_button('disk_usage', 'Disk Usage')}
    {_button('memory_usage', 'Memory Usage')}
    {_button('cpu_usage', 'CPU Usage')}
    </div></div>

    <div class=card><h2>Git</h2><p class=muted>Latest commits:</p><pre>{current}</pre><div class=row>
    {_button('git_status', 'Git Status')}
    {_button('git_pull', 'Git Pull + Restart Bot', 'blu', 'Pull latest main and restart bot?')}
    {_button('rollback', 'Rollback One Commit + Restart', 'red', 'Hard reset to HEAD~1 and restart bot?')}
    </div></div>

    <div class=card><h2>Logs</h2>
    <form method=get action='/admin/ops/logs'><div class=grid>
    <div><label>Service</label><input name=service value='{html.escape(SERVICE_NAME)}'></div>
    <div><label>Lines</label><input name=lines value='300'></div>
    <div><label>Search contains</label><input name=contains placeholder='optional text'></div>
    </div><button class=blu type=submit>View / Search Logs</button>
    <a class='btn gray' href='/admin/ops/logs/download?service={html.escape(SERVICE_NAME)}&lines=1000'>Download Logs</a></form>
    </div>

    <div class=card><h2>Files</h2><div class=row>
    <a class='btn blu' href='/admin/ops/file?target=env'>Edit .env</a>
    <a class='btn gray' href='/admin/ops/file?target=env_example'>View .env.example</a>
    </div>
    <form method=post action='/admin/ops/upload-config'>
    <label>Upload / replace config content into .env</label>
    <textarea name=content rows=12 style='width:100%;background:#0a1019;color:#e6edf3;border:1px solid #222d3a;border-radius:9px;padding:12px'></textarea>
    <button class=red type=submit onclick="return confirm('Replace the env file content?')">Upload Config to .env</button>
    </form></div>
    </div>"""
    return HTMLResponse(_page(body))


@router.post("/admin/ops/action", response_class=HTMLResponse)
def ops_action(request: Request, action: str = Form(...)) -> HTMLResponse:
    _check_auth(request)
    action = action.strip()
    actions: dict[str, tuple[str, Sequence[str], int]] = {
        "restart_server": ("Restart Server", ["sudo", "systemctl", "reboot"], 5),
        "shutdown_server": ("Shutdown Server OS", ["sudo", "systemctl", "poweroff"], 5),
        "restart_caddy": ("Restart Caddy", ["sudo", "systemctl", "restart", "caddy"], 15),
        "disk_usage": ("Disk Usage", ["df", "-h"], 10),
        "memory_usage": ("Memory Usage", ["free", "-h"], 10),
        "cpu_usage": ("CPU Usage", ["sh", "-c", "uptime && ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -20"], 10),
        "git_status": ("Git Status", ["git", "-C", str(APP_DIR), "status", "--short", "--branch"], 15),
    }
    if action == "reboot_bot":
        code, out = _restart_unit(SERVICE_NAME)
        return _render_output("Reboot Bot", f"exit={code}\n{out}")
    if action == "restart_streamlit":
        code, out = _restart_unit(STREAMLIT_SERVICE)
        return _render_output("Restart Streamlit", f"exit={code}\n{out}")
    if action == "git_pull":
        code, out = _run(["git", "-C", str(APP_DIR), "pull", "--ff-only", "origin", "main"], timeout=60)
        if code == 0:
            r_code, r_out = _restart_unit(SERVICE_NAME)
            out = f"{out}\n\nRestart exit={r_code}\n{r_out}"
        return _render_output("Git Pull", f"exit={code}\n{out}")
    if action == "rollback":
        code, out = _run(["git", "-C", str(APP_DIR), "reset", "--hard", "HEAD~1"], timeout=30)
        if code == 0:
            r_code, r_out = _restart_unit(SERVICE_NAME)
            out = f"{out}\n\nRestart exit={r_code}\n{r_out}"
        return _render_output("Rollback", f"exit={code}\n{out}")
    spec = actions.get(action)
    if spec is None:
        return _render_output("Unknown action", f"Rejected non-allowlisted action: {action}")
    title, cmd, timeout = spec
    code, out = _run(cmd, timeout=timeout)
    return _render_output(title, f"exit={code}\n{out}")


@router.get("/admin/ops/logs", response_class=HTMLResponse)
def ops_logs(request: Request, service: str = SERVICE_NAME, lines: int = 300, contains: str = "") -> HTMLResponse:
    _check_auth(request)
    if service not in LOG_SERVICE_NAMES:
        service = SERVICE_NAME
    return _render_output(f"Logs: {service}", _journal(service, lines=lines, contains=contains))


@router.get("/admin/ops/logs/download", response_class=PlainTextResponse)
def ops_logs_download(request: Request, service: str = SERVICE_NAME, lines: int = 1000, contains: str = "") -> PlainTextResponse:
    _check_auth(request)
    if service not in LOG_SERVICE_NAMES:
        service = SERVICE_NAME
    text = _journal(service, lines=lines, contains=contains)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return PlainTextResponse(
        text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{service}-logs-{ts}.txt"'},
    )


@router.get("/admin/ops/file", response_class=HTMLResponse)
def edit_file(request: Request, target: str = "env") -> HTMLResponse:
    _check_auth(request)
    path = _ALLOWED_CONFIG_FILES.get(target, ENV_PATH)
    content = path.read_text(errors="replace") if path.exists() else ""
    readonly = "readonly" if target != "env" else ""
    save_button = "" if readonly else "<button class=blu type=submit>Save .env</button>"
    body = f"""{_topbar(False)}<div class=wrap>
    <div class=card><h2>Edit {html.escape(target)}</h2><p class=muted>{html.escape(str(path))}</p>
    <form method=post action='/admin/ops/file'>
    <input type=hidden name=target value='{html.escape(target)}'>
    <textarea name=content rows=28 {readonly} style='width:100%;background:#05080d;color:#e6edf3;border:1px solid #222d3a;border-radius:9px;padding:12px;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace'>{html.escape(content)}</textarea>
    {save_button}<a class='btn gray' href='/admin/ops'>Back</a></form></div></div>"""
    return HTMLResponse(_page(body))


@router.post("/admin/ops/file", response_class=HTMLResponse)
def save_file(request: Request, target: str = Form("env"), content: str = Form("")) -> HTMLResponse:
    _check_auth(request)
    if target != "env":
        return _render_output("File save rejected", "Only the external .env file is writable from the dashboard.")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(content.rstrip() + "\n")
    os.chmod(ENV_PATH, 0o600)
    return _render_output("File saved", f"Updated {ENV_PATH}\nRestart the bot to apply changes.")


@router.post("/admin/ops/upload-config", response_class=HTMLResponse)
def upload_config(request: Request, content: str = Form("")) -> HTMLResponse:
    _check_auth(request)
    if not content.strip():
        return _render_output("Upload config", "No content supplied; .env was not changed.")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(content.rstrip() + "\n")
    os.chmod(ENV_PATH, 0o600)
    return _render_output("Upload config", f"Replaced {ENV_PATH}. Restart the bot to apply changes.")


__all__ = ["router"]
