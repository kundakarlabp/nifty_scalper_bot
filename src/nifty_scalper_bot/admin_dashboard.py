"""Browser admin dashboard for non-technical operation on the Lightsail VM.

The daily Zerodha session flow is deliberately small and canonical:
- open the official Kite login from this dashboard;
- Zerodha redirects the browser back here with request_token when the app redirect
  URL is configured to /admin/kite/callback;
- exchange request_token for access_token, verify it, persist aliases, restart.

A paste fallback accepts either the raw request token or the complete redirect URL.
The dashboard never collects or stores the Zerodha login password or TOTP secret.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from nifty_scalper_bot.superlite_admin_core import bounded_logs, quick_overview

from nifty_scalper_bot.ops.service_control import (
    ControlResult,
    queue_validated_restart,
    restart_deployer,
)

from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, JSONResponse

router = APIRouter()

ENV_PATH = Path(os.getenv("BOT_ENV_FILE", "/home/ubuntu/nifty_scalper_bot/.env"))
LOG_PATH = Path(os.getenv("BOT_LOG_FILE", str(Path(os.getenv("LOG_DIR", "logs")).expanduser() / "bot.log")))
SERVICE_NAME = os.getenv("BOT_SERVICE_NAME", "niftybot")
APP_DIR = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))

FIELDS: list[tuple[str, str, bool]] = [
    ("Zerodha API Key", "KITE_API_KEY", False),
    ("Zerodha API Secret", "KITE_API_SECRET", True),
    ("Zerodha Access Token", "KITE_ACCESS_TOKEN", True),
    ("Telegram Bot Token", "TELEGRAM_BOT_TOKEN", True),
    ("Telegram Chat ID", "TELEGRAM_CHAT_ID", False),
    ("Telegram Allowed ID", "TELEGRAM_ALLOWED_ID", False),
]

_ENV_CACHE: dict[str, object] = {"at": 0.0, "data": {}}


def _read_env() -> dict[str, str]:
    now = time.time()
    if now - float(_ENV_CACHE["at"]) < 5.0 and _ENV_CACHE["data"]:
        return dict(_ENV_CACHE["data"])
    data: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip().strip('"').strip("'")
    _ENV_CACHE.update({"at": now, "data": dict(data)})
    return data


def _write_env(updates: dict[str, str]) -> None:
    existing = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in existing:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            out.append(line)
            continue
        key = s.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text("\n".join(out) + "\n")
    os.chmod(ENV_PATH, 0o600)
    _ENV_CACHE.update({"at": 0.0, "data": {}})


def _check_auth(request: Request) -> None:
    # Host security-group/IP allowlist remains the access boundary for this legacy
    # console. Do not add broker login passwords or TOTP seeds to this application.
    return None


def _admin_password() -> str:
    return ""


# ---------------- Zerodha daily session ----------------

def _kite_login_url(api_key: str) -> str:
    return "https://kite.zerodha.com/connect/login?" + urllib.parse.urlencode({"v": "3", "api_key": api_key})


def _extract_request_token(value: str) -> str:
    """Accept a raw request_token or a complete Zerodha redirect URL."""
    value = (value or "").strip()
    if not value:
        return ""
    if "://" not in value and "request_token=" not in value:
        return value
    try:
        parsed = urllib.parse.urlparse(value if "://" in value else "http://local/?" + value.lstrip("?"))
        return (urllib.parse.parse_qs(parsed.query).get("request_token") or [""])[0].strip()
    except (ValueError, TypeError):
        return ""


def _exchange_request_token(api_key: str, api_secret: str, request_token: str) -> tuple[bool, str]:
    try:
        checksum = hashlib.sha256((api_key + request_token + api_secret).encode()).hexdigest()
        body = urllib.parse.urlencode({
            "api_key": api_key, "request_token": request_token, "checksum": checksum,
        }).encode()
        req = urllib.request.Request(
            "https://api.kite.trade/session/token", data=body,
            headers={"X-Kite-Version": "3", "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
        tok = (payload.get("data") or {}).get("access_token", "")
        if tok:
            return True, tok
        return False, "Kite did not return an access token"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _verify_access_token(api_key: str, access_token: str) -> tuple[bool, str]:
    """Verify a newly issued token before persisting/restarting the live engine."""
    try:
        req = urllib.request.Request(
            "https://api.kite.trade/user/profile",
            headers={
                "X-Kite-Version": "3",
                "Authorization": f"token {api_key}:{access_token}",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
        user_id = str((payload.get("data") or {}).get("user_id") or "").strip()
        return (True, user_id) if user_id else (False, "profile verification returned no user")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _activate_access_token(access_token: str) -> ControlResult:
    _write_env({
        "KITE_ACCESS_TOKEN": access_token,
        "ZERODHA_ACCESS_TOKEN": access_token,
        "BROKER_ACCESS_TOKEN": access_token,
    })
    return _restart_service()


def _complete_kite_session(request_token: str) -> RedirectResponse:
    env = _read_env()
    api_key = env.get("KITE_API_KEY", "").strip()
    api_secret = env.get("KITE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return RedirectResponse("/admin?token=err&msg=set+API+key+and+secret+first", status_code=303)
    ok, result = _exchange_request_token(api_key, api_secret, request_token)
    if not ok:
        return RedirectResponse(f"/admin?token=err&msg={urllib.parse.quote(result[:120])}", status_code=303)
    verified, detail = _verify_access_token(api_key, result)
    if not verified:
        return RedirectResponse(f"/admin?token=err&msg={urllib.parse.quote(('verification failed: ' + detail)[:120])}", status_code=303)
    restart_result = _activate_access_token(result)
    if restart_result.ok:
        return RedirectResponse("/admin?token=ok", status_code=303)
    return RedirectResponse(
        "/admin?token=saved_not_applied&msg="
        + urllib.parse.quote(restart_result.message[:120]),
        status_code=303,
    )


# ---------------- log/host helpers ----------------

_IST_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} IST)\]")


def _clean_log_line(line: str) -> str:
    m = _IST_RE.search(line)
    if not m:
        return ""
    return f"{m.group(1)}  {line[m.end():].strip()}"


def _tail_file(path: Path, lines: int, *, max_bytes: int = 2_000_000) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        read = min(size, max_bytes)
        fh.seek(size - read)
        chunk = fh.read(read)
    return "\n".join(chunk.decode("utf-8", errors="replace").splitlines()[-lines:])


def _gather_logs_window(lines: int, since: str, until: str, contains: str, clean: bool) -> str:
    text = ""
    try:
        cmd = ["journalctl", "-u", SERVICE_NAME, "-n", str(lines), "--no-pager", "-o", "cat"]
        if since:
            cmd += ["--since", since]
        if until:
            cmd += ["--until", until]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            text = out.stdout
    except Exception:
        text = ""
    if not text and LOG_PATH.exists():
        try:
            text = _tail_file(LOG_PATH, lines)
        except Exception as exc:  # noqa: BLE001
            return f"log read error: {exc}"
    rows = text.splitlines()
    if clean:
        rows = [c for c in (_clean_log_line(r) for r in rows) if c]
    if contains:
        needle = contains.lower()
        rows = [r for r in rows if needle in r.lower()]
    return "\n".join(rows) if rows else "No matching log lines."


def _gather_logs(lines: int, since: str = "", until: str = "", contains: str = "", clean: bool = True) -> str:
    return _gather_logs_window(max(50, min(int(lines or 400), 20000)), since, until, contains, clean)


def _restart_service() -> ControlResult:
    """Queue the validated release, never restart a live engine from HTTP."""
    return queue_validated_restart()


def _git_update() -> tuple[bool, str]:
    """Recover or trigger the canonical validated Lightsail deployer."""
    result = restart_deployer()
    return result.ok, result.message


# ---------------- UI ----------------

from nifty_scalper_bot.superlite_admin_style import STYLE as _CSS, SCRIPT as _SCRIPT

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><meta name="theme-color" content="#0b1522"><title>Nifty Scalper · Admin</title><style>{css}</style></head><body>{body}<script>{script}</script></body></html>"""


def _page(body: str) -> str:
    return _PAGE.format(css=_CSS, script=_SCRIPT, body=body)


def _topbar(live_on: bool) -> str:
    pill = ('<span class="pill on">Configured LIVE</span>' if live_on
            else '<span class="pill off">Configured SHADOW</span>')
    return (
        '<header class="top"><a href="/admin" class="brand"><span class="logo">N</span>'
        '<span>Nifty Scalper</span></a>' + pill
        + '<span class="sp"></span><nav class="nav" aria-label="Admin sections">'
        '<a href="#overview">Overview</a><a href="#session">Daily login</a>'
        '<a href="#logs">Logs</a><a href="#controls">Controls</a></nav></header>'
    )


def _flash(request: Request) -> str:
    q = request.query_params
    if q.get("saved"): return '<div class="flash ok">Settings saved. Restart to apply.</div>'
    if q.get("token") == "ok":
        return (
            '<div class="flash ok">Zerodha token verified and saved; '
            "validated engine restart queued, not yet applied.</div>"
        )
    if q.get("token") == "saved_not_applied":
        return (
            '<div class="flash err">Zerodha token saved, but restart not queued: '
            f'{html.escape(q.get("msg", ""))}. '
            "Engine may still use the prior token.</div>"
        )
    if q.get("token") == "err": return f'<div class="flash err">Zerodha session failed: {html.escape(q.get("msg", ""))}</div>'
    if q.get("mode") == "err": return '<div class="flash err">Cannot go LIVE: enter API key, secret and access token first.</div>'
    if q.get("mode") == "queued":
        return (
            '<div class="flash ok">Mode setting saved; '
            "validated restart queued, not yet applied.</div>"
        )
    if q.get("mode") == "saved_not_applied":
        return (
            '<div class="flash err">Mode setting saved, but restart not queued: '
            f'{html.escape(q.get("msg", ""))}. '
            "Running mode has not been verified.</div>"
        )
    if q.get("restart") == "queued":
        return (
            '<div class="flash ok">Validated bot restart queued. '
            "Restart requires broker-flat and disarmed-engine confirmation; "
            "queued does not mean completed.</div>"
        )
    if q.get("restart") == "err":
        return (
            '<div class="flash err">Restart request failed: '
            f'{html.escape(q.get("msg", ""))}</div>'
        )
    if q.get("riskreset") == "ok":
        return (
            '<div class="flash ok">Daily risk allowance rebased. '
            "Trade ledger and realised P&amp;L were preserved.</div>"
        )
    if q.get("riskreset") == "err":
        msg = html.escape(q.get("msg", ""))
        return f'<div class="flash err">Risk reset failed: {msg}</div>'
    if q.get("upd") in {"queued", "ok"}:
        return (
            '<div class="flash ok">GitHub update check requested. '
            "This is NOT deployment completion; check release status for "
            "deferred, failed or deployed results.</div>"
        )
    if q.get("upd") == "err": return f'<div class="flash err">Update failed: {html.escape(q.get("msg", ""))}</div>'
    return ""


@router.get("/admin", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """Serve the UI immediately; live metrics are fetched independently on demand."""
    _check_auth(request)
    env = _read_env()
    live_on = (
        env.get("ENABLE_LIVE", "false").strip().lower() in {"1", "true", "yes", "on"}
        and env.get("EXECUTION_MODE", "SHADOW").strip().upper() == "LIVE"
    )
    api_key = env.get("KITE_API_KEY", "").strip()
    login_url = _kite_login_url(api_key) if api_key else ""
    login_button = (
        f'<a class="btn primary full" href="{html.escape(login_url, quote=True)}" rel="noopener">'
        'Open official Zerodha login ↗</a>'
        if login_url
        else '<p class="warning">First save your Zerodha API key and secret under Advanced settings.</p>'
    )
    toggle = (
        '<form method="post" action="/admin/mode">'
        f'<input type="hidden" name="mode" value="{"shadow" if live_on else "live"}">'
        + (
            '<button type="submit" class="amb full">'
            'Switch to SHADOW (disable live orders)</button>'
            if live_on
            else '<button type="submit" class="danger full" '
                 'onclick="return confirm(\'Enable LIVE trading with real capital?\')">'
                 'Enable LIVE trading</button>'
        )
        + '</form>'
    )
    rows = ""
    for label, key, sec in FIELDS:
        value = env.get(key, "")
        shown = "••••••••" if (sec and value) else value
        safe_key = html.escape(key, quote=True)
        rows += (
            f'<label for="cfg-{safe_key}">{html.escape(label)} '
            f'<span class="badge">{safe_key}</span></label>'
            f'<input id="cfg-{safe_key}" name="{safe_key}" '
            f'type="{"password" if sec else "text"}" '
            f'value="{html.escape(shown, quote=True)}" '
            'autocomplete="off" placeholder="Leave blank to keep existing value">'
        )
    body = f"""{_topbar(live_on)}<main class="wrap">
    <div class="intro"><div><div class="eyebrow">Operations center</div>
    <h1>Bot administration</h1>
    <p>Daily access, deployment status and safe controls in one place.</p></div>
    <div class="statusline"><span class="dot" aria-hidden="true"></span>
    <span id="status-note" role="status" aria-live="polite">Checking live status…</span>
    <button id="refresh-status" type="button" class="gray">Refresh</button></div></div>
    {_flash(request)}
    <section id="overview" aria-label="Live status" class="grid">
      <div class="metric"><div class="k">Engine</div><div class="v" id="engine-value">Checking…</div>
      <small>Actual HTTP health</small></div>
      <div class="metric"><div class="k">Zerodha</div><div class="v" id="broker-value">Checking…</div>
      <small>Broker authentication</small></div>
      <div class="metric"><div class="k">Running mode</div><div class="v" id="mode-value">Checking…</div>
      <small>Engine state, not requested mode</small></div>
      <div class="metric"><div class="k">Blocker</div><div class="v" id="blocker-value">Checking…</div>
      <small>Trading readiness</small></div>
    </section>
    <div class="cta-strip"><div class="statusline">
      <span>Release: <b id="release-value">Checking…</b></span>
      <span>Engine SHA: <b id="revision-value">—</b></span>
    </div><a href="/admin/api/status" class="muted">Detailed diagnostics ↗</a></div>
    <div class="layout">
      <div class="stack">
        <section class="card session" id="session">
          <div class="eyebrow">Every trading day</div>
          <h2>Zerodha access token</h2>
          <p>Use the official login and TOTP. Your login password and TOTP are never entered into this dashboard.</p>
          {login_button}
          <ol class="steps">
            <li>Authenticate at Zerodha, then return using the configured callback.</li>
            <li>The token is verified and saved. A validated restart is queued.</li>
            <li>Check <b>Zerodha</b> and the running engine above; queued is not completed.</li>
          </ol>
          <details><summary>Paste callback URL / request token instead</summary><div class="detail-body">
            <form method="post" action="/admin/token">
            <label for="request-token">Zerodha redirect URL or request token</label>
            <input id="request-token" name="request_token" autocomplete="off" spellcheck="false"
              placeholder="Paste the callback URL or request token" required>
            <button type="submit" class="full green">Verify token and queue restart</button></form>
          </div></details>
          <div class="hint">Automatic callback needs the Kite app redirect URL set to this admin host’s
          <code>/admin/kite/callback</code> path.</div>
        </section>
        <section class="card" id="logs">
          <h2>Live service logs</h2>
          <p>Read-only, bounded journal snapshot. Refresh on demand; no continuous heavy polling.</p>
          <div class="split"><div><label for="log-filter">Filter</label>
          <select id="log-filter"><option value="">Recent activity</option><option value="ERROR">Errors</option>
          <option value="RUNNER">Runner</option><option value="ORDER">Orders</option>
          <option value="WARN">Warnings</option><option value="custom">Custom search</option></select>
          <input id="log-query" type="text" placeholder="Search log text" hidden
          aria-label="Custom log filter"></div>
          <div><label for="log-lines">Journal window</label>
          <select id="log-lines"><option value="100">Last 100 lines</option>
          <option value="250" selected>Last 250 lines</option>
          <option value="500">Last 500 lines</option></select></div></div>
          <div class="actions" style="margin-top:12px">
          <button id="load-logs" type="button">Load logs</button>
          <a class="btn gray" href="/admin/logs/download">Download Logs</a></div>
          <pre id="log-output" class="console" role="log" aria-live="off">Select a filter and tap Load logs.</pre>
        </section>
      </div>
      <div class="stack">
        <section class="card" id="controls">
          <h2>Deployment &amp; service</h2>
          <p>Automatic release checks run independently. Manual actions only <b>queue</b>
          the validated operation, subject to broker-flat and disarmed-engine checks.</p>
          <div class="flow">
            <form method="post" action="/admin/update"><button type="submit" class="full">
              Check GitHub for updates</button></form>
            <form method="post" action="/admin/restart"><button type="submit" class="full amb"
              onclick="return confirm('Queue a validated safe restart? This will not bypass trade exposure checks.')">
              Queue safe restart</button></form>
          </div>
          <div class="hint">If a request is deferred, review release status.
          Requests are not confirmation that code deployed or the engine restarted.</div>
        </section>
        <section class="card"><h2>Advanced settings</h2>
          <p>Occasional changes. Hidden by default to keep daily operation simple.</p>
          <details><summary>Trading mode &amp; risk controls</summary><div class="detail-body">
            <div class="subheading">Configured mode</div>
            <p>{'LIVE requested — real orders possible when engine permits.' if live_on
                else 'SHADOW requested — simulation only.'}</p>{toggle}
            <div class="subheading">Daily risk allowance</div>
            <p>Rebase the allowance only. This does not erase trades, realised P&amp;L or history.</p>
            <form method="post" action="/admin/risk/reset-day">
              <input type="hidden" name="confirm" value="RESET">
              <button type="submit" class="amb full"
              onclick="return confirm('Reset the daily risk allowance? Ledger and realised P&L remain unchanged.')">
              Reset today’s risk allowance</button></form>
          </div></details>
          <details><summary>Credentials &amp; integration settings</summary><div class="detail-body">
            <p>Existing secrets remain unchanged when their placeholders are left as-is.
            Never share an access token in screenshots.</p>
            <form method="post" action="/admin/save">{rows}
              <button type="submit" class="full">Save settings</button></form>
          </div></details>
          <details><summary>Research &amp; backtesting</summary><div class="detail-body">
            <p>On-demand replay and component analysis, separate from live order execution.</p>
            <div class="flow">
              <form method="post" action="/admin/research/start">
              <button type="submit" class="full gray">Run research backtest</button></form>
              <form method="post" action="/admin/research/start?mode=runtime">
              <button type="submit" class="full gray">Replay recorded bot sessions</button></form>
            </div><p><a href="/admin/research/status">Research job</a> ·
            <a href="/admin/research/report">Latest research report</a></p>
            <p>Missing sessions, slippage assumptions and coverage limitations are shown in the report.</p>
          </div></details>
        </section>
        <div class="sidehelp">If this page loads but metrics do not, the admin process may be healthy while
        the engine API is unreachable. Refresh and consult detailed diagnostics before taking action.</div>
      </div>
    </div>
    <div class="footer">Nifty Scalper · Independent admin control plane · All times reflect your device’s clock.</div>
    </main>"""
    return HTMLResponse(_page(body), headers={"Cache-Control": "no-store"})


@router.get("/admin/kite/callback")
def kite_callback(request: Request, request_token: str = "", status: str = "", action: str = "") -> RedirectResponse:
    """Registered Kite redirect target; no request/access token is rendered back to the browser."""
    _check_auth(request)
    req_tok = _extract_request_token(request_token)
    if status and status.lower() not in {"success", "ok"}:
        return RedirectResponse("/admin?token=err&msg=Zerodha+login+was+not+successful", status_code=303)
    if not req_tok:
        return RedirectResponse("/admin?token=err&msg=no+request+token+received", status_code=303)
    return _complete_kite_session(req_tok)


@router.post("/admin/token")
def update_token(request: Request, request_token: str = Form(""), access_token: str = Form("")) -> RedirectResponse:
    _check_auth(request)
    req_tok = _extract_request_token(request_token)
    acc_tok = access_token.strip()
    if req_tok:
        return _complete_kite_session(req_tok)
    if not acc_tok:
        return RedirectResponse("/admin?token=err&msg=no+token+provided", status_code=303)
    env = _read_env()
    api_key = env.get("KITE_API_KEY", "").strip()
    if not api_key:
        return RedirectResponse("/admin?token=err&msg=set+API+key+first", status_code=303)
    verified, detail = _verify_access_token(api_key, acc_tok)
    if not verified:
        return RedirectResponse(f"/admin?token=err&msg={urllib.parse.quote(('verification failed: ' + detail)[:120])}", status_code=303)
    restart_result = _activate_access_token(acc_tok)
    if restart_result.ok:
        return RedirectResponse("/admin?token=ok", status_code=303)
    return RedirectResponse(
        "/admin?token=saved_not_applied&msg="
        + urllib.parse.quote(restart_result.message[:120]),
        status_code=303,
    )


@router.post("/admin/mode")
def set_mode(request: Request, mode: str = Form(...)) -> RedirectResponse:
    _check_auth(request)
    if mode.strip().lower() == "live":
        env = _read_env()
        has_key = any(env.get(k, "").strip() for k in ("KITE_API_KEY", "ZERODHA_API_KEY", "BROKER_API_KEY"))
        has_secret = any(env.get(k, "").strip() for k in ("KITE_API_SECRET", "ZERODHA_API_SECRET", "BROKER_API_SECRET"))
        has_token = any(env.get(k, "").strip() for k in ("KITE_ACCESS_TOKEN", "ZERODHA_ACCESS_TOKEN", "BROKER_ACCESS_TOKEN"))
        if not (has_key and has_secret and has_token):
            return RedirectResponse("/admin?mode=err", status_code=303)
        _write_env({"ENABLE_LIVE": "true", "EXECUTION_MODE": "LIVE"})
    else:
        _write_env({"ENABLE_LIVE": "false", "EXECUTION_MODE": "SHADOW"})
    restart_result = _restart_service()
    if restart_result.ok:
        return RedirectResponse("/admin?mode=queued", status_code=303)
    return RedirectResponse(
        "/admin?mode=saved_not_applied&msg="
        + urllib.parse.quote(restart_result.message[:120]),
        status_code=303,
    )


@router.post("/admin/save")
async def save(request: Request) -> RedirectResponse:
    _check_auth(request)
    form = await request.form()
    updates: dict[str, str] = {}
    for _, key, sec in FIELDS:
        if key not in form:
            continue
        val = str(form[key]).strip()
        if val == "" or (sec and set(val) <= {"•"}):
            continue
        updates[key] = val
    if "KITE_API_KEY" in updates:
        updates["ZERODHA_API_KEY"] = updates["BROKER_API_KEY"] = updates["KITE_API_KEY"]
    if "KITE_API_SECRET" in updates:
        updates["ZERODHA_API_SECRET"] = updates["BROKER_API_SECRET"] = updates["KITE_API_SECRET"]
    if updates:
        _write_env(updates)
    return RedirectResponse("/admin?saved=1", status_code=303)


@router.post("/admin/risk/reset-day")
def reset_daily_risk(request: Request, confirm: str = Form("")) -> RedirectResponse:
    """Explicitly rebase the live daily-loss allowance without altering accounting."""
    _check_auth(request)
    if confirm != "RESET":
        return RedirectResponse(
            "/admin?riskreset=err&msg=confirmation+required", status_code=303
        )
    try:
        from nifty_scalper_bot.main import _latest_context

        ctx = _latest_context()
        risk = getattr(ctx, "risk_manager", None)
        if risk is None:
            order_manager = getattr(ctx, "order_manager", None)
            risk = getattr(order_manager, "_risk_manager", None)
        resetter = getattr(risk, "operator_reset_daily_loss_budget", None)
        if not callable(resetter):
            raise RuntimeError("live risk manager unavailable")
        resetter()
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse(
            f"/admin?riskreset=err&msg={urllib.parse.quote(str(exc)[:120])}",
            status_code=303,
        )
    return RedirectResponse("/admin?riskreset=ok", status_code=303)


@router.post("/admin/restart")
def restart(request: Request) -> Response:
    _check_auth(request)
    result = queue_validated_restart()
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(
            {
                "ok": result.ok,
                "state": "queued" if result.ok else "failed",
                "message": result.message,
            },
            status_code=202 if result.ok else 503,
        )
    if result.ok:
        return RedirectResponse("/admin?restart=queued", status_code=303)
    return RedirectResponse(
        "/admin?restart=err&msg=" + urllib.parse.quote(result.message[:120]),
        status_code=303,
    )


@router.post("/admin/update")
def update_from_github(request: Request) -> Response:
    _check_auth(request)
    ok, message = _git_update()
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(
            {
                "ok": ok,
                "state": "queued" if ok else "failed",
                "message": (
                    "Deployment check queued; completion is not yet verified"
                    if ok
                    else message
                ),
            },
            status_code=202 if ok else 503,
        )
    if ok:
        return RedirectResponse("/admin?upd=queued", status_code=303)
    return RedirectResponse(
        "/admin?upd=err&msg=" + urllib.parse.quote(message[:120]),
        status_code=303,
    )


@router.get("/admin/api/overview")
def overview_api(request: Request) -> JSONResponse:
    """Lightweight independent heartbeat for mobile auto-refresh."""
    _check_auth(request)
    return JSONResponse(quick_overview(), headers={"Cache-Control": "no-store"})


@router.get("/admin/api/logs")
def logs_api(request: Request, lines: int = 250, contains: str = "") -> JSONResponse:
    """Bounded, redacted, explicitly requested journal sample."""
    _check_auth(request)
    limit = max(50, min(lines, 500))
    needle = contains[:80]
    text = bounded_logs(limit, needle)
    # Do not send megabytes of journal text to a constrained mobile browser.
    return JSONResponse(
        {"text": text[-100_000:] if text else "No matching log lines."},
        headers={"Cache-Control": "no-store"},
    )


_STATUS_CACHE: dict[str, object] = {"at": 0.0, "label": "running", "color": "#3fb950"}


@router.get("/admin/status.json")
def status_json(request: Request) -> JSONResponse:
    _check_auth(request)
    now = time.time()
    ttl = float(os.getenv("ADMIN_STATUS_CACHE_SECONDS", "10") or "10")
    if now - float(_STATUS_CACHE["at"]) < ttl:
        return JSONResponse({"label": _STATUS_CACHE["label"], "color": _STATUS_CACHE["color"]})
    label, color = "running", "#3fb950"
    try:
        tail = _gather_logs(60, clean=True)
        if "degraded mode" in tail or "Missing required env" in tail:
            label, color = "degraded", "#d29922"
        elif "fully operational" in tail:
            label, color = "operational", "#3fb950"
    except Exception:
        label, color = "unknown", "#8b97a6"
    _STATUS_CACHE.update({"at": now, "label": label, "color": color})
    return JSONResponse({"label": label, "color": color})


_TRADE_MARKERS = (
    "ORDER_SENT", "Sending Order", "FILLED", "average_price", "EXIT",
    "TRADE_ATTEMPT", "ORDER_REJECTED", "ORDER_BROKER_CONFIG_ERROR", "pnl",
    "TRADE_CLOSED", "ORDER_COMPLETE", "SIGNAL_GENERATED",
)


@router.get("/admin/trades.json")
def trades_json(request: Request, lines: int = 4000) -> JSONResponse:
    _check_auth(request)
    text = _gather_logs(lines, clean=True)
    hits = [ln for ln in text.splitlines() if any(m in ln for m in _TRADE_MARKERS)]
    return JSONResponse({"text": "\n".join(hits) or "No trade events in the recent log window."})


@router.get("/admin/logs/download")
def logs_download(request: Request, fmt: str = "txt", lines: int = 2000, contains: str = "") -> PlainTextResponse:
    _check_auth(request)
    text = _gather_logs(lines, contains=contains, clean=True)
    rows = text.splitlines()
    ts = time.strftime("%Y%m%d-%H%M%S")
    if fmt == "json":
        data, media, ext = json.dumps([{"line": i + 1, "text": r} for i, r in enumerate(rows)], indent=2), "application/json", "json"
    elif fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["line", "text"])
        for i, r in enumerate(rows):
            writer.writerow([i + 1, r])
        data, media, ext = buf.getvalue(), "text/csv", "csv"
    else:
        data, media, ext = text, "text/plain", "txt"
    return PlainTextResponse(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="niftybot-logs-{ts}.{ext}"'})


@router.post("/admin/research/start")
def start_research(request: Request, mode: str = "all", days: int = 30) -> JSONResponse:
    """Start the fixed research job; never accept shell commands from the UI."""
    from nifty_scalper_bot.ops.research_jobs import new_request, start_job
    from nifty_scalper_bot.superlite_admin_core import same_origin

    _check_auth(request)
    same_origin(request)
    try:
        result = start_job(APP_DIR, new_request(days=days, mode=mode))
    except ValueError:
        return JSONResponse(
            {"error": "Invalid research mode or date range"}, status_code=422
        )
    return JSONResponse(result, status_code=409 if result["state"] == "busy" else 202)


@router.post("/admin/research/poll-manifest")
def poll_research_manifest(request: Request) -> JSONResponse:
    """Launch the fixed manifest request from the persistent admin service."""

    from nifty_scalper_bot.ops.research_jobs import start_job
    from nifty_scalper_bot.superlite_admin_core import same_origin

    _check_auth(request)
    same_origin(request)
    manifest = APP_DIR / "deploy/research_request.json"
    if not manifest.is_file():
        return JSONResponse(
            {"state": "not_requested", "backtest_completed": False},
            status_code=200,
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        result = start_job(APP_DIR, payload)
    except (OSError, ValueError, json.JSONDecodeError):
        return JSONResponse(
            {"error": "Invalid research request manifest"},
            status_code=422,
        )
    return JSONResponse(result, status_code=409 if result["state"] == "busy" else 202)


@router.get("/admin/research/status")
def research_status(request: Request) -> JSONResponse:
    from nifty_scalper_bot.ops.research_jobs import read_status

    _check_auth(request)
    return JSONResponse(read_status(APP_DIR), headers={"Cache-Control": "no-store"})


@router.get("/admin/research/report")
def research_report(request: Request) -> JSONResponse:
    from nifty_scalper_bot.ops.research_jobs import read_status, validate_request

    _check_auth(request)
    status = read_status(APP_DIR)
    report = {"status": status}
    if status.get("id"):
        validate_request({"id": status["id"]})
        for name in (
            "completed_trade_analysis",
            "strategy_bar_research",
            "orb_comparison",
            "vwap_comparison",
            "runtime_replay",
        ):
            path = APP_DIR / "data/research" / status["id"] / f"{name}.json"
            if path.is_file():
                report[name] = json.loads(path.read_text())
    return JSONResponse(report, headers={"Cache-Control": "no-store"})
