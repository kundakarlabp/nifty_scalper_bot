"""Purpose: Expose daily broker authentication and operational admin controls.

Responsibilities:
- dashboard redirects legacy HTTP /admin to the secure original dashboard.
- Render existing daily request-token form immediately without separate login page.
- Verify owner through the same request-token form before enabling any writes.
- Signed Zerodha owner session replaces the redundant admin Basic password.
- Serve sanitized runtime overview, protected logs and research-status endpoints.
- Route GitHub-update/restart requests through service-control owners.

Boundaries: The release runner and engine retain execution authority.
Safety: Public status never exposes secrets, logs, or mutable broker controls.
Sensitive writes and the full UI keep HTTPS/authentication and broker release gates.
Do not collect Kite passwords/TOTP or bypass broker release safeguards.
"""

# Embedded HTML/CSS/JS assets have intentional long literal lines.
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)

from nifty_scalper_bot.admin_owner_auth import (
    LOGIN_COOKIE,
    clear_challenge,
    create_challenge,
    grant_owner_session,
    owner_user_id,
    require_challenge_bound_form,
    require_https,
    valid_owner_session,
    verify_challenge,
)
from nifty_scalper_bot.ops.service_control import (
    ControlResult,
    queue_validated_restart,
    restart_deployer,
)
from nifty_scalper_bot.superlite_admin_core import (
    ENV_PATH as CANONICAL_ENV_PATH,
    bounded_logs,
    quick_overview,
    write_env as write_canonical_env,
)
from nifty_scalper_bot.superlite_admin_style import SCRIPT as _SCRIPT, STYLE as _CSS

router = APIRouter()

ENV_PATH = CANONICAL_ENV_PATH
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
    """Update the same protected environment that the runtime reads."""
    write_canonical_env(updates)
    _ENV_CACHE.update({"at": 0.0, "data": {}})

def _check_auth(request: Request) -> None:
    """Require the verified Zerodha-owner session, never a second admin password.

    The public engine also mounts these routes, so HTTP access to mutable
    controls remains denied. Host-local diagnostics stay available.
    """
    if os.getenv("DEPLOYMENT_PLATFORM") != "aws_lightsail":
        return
    peer = request.client.host if request.client else ""
    if peer in {"127.0.0.1", "::1"} and not request.headers.get("x-forwarded-for"):
        return
    if request.url.scheme != "https":
        raise HTTPException(status_code=403, detail="admin_https_required")
    owner = owner_user_id(_read_env())
    if not owner:
        raise HTTPException(status_code=503, detail="admin_owner_not_configured")
    if not valid_owner_session(request, owner):
        # No WWW-Authenticate: browsers must never show an admin Basic popup.
        raise HTTPException(status_code=401, detail="owner_login_required")
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        source = request.headers.get("origin") or request.headers.get("referer", "")
        parsed = urllib.parse.urlsplit(source)
        if parsed.scheme != "https" or parsed.netloc != request.headers.get("host"):
            raise HTTPException(status_code=403, detail="admin_same_origin_required")


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
    """Persist verified token, then queue restart with a bounded rotation proof."""
    _write_env(
        {
            "KITE_ACCESS_TOKEN": access_token,
            "ZERODHA_ACCESS_TOKEN": access_token,
            "BROKER_ACCESS_TOKEN": access_token,
        }
    )
    # Ordinary restart markers never authorize an old-authentication bypass.
    # The broker's /user/profile was verified immediately before this call.
    import tempfile

    proof_dir = Path(os.getenv("NIFTYBOT_CONFIG_DIR", "/home/ubuntu/.config/niftybot"))
    proof_file = proof_dir / "auto_release_verified_token_rotation"
    if not proof_dir.is_dir() or proof_dir.is_symlink() or proof_file.is_symlink():
        return ControlResult(False, "restart_bot", "token proof location unavailable")
    try:
        fd, temporary = tempfile.mkstemp(prefix=".verified-token.", dir=proof_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(
                    {
                        "token_sha256": hashlib.sha256(
                            access_token.encode("utf-8")
                        ).hexdigest(),
                        "verified_at": time.time(),
                    },
                    target,
                )
                target.flush()
                os.fsync(target.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, proof_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError:
        return ControlResult(False, "restart_bot", "token proof could not be stored")
    return _restart_service()


def _complete_kite_session(request_token: str) -> RedirectResponse:
    env = _read_env()
    api_key = next(
        (
            env.get(k, "").strip()
            for k in ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
            if env.get(k, "").strip()
        ),
        "",
    )
    api_secret = next(
        (
            env.get(k, "").strip()
            for k in ("BROKER_API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET")
            if env.get(k, "").strip()
        ),
        "",
    )
    if not api_key or not api_secret:
        return RedirectResponse("/admin?token=err&msg=set+API+key+and+secret+first", status_code=303)
    ok, result = _exchange_request_token(api_key, api_secret, request_token)
    if not ok:
        return RedirectResponse(f"/admin?token=err&msg={urllib.parse.quote(result[:120])}", status_code=303)
    verified, detail = _verify_access_token(api_key, result)
    if not verified:
        return RedirectResponse(f"/admin?token=err&msg={urllib.parse.quote(('verification failed: ' + detail)[:120])}", status_code=303)
    if os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail":
        expected_owner = owner_user_id(env)
        if not expected_owner or not hmac.compare_digest(detail, expected_owner):
            return RedirectResponse(
                "/admin?token=err&msg=Zerodha+account+mismatch", status_code=303
            )
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

# fmt: off  # The following block owns literal HTML markup.
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
    if q.get("saved"):
        return '<div class="flash ok">Settings saved. Engine changes require a validated restart.</div>'
    if q.get("power") == "queued":
        return ('<div class="flash ok">Operating mode saved; safe transition '
                'queued. Watch the Engine status for confirmation.</div>')
    if q.get("power") == "saved_not_applied":
        return ('<div class="flash err">Operating mode saved, but the validated '
                'transition could not be queued. Engine state is unchanged.</div>')
    if q.get("power") == "unchanged":
        return '<div class="flash ok">Operating mode is already selected.</div>'
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
            "Restart waits for verified flat broker positions and terminal orders; "
            "protective position management must not be interrupted. "
            "Queued does not mean completed.</div>"
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
def dashboard(request: Request) -> Response:
    """Use the familiar /admin URL for complete owner controls over HTTPS."""
    if os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail":
        if request.url.scheme != "https":
            # The old :8080 bookmark opens the secure dashboard, not a dead end.
            return RedirectResponse(
                "https://65.0.255.84/admin", status_code=307,
                headers={"Cache-Control": "no-store"},
            )
        return admin_manage(request)
    return _full_dashboard(request)


@router.get("/admin/manage", response_class=HTMLResponse)  # type: ignore[untyped-decorator]
def admin_manage(request: Request) -> Response:
    """Render the original dashboard immediately; verified token unlocks controls."""
    if os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail":
        if request.url.scheme != "https":
            raise HTTPException(status_code=403, detail="admin_https_required")
        return _full_dashboard(request, preview=True)
    return _full_dashboard(request)


def _full_dashboard(request: Request, *, preview: bool = False) -> HTMLResponse:
    owner_verified = False
    if preview and os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail":
        owner = owner_user_id(_read_env())
        if not owner:
            raise HTTPException(status_code=503, detail="admin_owner_not_configured")
        owner_verified = valid_owner_session(request, owner)
        if owner_verified:
            _check_auth(request)
    else:
        _check_auth(request)
        owner_verified = True
    private_env = _read_env()
    # The page may render before daily login, but never populate credential
    # values or expose private integration IDs to an unauthenticated browser.
    env = private_env if owner_verified else {
        "KITE_API_KEY": next(
            (private_env.get(k, "").strip()
             for k in ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
             if private_env.get(k, "").strip()), ""
        ),
        "ENABLE_LIVE": private_env.get("ENABLE_LIVE", "false"),
        "EXECUTION_MODE": private_env.get("EXECUTION_MODE", "SHADOW"),
    }
    live_on = (
        env.get("ENABLE_LIVE", "false").strip().lower() in {"1", "true", "yes", "on"}
        and env.get("EXECUTION_MODE", "SHADOW").strip().upper() == "LIVE"
    )
    api_key = next(
        (
            env.get(key, "").strip()
            for key in ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
            if env.get(key, "").strip()
        ),
        "",
    )
    # Import lazily: operating_control uses this module for canonical writes.
    from nifty_scalper_bot.operating_control import admin_power_card

    power_card = admin_power_card(editable=owner_verified)
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
    {power_card}
    <div class="cta-strip"><div class="statusline">
      <span>Release: <b id="release-value">Checking…</b></span>
      <span>Engine SHA: <b id="revision-value">—</b></span>
    </div><div class="actions"><a href="/admin/status.json">Service status</a><a href="/admin/api/status" class="muted">Detailed diagnostics ↗</a></div></div>
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
          validated operations. Restarting may interrupt live order management.</p>
          <div class="flow">
            <form method="post" action="/admin/update"><button type="submit" class="full">
              Check GitHub for updates</button></form>
            <form method="post" action="/admin/restart"><button type="submit" class="full amb"
              onclick="return confirm('Restart the LIVE engine as soon as validated? Open-position and stop management may be interrupted.')">
              Queue engine restart</button></form>
          </div>
          <div class="hint">Review release status for CI, validation or service failures.
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
            <div class="subheading">Daily Zerodha request token</div>
            <p>Paste the full Zerodha callback URL or one-time request token.
            Verification exchanges it for an access token, checks your broker
            account and queues the safe engine refresh. The request token is
            never stored in settings.</p>
            <form method="post" action="/admin/token">
              <label for="settings-request-token">Request token / callback URL</label>
              <input id="settings-request-token" name="request_token"
                type="text" autocomplete="off" spellcheck="false"
                placeholder="Paste fresh Zerodha request token or URL" required>
              <button type="submit" class="full green">Verify and save Zerodha token</button>
            </form>
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
    # Always expose the existing daily token paste form on mobile.
    body = body.replace(
        "<details><summary>Paste callback URL / request token instead</summary>",
        "<details open><summary>Paste callback URL / request token</summary>",
        1,
    )
    if preview and not owner_verified:
        # Other controls remain unavailable until Kite owner verification.
        body = re.sub(
            r'(<form method="post" action="/admin/(?!token\b)[^"]+")',
            r"\1 inert", body,
        )
        body = body.replace(
            '<div class="intro">',
            '<p id="admin-access-note" class="flash">Dashboard is open. '
            'Loading the real engine and broker status below; a browser '
            'session is not the same as Zerodha broker authentication.'
            '</p><div class="intro">', 1,
        )
        response = HTMLResponse(
            "", headers={"Cache-Control": "no-store", "Referrer-Policy": "same-origin"}
        )
        challenge = create_challenge(response)
        body = body.replace(
            '<form method="post" action="/admin/token">',
            '<form method="post" action="/admin/token">'
            f'<input type="hidden" name="owner_state" value="{html.escape(challenge, quote=True)}">',
        )
        response.body = _page(body).encode("utf-8")
        response.headers["content-length"] = str(len(response.body))
        return response
    return HTMLResponse(_page(body), headers={"Cache-Control": "no-store"})


# fmt: on



# fmt: off  # This block owns intentional long embedded HTML literals.
# A single official Zerodha login can both rotate the daily broker token and
# establish a short-lived, HttpOnly owner session. No public unauthenticated
# token, LIVE-mode, or restart endpoint is created.
@router.get("/admin/kite/start")  # type: ignore[untyped-decorator]
def owner_login_start(request: Request) -> RedirectResponse:
    """Legacy bookmark opens the original dashboard, not a second login form."""
    require_https(request)
    return RedirectResponse("/admin", status_code=303)


def _finish_owner_login(request: Request, owner_state: str, request_token: str) -> RedirectResponse:
    require_https(request)
    if not verify_challenge(request, owner_state):
        raise HTTPException(status_code=403, detail="owner_login_challenge_invalid")
    env = _read_env()
    owner = owner_user_id(env)
    if not owner:
        raise HTTPException(status_code=503, detail="owner_account_not_configured")
    key = next(
        (env[k].strip() for k in ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
         if env.get(k, "").strip()), ""
    )
    secret = next(
        (env[k].strip() for k in ("BROKER_API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET")
         if env.get(k, "").strip()), ""
    )
    if not key or not secret:
        raise HTTPException(status_code=503, detail="kite_credentials_missing")
    token = _extract_request_token(request_token)
    if not token:
        raise HTTPException(status_code=422, detail="request_token_missing")
    ok, access_token = _exchange_request_token(key, secret, token)
    if not ok:
        # Never return raw exchange errors to an unauthenticated browser.
        raise HTTPException(status_code=401, detail="kite_token_exchange_failed")
    verified, user_id = _verify_access_token(key, access_token)
    if not verified or not hmac.compare_digest(user_id, owner):
        raise HTTPException(status_code=403, detail="zerodha_owner_mismatch")
    restart = _activate_access_token(access_token)
    location = (
        "/admin?token=ok"
        if restart.ok
        else "/admin?token=saved_not_applied"
    )
    response = RedirectResponse(location, status_code=303)
    clear_challenge(response)
    grant_owner_session(response, owner)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/admin/kite/complete")  # type: ignore[untyped-decorator]
def owner_login_complete(
    request: Request,
    owner_state: str = Form(...),
    request_token: str = Form(...),
) -> RedirectResponse:
    # Mobile browsers may suppress Referer; the signed cookie-bound challenge
    # is mandatory, and any explicit foreign Origin/Referer is still rejected.
    require_challenge_bound_form(request, owner_state)
    return _finish_owner_login(request, owner_state, request_token)


@router.get("/admin/kite/callback")
def kite_callback(request: Request, request_token: str = "", status: str = "", action: str = "", owner_state: str = "") -> RedirectResponse:
    """Registered Kite redirect; owner-state permits passwordless verification."""
    if os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail":
        owner = owner_user_id(_read_env())
        if owner and not valid_owner_session(request, owner):
            if status and status.lower() not in {"success", "ok"}:
                return RedirectResponse("/admin?token=err", status_code=303)
            challenge = owner_state or request.cookies.get(LOGIN_COOKIE, "")
            return _finish_owner_login(request, challenge, request_token)
    _check_auth(request)
    req_tok = _extract_request_token(request_token)
    if status and status.lower() not in {"success", "ok"}:
        return RedirectResponse("/admin?token=err&msg=Zerodha+login+was+not+successful", status_code=303)
    if not req_tok:
        return RedirectResponse("/admin?token=err&msg=no+request+token+received", status_code=303)
    return _complete_kite_session(req_tok)


# fmt: on


@router.post("/admin/token")
def update_token(
    request: Request,
    request_token: str = Form(""),
    access_token: str = Form(""),
    owner_state: str = Form(""),
) -> RedirectResponse:
    if os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail":
        owner = owner_user_id(_read_env())
        if not owner or not valid_owner_session(request, owner):
            require_challenge_bound_form(request, owner_state)
            if access_token:
                raise HTTPException(status_code=401, detail="owner_login_required")
            return _finish_owner_login(request, owner_state, request_token)
    _check_auth(request)
    req_tok = _extract_request_token(request_token)
    acc_tok = access_token.strip()
    if req_tok:
        return _complete_kite_session(req_tok)
    if not acc_tok:
        return RedirectResponse("/admin?token=err&msg=no+token+provided", status_code=303)
    env = _read_env()
    api_key = next(
        (
            env.get(k, "").strip()
            for k in ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
            if env.get(k, "").strip()
        ),
        "",
    )
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
        has_key = any(
            env.get(k, "").strip()
            for k in ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
        )
        has_secret = any(
            env.get(k, "").strip()
            for k in ("BROKER_API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET")
        )
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


# fmt: off  # Short, stable JSON UI endpoints.
@router.get("/admin/api/overview")  # type: ignore[untyped-decorator]
def overview_api(request: Request) -> JSONResponse:
    """Show the existing sanitized runtime heartbeat before daily owner login.

    quick_overview returns fixed process, broker-auth state and release
    labels, never credentials, orders, account IDs or private journal lines.
    Browser ownership is required for *mutating* admin controls, not for
    recognizing a broker session already verified by the running engine.
    """
    if (
        os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail"
        and request.url.scheme != "https"
    ):
        raise HTTPException(status_code=403, detail="admin_https_required")
    return JSONResponse(quick_overview(), headers={"Cache-Control": "no-store"})


@router.get("/admin/api/logs")  # type: ignore[untyped-decorator]
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


# fmt: on


_STATUS_CACHE: dict[str, object] = {"at": 0.0, "label": "running", "color": "#3fb950"}


@router.get("/admin/status.json")
def status_json(request: Request) -> JSONResponse:
    """Expose only the process-status label; no private controls or logs."""
    if (
        os.getenv("DEPLOYMENT_PLATFORM") == "aws_lightsail"
        and request.url.scheme != "https"
    ):
        raise HTTPException(status_code=403, detail="admin_https_required")
    now = time.time()
    ttl = float(os.getenv("ADMIN_STATUS_CACHE_SECONDS", "10") or "10")
    if now - float(_STATUS_CACHE["at"]) < ttl:
        return JSONResponse(
            {"label": _STATUS_CACHE["label"], "color": _STATUS_CACHE["color"]},
            headers={"Cache-Control": "no-store"},
        )
    # Do not infer engine health from old log lines: use the same canonical
    # fast status snapshot as the visible dashboard, with no journal scan.
    try:
        overview = quick_overview()
        if overview.get("engine_ok") is not True:
            label, color = "unreachable", "#ff7586"
        elif overview.get("quiet") is True:
            label, color = "quiet", "#38d6a0"
        elif overview.get("bot_loaded") is True:
            label, color = "operational", "#38d6a0"
        else:
            label, color = "starting", "#f5bb5a"
    except Exception:
        label, color = "unknown", "#8b97a6"
    _STATUS_CACHE.update({"at": now, "label": label, "color": color})
    return JSONResponse(
        {"label": label, "color": color},
        headers={"Cache-Control": "no-store"},
    )


_TRADE_MARKERS = (
    "ORDER_SENT",
    "Sending Order",
    "FILLED",
    "average_price",
    "EXIT",
    "TRADE_ATTEMPT",
    "ORDER_REJECTED",
    "ORDER_BROKER_CONFIG_ERROR",
    "pnl",
    "TRADE_CLOSED",
    "ORDER_COMPLETE",
    "SIGNAL_GENERATED",
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
