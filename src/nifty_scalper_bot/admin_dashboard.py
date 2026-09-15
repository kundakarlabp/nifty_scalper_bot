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

from fastapi import APIRouter, Form, Request
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


def _activate_access_token(access_token: str) -> None:
    _write_env({
        "KITE_ACCESS_TOKEN": access_token,
        "ZERODHA_ACCESS_TOKEN": access_token,
        "BROKER_ACCESS_TOKEN": access_token,
    })
    _restart_service()


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
    _activate_access_token(result)
    return RedirectResponse("/admin?token=ok", status_code=303)


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


def _restart_service() -> None:
    try:
        subprocess.Popen(
            ["sudo", "systemctl", "restart", "--no-block", SERVICE_NAME],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except Exception:
        os._exit(0)


def _git_update() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(APP_DIR), "pull", "--ff-only", "origin", "main"],
            capture_output=True, text=True, timeout=60,
        )
        log = (out.stdout + out.stderr).strip()
        if out.returncode == 0:
            _restart_service()
            return True, log
        return False, log
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------- UI ----------------

_CSS = """
:root{--bg:#0a0e14;--card:#141b24;--bd:#222d3a;--fg:#e6edf3;--mut:#8b97a6;--grn:#2ea043;--red:#da3633;--amb:#bb8009;--blu:#1f6feb}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,Segoe UI,Arial;background:var(--bg);color:var(--fg);margin:0}.top{position:sticky;top:0;background:#0d1420ee;border-bottom:1px solid var(--bd);padding:12px 18px;display:flex;align-items:center;gap:14px;z-index:5}.top h1{font-size:16px;margin:0}.top .sp{flex:1}.pill{padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}.pill.on{background:#0c2a16;color:#3fb950;border:1px solid #1c5c30}.pill.off{background:#2a210c;color:#d29922;border:1px solid #5c4a1c}.wrap{max-width:900px;margin:18px auto;padding:0 16px}.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px;margin-bottom:16px}.card h2{margin:0 0 4px;font-size:15px}.card p{margin:4px 0 14px;font-size:13px;color:var(--mut)}label{display:block;font-size:12px;color:var(--mut);margin:12px 0 5px}input{width:100%;padding:11px 12px;border-radius:9px;border:1px solid var(--bd);background:#0a1019;color:var(--fg);font-size:15px}.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;background:#0a1019;color:var(--mut);border:1px solid var(--bd)}button,.btn{margin-top:14px;padding:11px 16px;border:0;border-radius:9px;background:var(--grn);color:#fff;font-size:14px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}button.blu,.btn.blu{background:var(--blu)}button.amb,.btn.amb{background:var(--amb)}button.gray,.btn.gray{background:#30363d}.row{display:flex;gap:10px;flex-wrap:wrap}.muted{font-size:12px;color:var(--mut)}.flash{padding:10px 12px;border-radius:9px;margin-bottom:14px;font-size:13px}.flash.ok{background:#0c2a16;color:#3fb950}.flash.err{background:#2a0c0c;color:#f85149}.session{border-color:#315d92;background:#101b2a}.session .btn{font-size:15px;padding:12px 18px}
"""
_PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Nifty Bot</title><style>{css}</style></head><body>{body}</body></html>"""


def _page(body: str) -> str:
    return _PAGE.format(css=_CSS, body=body)


def _topbar(live_on: bool) -> str:
    pill = '<span class="pill on">● LIVE</span>' if live_on else '<span class="pill off">● SHADOW</span>'
    return (f'<div class=top><h1>⚡ Nifty Scalper Bot</h1>{pill}<span class=sp></span>'
            f'<a class="btn gray" href="/admin">Dashboard</a>'
            f'<a class="btn blu" href="/admin/logs/download">Download Logs</a></div>')


def _flash(request: Request) -> str:
    q = request.query_params
    if q.get("saved"): return '<div class="flash ok">Settings saved. Restart to apply.</div>'
    if q.get("token") == "ok": return '<div class="flash ok">Zerodha authenticated. Access token verified, saved and bot restart requested.</div>'
    if q.get("token") == "err": return f'<div class="flash err">Zerodha session failed: {html.escape(q.get("msg", ""))}</div>'
    if q.get("mode") == "err": return '<div class="flash err">Cannot go LIVE: enter API key, secret and access token first.</div>'
    if q.get("mode"): return '<div class="flash ok">Mode changed and bot restarted.</div>'
    if q.get("restart"): return '<div class="flash ok">Bot restarting…</div>'
    if q.get("upd") == "ok": return '<div class="flash ok">Updated from GitHub and restarted.</div>'
    if q.get("upd") == "err": return f'<div class="flash err">Update failed: {html.escape(q.get("msg", ""))}</div>'
    return ""


@router.get("/admin", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    _check_auth(request)
    env = _read_env()
    live_on = (env.get("ENABLE_LIVE", "false").strip().lower() in {"1", "true", "yes", "on"}
               and env.get("EXECUTION_MODE", "SHADOW").strip().upper() == "LIVE")
    api_key = env.get("KITE_API_KEY", "").strip()
    login_url = _kite_login_url(api_key) if api_key else ""
    login_button = (f'<a class="btn blu" href="{html.escape(login_url, quote=True)}">Authenticate Zerodha</a>'
                    if login_url else '<span class="muted">Save the Zerodha API key and secret below first.</span>')
    toggle = (
        f'<form method=post action="/admin/mode"><input type=hidden name=mode value="{"shadow" if live_on else "live"}">'
        + ('<button class=amb type=submit>Switch to SHADOW (stop real trading)</button>' if live_on
           else '<button type=submit onclick="return confirm(\'Turn ON live trading with REAL money?\')">Turn ON Live Trading</button>')
        + '</form>')
    rows = ""
    for label, key, sec in FIELDS:
        v = env.get(key, "")
        shown = ("••••••••" if v else "") if sec else v
        rows += f'<label>{label} <span class=badge>{key}</span></label><input name="{key}" value="{html.escape(shown, quote=True)}" placeholder="(unchanged)">'
    body = f"""{_topbar(live_on)}<div class=wrap>{_flash(request)}
    <div class="card session"><h2>Daily Zerodha Session</h2>
    <p><b>Recommended:</b> click Authenticate Zerodha, complete the official Zerodha login/TOTP screen, and return automatically. The server then exchanges, verifies and saves the access token and restarts the bot.</p>
    {login_button}
    <p class=muted style="margin-top:12px">For automatic return, set the Kite app redirect URL to this server's <b>/admin/kite/callback</b> endpoint.</p>
    <form method=post action="/admin/token">
    <label>Fallback — paste the complete redirect URL or request token</label>
    <input name="request_token" autocomplete="off" placeholder="paste callback URL or request_token">
    <button type=submit>Generate access token &amp; restart</button></form></div>

    <div class=card><h2>Live Trading</h2><p>{'Placing REAL orders with REAL money.' if live_on else 'Analysing only — no real orders.'}</p>{toggle}</div>

    <div class=card><h2>Credentials &amp; Settings</h2><p>Secrets show dots; leave them unchanged unless replacing them.</p>
    <form method=post action="/admin/save">{rows}<button type=submit>Save settings</button></form></div>

    <div class=card><h2>Controls</h2><div class=row>
    <form method=post action="/admin/update"><button class=blu type=submit>Update from GitHub</button></form>
    <form method=post action="/admin/restart"><button class=amb type=submit>Restart Bot</button></form>
    <a class="btn gray" href="/health">Health</a></div></div></div>"""
    return HTMLResponse(_page(body))


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
    _activate_access_token(acc_tok)
    return RedirectResponse("/admin?token=ok", status_code=303)


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
    _restart_service()
    return RedirectResponse("/admin?mode=1", status_code=303)


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


@router.post("/admin/restart")
def restart(request: Request) -> RedirectResponse:
    _check_auth(request)
    _restart_service()
    return RedirectResponse("/admin?restart=1", status_code=303)


@router.post("/admin/update")
def update_from_github(request: Request) -> RedirectResponse:
    _check_auth(request)
    ok, log = _git_update()
    if ok:
        return RedirectResponse("/admin?upd=ok", status_code=303)
    return RedirectResponse(f"/admin?upd=err&msg={urllib.parse.quote(log[:120])}", status_code=303)


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
