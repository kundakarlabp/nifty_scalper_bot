"""Filesystem, service and status helpers for the superlite admin process."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from fastapi import HTTPException, Request

APP_DIR = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))
ENV_PATH = Path(os.getenv("BOT_ENV_FILE", "/home/ubuntu/.config/niftybot/niftybot.env"))
STATUS_PATH = Path(os.getenv("BOT_UPDATE_STATUS_FILE", str(APP_DIR / "data/auto_update_status.json")))
ENGINE_URL = os.getenv("BOT_API_URL", "http://127.0.0.1:8080").rstrip("/")
ENGINE_SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
STREAMLIT_SERVICE = os.getenv("BOT_STREAMLIT_SERVICE_NAME", "niftybot-streamlit")
DEPLOY_SERVICE = os.getenv("BOT_DEPLOY_SERVICE_NAME", "niftybot-autodeploy.service")
SECRET_RE = re.compile(
    r"(?i)(api[_ -]?key|api[_ -]?secret|access[_ -]?token|request[_ -]?token|password)"
    r"\s*[:=]\s*\S+"
)
SELECTED_PAIR_LOG_PATTERNS = (
    re.compile(
        r"ACTIVE_DYNAMIC_BASKET_COMMITTED.*?selected_ce=(\S+).*?selected_pe=(\S+).*?atm_strike=(\S+)"
    ),
    re.compile(
        r"CONTRACT_SSOT_ATM_PAIR_SELECTED.*?selected_ce=(\S+).*?selected_pe=(\S+).*?atm_strike=(\S+)"
    ),
)
ENV_CACHE: dict[str, Any] = {"at": 0.0, "data": {}}
STATUS_CACHE: dict[str, Any] = {"at": 0.0, "data": {}}


def read_env() -> dict[str, str]:
    now = time.monotonic()
    if now - float(ENV_CACHE["at"]) < 3 and ENV_CACHE["data"]:
        return dict(ENV_CACHE["data"])
    data: dict[str, str] = {}
    try:
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    ENV_CACHE.update({"at": now, "data": dict(data)})
    return data


def write_env(updates: dict[str, str]) -> None:
    """Atomically update supplied keys and retain every unspecified credential."""
    existing = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    output: list[str] = []
    seen: set[str] = set()
    for raw in existing:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(raw)
    output.extend(f"{key}={value}" for key, value in updates.items() if key not in seen)
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".niftybot.env.", dir=str(ENV_PATH.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write("\n".join(output).rstrip() + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, ENV_PATH)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    ENV_CACHE.update({"at": 0.0, "data": {}})


def same_origin(request: Request) -> None:
    """Keep the UI password-free while rejecting cross-site browser form posts."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if source and urllib.parse.urlparse(source).netloc != request.headers.get("host", ""):
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def systemctl(*args: str) -> None:
    subprocess.Popen(
        ["sudo", "systemctl", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def restart(service: str) -> None:
    systemctl("restart", "--no-block", service)


def _http_json(path: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(ENGINE_URL + path, timeout=1.2) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else {}
    except TimeoutError:
        return {"_error": "ENGINE HTTP TIMEOUT"}
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError):
            return {"_error": "ENGINE HTTP TIMEOUT"}
        return {"_error": "ENGINE HTTP UNRESPONSIVE"}
    except (OSError, ValueError):
        return {"_error": "ENGINE HTTP UNRESPONSIVE"}


def _git_ref(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(APP_DIR), "rev-parse", "--short", ref],
            text=True,
            timeout=1.5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "—"


def bounded_logs(lines: int = 400, contains: str = "") -> str:
    limit = max(50, min(int(lines), 5000))
    try:
        result = subprocess.run(
            ["journalctl", "-u", ENGINE_SERVICE, "-n", str(limit), "--no-pager", "-o", "cat"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        text = result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        text = ""
    rows = [SECRET_RE.sub(r"\1=[REDACTED]", row) for row in text.splitlines()]
    needle = contains.strip().lower()
    if needle:
        rows = [row for row in rows if needle in row.lower()]
    return "\n".join(rows)


def _parse_status_timestamp(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _update_state() -> dict[str, Any]:
    try:
        value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        data = value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
    transient = {"fetching", "validating", "deploying", "restarting"}
    state = str(data.get("state") or "").lower()
    if state in transient:
        timeout = float(os.getenv("BOT_UPDATER_STALE_TIMEOUT_SECONDS", "900") or 900)
        raw_ts = data.get("updated_at") or data.get("updated_ts") or data.get("ts") or data.get("timestamp")
        parsed_ts = _parse_status_timestamp(raw_ts)
        if parsed_ts is None:
            stale = dict(data)
            stale["previous_state"] = data.get("state")
            stale["state"] = "stale_interrupted"
            stale["stale"] = True
            stale["stale_reason"] = "malformed_timestamp"
            stale["stale_after_seconds"] = timeout
            return stale
        age = time.time() - parsed_ts
        if age > timeout:
            stale = dict(data)
            stale["previous_state"] = data.get("state")
            stale["state"] = "stale_interrupted"
            stale["stale"] = True
            stale["stale_after_seconds"] = timeout
            return stale
    return data


def _service_process_known() -> bool | None:
    try:
        result = subprocess.run(["systemctl", "is-active", "--quiet", ENGINE_SERVICE], timeout=1.0, check=False)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None


def _clean_selected_value(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text in {"None", "none", "null", "—", "-"} else text


def _normalise_selected_payload(payload: Mapping[str, Any] | None, *, source: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    ce = _clean_selected_value(
        payload.get("ce")
        or payload.get("selected_ce")
        or payload.get("atm_ce")
        or payload.get("call")
    )
    pe = _clean_selected_value(
        payload.get("pe")
        or payload.get("selected_pe")
        or payload.get("atm_pe")
        or payload.get("put")
    )
    atm = payload.get("atm") or payload.get("atm_strike") or payload.get("strike")
    if not ce and not pe:
        return {}
    result: dict[str, Any] = {"_source": source}
    if ce:
        result["ce"] = ce
    if pe:
        result["pe"] = pe
    if atm not in (None, ""):
        result["atm"] = atm
    for key in ("ce_ready", "pe_ready", "execution_ready", "evaluation_ready", "basket_version"):
        if key in payload:
            result[key] = payload[key]
    return result


def _extract_selected_from_structured(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    for payload_index, payload in enumerate(payloads):
        if not isinstance(payload, Mapping):
            continue
        source_name = "health_trading" if payload_index == 0 else "trading_status"
        for key in ("selected", "selected_options", "active_selection", "active_basket"):
            selected = _normalise_selected_payload(payload.get(key), source=f"{source_name}.{key}")
            if selected:
                return selected
        selected = _normalise_selected_payload(payload, source=f"{source_name}.top_level")
        if selected:
            return selected
    return {}


def _extract_selected_from_logs(recent_logs: str) -> dict[str, Any]:
    for line in reversed(str(recent_logs or "").splitlines()):
        for pattern in SELECTED_PAIR_LOG_PATTERNS:
            match = pattern.search(line)
            if match:
                return {
                    "ce": match.group(1),
                    "pe": match.group(2),
                    "atm": match.group(3),
                    "_source": "journal:active_dynamic_basket" if "ACTIVE_DYNAMIC_BASKET_COMMITTED" in line else "journal:contract_ssot",
                }
    return {}


def status_snapshot() -> dict[str, Any]:
    now = time.monotonic()
    if now - float(STATUS_CACHE["at"]) < 5 and STATUS_CACHE["data"]:
        return dict(STATUS_CACHE["data"])
    livez = _http_json("/livez")
    trading = _http_json("/health/trading")
    mode = _http_json("/trading/status")
    engine_http_responsive = not any((payload or {}).get("_error") for payload in (livez, trading, mode))
    blockers = [str(value) for value in trading.get("blockers") or [] if str(value).strip()]
    execution_only = {"not_live_mode", "market_closed", "exchange_holiday", "outside_session"}
    operational_blockers = [value for value in blockers if value not in execution_only]
    recent = bounded_logs(180)
    selected_options = _extract_selected_from_structured(trading, mode) or _extract_selected_from_logs(recent)
    selected_source = selected_options.pop("_source", None) if isinstance(selected_options, dict) else None
    running, remote = _git_ref("HEAD"), _git_ref("origin/main")
    data = {
        "service_process_known": _service_process_known(),
        "process_up": bool(livez) and engine_http_responsive,
        "engine_http_responsive": engine_http_responsive,
        "engine_http_status": "RESPONSIVE" if engine_http_responsive else (livez.get("_error") or trading.get("_error") or mode.get("_error") or "ENGINE HTTP UNRESPONSIVE"),
        "bot_loaded": bool(livez.get("bot_loaded")) if engine_http_responsive else None,
        "engine_loaded": bool(livez.get("bot_loaded")) if engine_http_responsive else False,
        "operational_ready": bool(livez.get("bot_loaded")) and engine_http_responsive and not operational_blockers,
        "evaluation_ready": bool(trading.get("evaluation_ready") or trading.get("ready")) if engine_http_responsive else None,
        "live_orders_armed": bool(trading.get("live_orders_armed")) if engine_http_responsive else False,
        "broker_authenticated": (
            trading.get("broker_authentication")
            or (trading.get("broker") or {}).get("authentication")
            or "unknown"
        ) if engine_http_responsive else "unknown",
        "reconciled": (trading.get("reconciliation") or {}).get("completed") if engine_http_responsive else None,
        "mode": str(mode.get("execution_mode") or ("ENGINE HTTP TIMEOUT" if not engine_http_responsive else "UNKNOWN")).upper(),
        "broker": trading.get("broker") or {},
        "reconciliation": trading.get("reconciliation") or {},
        "primary_blocker": trading.get("primary_blocker") or (blockers[0] if blockers else None),
        "blockers": blockers,
        "operational_blockers": operational_blockers,
        "selected": selected_options,
        "selected_source": selected_source,
        "running": running,
        "remote": remote,
        "stale": running not in {"—", ""} and remote not in {"—", ""} and running != remote,
        "updater": _update_state(),
    }
    STATUS_CACHE.update({"at": now, "data": dict(data)})
    return data


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> tuple[bool, str]:
    try:
        checksum = hashlib.sha256((api_key + request_token + api_secret).encode()).hexdigest()
        payload = urllib.parse.urlencode(
            {"api_key": api_key, "request_token": request_token, "checksum": checksum}
        ).encode()
        request = urllib.request.Request(
            "https://api.kite.trade/session/token",
            data=payload,
            headers={"X-Kite-Version": "3", "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        token = str((result.get("data") or {}).get("access_token") or "")
        return (True, token) if token else (False, "access token missing in broker response")
    except Exception as exc:
        return False, str(exc)[:160]
