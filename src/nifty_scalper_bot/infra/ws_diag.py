"""Diagnostics helpers for broker websocket connectivity."""

from __future__ import annotations

import importlib
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["WsDiag", "run_diag"]


@dataclass(slots=True)
class WsDiag:
    """Snapshot of websocket client and network health."""

    lib_kite: str
    lib_wsclient: str
    python: str
    ssl_version: str
    token_hint: str
    token_age_s: float | None
    ws_dns: list[str]
    tcp_ok: bool
    tcp_peer: str | None
    state: str
    last_error: str | None
    breaker_open: bool
    next_retry_s: float | None


def _ver(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return "missing"
    return getattr(module, "__version__", "unknown")


def _token_hint(token: str | None) -> str:
    if not token:
        return ""
    token = token.strip()
    if len(token) <= 8:
        return token
    colon = "yes" if ":" in token else "no"
    return f"{token[:4]}…{token[-4:]} (colon={colon})"


def run_diag(
    ws_manager: Any,
    ws_host: str | None,
    token: str | None,
    token_issued_at: float | None,
) -> WsDiag:
    """Collect diagnostic information about websocket health."""

    lib_kite = _ver("kiteconnect")
    lib_wsclient = _ver("websocket")
    python_version = sys.version.split()[0]
    ssl_version = ssl.OPENSSL_VERSION

    hint = _token_hint(token)
    age = None
    if token and token_issued_at:
        age = max(0.0, time.time() - float(token_issued_at))

    if ws_manager is None or not ws_host:
        return WsDiag(
            lib_kite=lib_kite,
            lib_wsclient=lib_wsclient,
            python=python_version,
            ssl_version=ssl_version,
            token_hint=hint,
            token_age_s=age,
            ws_dns=[],
            tcp_ok=False,
            tcp_peer=None,
            state="DISABLED",
            last_error="Websocket disabled; set WEBSOCKET__DISABLED=false to enable",
            breaker_open=False,
            next_retry_s=None,
        )

    ips: list[str] = []
    if ws_host:
        try:
            infos = socket.getaddrinfo(ws_host, 443, type=socket.SOCK_STREAM)
            for _family, _type, _proto, _canon, sockaddr in infos:
                ip = str(sockaddr[0])
                if ip not in ips:
                    ips.append(ip)
        except Exception:
            ips = []

    tcp_ok = False
    peer = None
    if ws_host:
        try:
            with socket.create_connection((ws_host, 443), timeout=3) as sock:
                peer = str(sock.getpeername()[0])
            tcp_ok = True
        except Exception:
            tcp_ok = False
            peer = None

    state = "unknown"
    last_error = None
    try:
        ws_state = ws_manager.connection_state()
        state = getattr(ws_state, "name", str(ws_state))
    except Exception:
        state = "unknown"
    try:
        err = ws_manager.last_error()
    except Exception:
        err = None
    if err is not None:
        last_error = f"{err.__class__.__name__}: {err}"

    breaker_open = False
    try:
        until = float(getattr(ws_manager, "_breaker_open_until", 0.0))
        breaker_open = until > time.monotonic()
    except Exception:
        breaker_open = False

    next_retry = None
    try:
        timer = getattr(ws_manager, "_reconnect_timer", None)
        if timer is not None:
            next_retry = float(getattr(ws_manager, "_backoff_s", 0.0)) or None
    except Exception:
        next_retry = None

    return WsDiag(
        lib_kite=lib_kite,
        lib_wsclient=lib_wsclient,
        python=python_version,
        ssl_version=ssl_version,
        token_hint=hint,
        token_age_s=age,
        ws_dns=ips,
        tcp_ok=tcp_ok,
        tcp_peer=peer,
        state=state,
        last_error=last_error,
        breaker_open=breaker_open,
        next_retry_s=next_retry,
    )
