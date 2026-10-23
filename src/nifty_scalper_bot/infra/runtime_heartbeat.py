"""Private, quota-independent Supabase health mirror for the Lightsail engine.

This reads only the local readiness endpoint, and posts sanitized diagnostics
off the trading hot path. It never sends orders, tokens, balances or prices.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import requests

from nifty_scalper_bot.config.env_utils import resolve_build_sha

DEFAULT_HEARTBEAT_URL = (
    "https://dehdptgkqbrkyzyodicd.supabase.co/functions/v1/"
    "nifty-runtime-heartbeat-ingest"
)
_HEX_SHA = re.compile(r"^[0-9a-f]{7,40}$")
HeartbeatTransport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]
HealthReader = Callable[[], tuple[int, Mapping[str, Any]]]


def heartbeat_interval_seconds() -> float:
    try:
        return max(
            60.0, float(os.getenv("SUPABASE_RUNTIME_HEARTBEAT_INTERVAL_SECONDS", "120"))
        )
    except (TypeError, ValueError):
        return 120.0


def build_runtime_heartbeat() -> "RuntimeHeartbeat | None":
    if os.getenv("SUPABASE_RUNTIME_HEARTBEAT_ENABLED", "false").strip().lower() not in {
        "true",
        "1",
        "yes",
        "on",
    }:
        return None
    return RuntimeHeartbeat(
        endpoint_url=os.getenv("SUPABASE_RUNTIME_HEARTBEAT_URL", DEFAULT_HEARTBEAT_URL)
    )


class RuntimeHeartbeat:
    """Emit only whitelisted local health fields through private ingestion."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        health_reader: HealthReader | None = None,
        transport: HeartbeatTransport | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.health_reader = health_reader or _read_local_health
        self.transport = transport or _post_json
        self.timeout_seconds = max(1.0, min(10.0, float(timeout_seconds)))

    def report_once(self, now: datetime | None = None) -> dict[str, Any]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        engine_healthy = False
        http_status: int | None = None
        status: str | None = None
        failure_reason: str | None = None
        blockers: list[str] = []
        try:
            http_status, raw = self.health_reader()
            # QUIET is a planned unloaded-engine state, not a process outage.
            engine_healthy = http_status == 200 and raw.get("status") != "quiet"
            if http_status == 200:
                candidate = raw.get("status")
                if isinstance(candidate, str) and re.fullmatch(
                    r"[a-zA-Z0-9_:-]{1,90}", candidate
                ):
                    status = candidate
                block_list = raw.get("blockers")
                if isinstance(block_list, list):
                    blockers = [
                        item
                        for item in block_list[:20]
                        if isinstance(item, str)
                        and re.fullmatch(r"[a-zA-Z0-9_:-]{1,90}", item)
                    ]
            else:
                failure_reason = "local_health_http_error"
        except (requests.RequestException, TimeoutError, OSError, ValueError):
            failure_reason = "local_health_unavailable"
        sha = str(resolve_build_sha() or "").strip().lower()
        payload = {
            "observed_at": current.isoformat(),
            "build_sha": sha if _HEX_SHA.fullmatch(sha) else None,
            "engine_healthy": engine_healthy,
            "http_status": http_status,
            "trading_status": status,
            "blockers": blockers,
            "failure_reason": failure_reason,
        }
        reply = self.transport(self.endpoint_url, payload, self.timeout_seconds)
        if reply.get("ok") is not True:
            raise RuntimeError("runtime_heartbeat_ingest_failed")
        return {"published": True, "engine_healthy": engine_healthy}


def _read_local_health() -> tuple[int, Mapping[str, Any]]:
    try:
        port = int(os.getenv("PORT", "8080"))
    except ValueError:
        port = 8080
    if port < 1 or port > 65535:
        port = 8080
    reply = requests.get(
        f"http://127.0.0.1:{port}/health/trading",
        timeout=2.0,
        allow_redirects=False,
    )
    if reply.status_code == 404:
        # AUTO's off-hours ASGI wrapper intentionally has no trading endpoint.
        live = requests.get(
            f"http://127.0.0.1:{port}/livez",
            timeout=2.0,
            allow_redirects=False,
        )
        if live.status_code == 200:
            body = live.json()
            if (
                isinstance(body, Mapping)
                and body.get("quiet") is True
                and body.get("bot_loaded") is False
                and body.get("engine_http_responsive") is True
            ):
                return 200, {"status": "quiet", "blockers": ["auto_quiet"]}
    body = reply.json() if reply.status_code == 200 else {}
    return reply.status_code, body if isinstance(body, Mapping) else {}


def _post_json(
    url: str, payload: Mapping[str, Any], timeout_seconds: float
) -> Mapping[str, Any]:
    reply = requests.post(
        url,
        json=dict(payload),
        timeout=timeout_seconds,
        headers={
            "Content-Type": "application/json",
            "X-Nifty-Source": "lightsail-heartbeat-v1",
        },
        allow_redirects=False,
    )
    reply.raise_for_status()
    parsed = reply.json()
    if not isinstance(parsed, Mapping):
        raise RuntimeError("runtime_heartbeat_invalid_response")
    return parsed


__all__ = [
    "DEFAULT_HEARTBEAT_URL",
    "RuntimeHeartbeat",
    "build_runtime_heartbeat",
    "heartbeat_interval_seconds",
]
