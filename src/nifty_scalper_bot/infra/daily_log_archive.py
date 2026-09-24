"""Off-hot-path archival of the complete trading-session service log."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_OPEN = time(9, 15)
_CLOSE = time(15, 30)
DEFAULT_ARCHIVE_URL = (
    "https://dehdptgkqbrkyzyodicd.supabase.co/functions/v1/nifty-log-archive-ingest"
)
Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def archive_interval_seconds() -> float:
    try:
        raw = os.getenv("SUPABASE_LOG_ARCHIVE_INTERVAL_SECONDS", "300")
        return max(60.0, float(raw))
    except (TypeError, ValueError):
        return 300.0


def build_daily_log_archiver() -> "DailyLogArchiver | None":
    raw = os.getenv("SUPABASE_LOG_ARCHIVE_ENABLED", "false").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None
    return DailyLogArchiver(
        endpoint_url=os.getenv("SUPABASE_LOG_ARCHIVE_URL", DEFAULT_ARCHIVE_URL),
        service_name=os.getenv("BOT_SERVICE_NAME", "niftybot"),
    )


class DailyLogArchiver:
    """Snapshot one IST trading session from journald into the remote archive."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        service_name: str = "niftybot",
        timeout_seconds: float = 15.0,
        transport: Transport | None = None,
        log_reader: Callable[[datetime, datetime], str] | None = None,
    ) -> None:
        self._endpoint_url = str(endpoint_url).strip()
        self._service_name = str(service_name).strip() or "niftybot"
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._transport = transport or _post_json
        self._log_reader = log_reader or self._read_journal

    def archive_once(self, now: datetime | None = None) -> dict[str, Any]:
        current = (now or datetime.now(timezone.utc)).astimezone(_IST)
        if current.weekday() >= 5 or current.time() < _OPEN:
            return {"archived": False, "reason": "outside_session"}

        session_date = current.date()
        start = datetime.combine(session_date, _OPEN, tzinfo=_IST)
        close = datetime.combine(session_date, _CLOSE, tzinfo=_IST)
        end = min(current, close)
        text = self._log_reader(start, end)
        if not text.strip():
            return {"archived": False, "reason": "no_logs"}

        lines = text.splitlines()
        payload = {
            "trading_date": session_date.isoformat(),
            "captured_at": current.astimezone(timezone.utc).isoformat(),
            "source": "lightsail-journald",
            "build_sha": os.getenv("GIT_SHA") or os.getenv("BUILD_SHA") or None,
            "line_count": len(lines),
            "byte_count": len(text.encode("utf-8")),
            "log_text": text,
            "first_log_at": start.astimezone(timezone.utc).isoformat(),
            "last_log_at": end.astimezone(timezone.utc).isoformat(),
            "finalized": current >= close,
        }
        response = self._transport(self._endpoint_url, payload, self._timeout_seconds)
        if response.get("ok") is not True:
            raise RuntimeError(f"daily log archive rejected: {response}")
        return {
            "archived": True,
            "line_count": len(lines),
            "finalized": bool(payload["finalized"]),
        }

    def _read_journal(self, start: datetime, end: datetime) -> str:
        result = subprocess.run(
            [
                "journalctl",
                "-u",
                self._service_name,
                "--since",
                start.isoformat(),
                "--until",
                end.isoformat(),
                "--no-pager",
                "-o",
                "cat",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "journalctl failed")
        return result.stdout.rstrip()


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    response = requests.post(
        url,
        json=dict(payload),
        timeout=timeout_seconds,
        headers={"Content-Type": "application/json", "X-Nifty-Source": "lightsail"},
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, Mapping):
        raise RuntimeError("daily log archive returned a non-object response")
    return body


__all__ = [
    "DEFAULT_ARCHIVE_URL",
    "DailyLogArchiver",
    "archive_interval_seconds",
    "build_daily_log_archiver",
]
