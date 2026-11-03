"""Purpose: Archive finalized NIFTY trading-session logs off the hot path.

Responsibilities:
- build_daily_log_archiver uses GitHub-owned Supabase monitoring defaults.
- DailyLogArchiver.archive_once snapshots bounded IST journald sessions.
- Include source-only startup/recovery evidence in the existing archive summary.
- Archive bounded post-close parity and exact-contract markout summaries.
- archive_interval_seconds enforces a minimum reporting interval.

Boundaries: Observability only; the local journal/engine remain authoritative.
Safety: Archiving failures cannot create orders, change risk or enable LIVE.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from nifty_scalper_bot.config.defaults import DEFAULT_SUPABASE_ARCHIVE_URL
from nifty_scalper_bot.config.env_utils import (
    resolve_build_sha,
    supabase_telemetry_enabled,
)
from nifty_scalper_bot.infra.session_startup_audit import audit_session_startups
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_OPEN = time(9, 15)
_CLOSE = time(15, 30)
_RESEARCH_SETTLE = timedelta(minutes=3)
DEFAULT_ARCHIVE_URL = DEFAULT_SUPABASE_ARCHIVE_URL
Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def archive_interval_seconds() -> float:
    try:
        raw = os.getenv("SUPABASE_LOG_ARCHIVE_INTERVAL_SECONDS", "300")
        return max(60.0, float(raw))
    except (TypeError, ValueError):
        return 300.0


def build_daily_log_archiver() -> "DailyLogArchiver | None":
    if not supabase_telemetry_enabled("SUPABASE_LOG_ARCHIVE_ENABLED"):
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
        research_runner: Callable[[date, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self._endpoint_url = str(endpoint_url).strip()
        self._service_name = str(service_name).strip() or "niftybot"
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._transport = transport or _post_json
        self._log_reader = log_reader or self._read_journal
        self._capture_storage_probe_enabled = log_reader is None
        # The real host enables post-close research; injected log readers in
        # isolated tests require an explicitly provided research runner.
        if research_runner is None and log_reader is None:
            from nifty_scalper_bot.infra.market_research_report import (
                run_postmarket_research,
            )

            research_runner = run_postmarket_research
        self._research_runner = research_runner
        self._finalized_session_date: date | None = None

    def archive_once(self, now: datetime | None = None) -> dict[str, Any]:
        current = (now or datetime.now(timezone.utc)).astimezone(_IST)
        if current.weekday() >= 5 or current.time() < _OPEN:
            return {"archived": False, "reason": "outside_session"}

        session_date = current.date()
        start = datetime.combine(session_date, _OPEN, tzinfo=_IST)
        close = datetime.combine(session_date, _CLOSE, tzinfo=_IST)
        settled = current >= close + _RESEARCH_SETTLE
        if settled and self._finalized_session_date == session_date:
            return {"archived": False, "reason": "already_finalized"}
        end = min(current, close)
        text = self._log_reader(start, end)
        if not text.strip():
            return {"archived": False, "reason": "no_logs"}
        if self._capture_storage_probe_enabled:
            try:
                from nifty_scalper_bot.infra.market_research_report import (
                    capture_storage_status,
                )

                storage = capture_storage_status(session_date)
                prefix = f"[{current.strftime('%Y-%m-%d %H:%M:%S')} IST] "
                text += (
                    "\n"
                    + prefix
                    + "REPLAY_CAPTURE_STORAGE "
                    + json.dumps(storage, separators=(",", ":"), allow_nan=False)
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "REPLAY_CAPTURE_STORAGE_UNAVAILABLE error_type=%s",
                    type(exc).__name__,
                )

        # Keep replay and integrity sidecars out of the remote payload.
        # Run only after the final options session and writer settle window;
        # append bounded results to the already-authorized journal transport.
        if settled and self._research_runner is not None:
            try:
                from nifty_scalper_bot.infra.market_research_report import (
                    research_report_log_lines,
                )

                research = self._research_runner(session_date, text)
                lines_for_archive = research_report_log_lines(research)
            except Exception as exc:  # noqa: BLE001 - archive must survive
                LOGGER.warning(
                    "RESEARCH_POSTMARKET_UNAVAILABLE error_type=%s",
                    type(exc).__name__,
                )
                lines_for_archive = [
                    "RESEARCH_SESSION_EVIDENCE "
                    + '{"v":1,"status":"unavailable","blocker":"postmarket_runner_failed"}'
                ]
            if lines_for_archive:
                prefix = f"[{current.strftime('%Y-%m-%d %H:%M:%S')} IST] "
                text += "\n" + "\n".join(prefix + line for line in lines_for_archive)
        lines = text.splitlines()
        payload = {
            "trading_date": session_date.isoformat(),
            "captured_at": current.astimezone(timezone.utc).isoformat(),
            "source": "lightsail-journald",
            "build_sha": resolve_build_sha(),
            "line_count": len(lines),
            "byte_count": len(text.encode("utf-8")),
            "log_text": text,
            "startup_audit": audit_session_startups(text),
            "first_log_at": start.astimezone(timezone.utc).isoformat(),
            "last_log_at": end.astimezone(timezone.utc).isoformat(),
            "finalized": settled,
        }
        response = self._transport(self._endpoint_url, payload, self._timeout_seconds)
        if response.get("ok") is not True:
            raise RuntimeError(f"daily log archive rejected: {response}")
        if payload["finalized"]:
            self._finalized_session_date = session_date
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
