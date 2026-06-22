"""Read-only monitoring endpoints for the Streamlit console.

The endpoints expose cleaned operational log messages only. They never expose
configuration files, environment variables, credentials, or trading controls.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, time

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from nifty_scalper_bot.admin_dashboard import _gather_logs

router = APIRouter(prefix="/monitor", tags=["monitoring"])

_STAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) IST\s+(?P<message>.*)$"
)
_SECRET_RE = re.compile(
    r"(?i)(api[_ -]?key|api[_ -]?secret|access[_ -]?token|request[_ -]?token|password)\s*[:=]\s*\S+"
)
_TRADE_MARKERS = (
    "ORDER_SENT", "FILLED", "TRADE_ATTEMPT", "ORDER_REJECTED", "TRADE_CLOSED",
    "ORDER_COMPLETE", "SIGNAL_GENERATED", "EXIT", "PNL", "TARGET", "STOP_LOSS",
)


def _redact(message: str) -> str:
    return _SECRET_RE.sub(r"\1=[REDACTED]", message)


def _parse_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _STAMP_RE.match(line.strip())
        if not match:
            continue
        message = _redact(match.group("message"))
        upper = message.upper()
        if "ERROR" in upper or "FAIL" in upper or "❌" in message:
            level = "ERROR"
        elif "WARN" in upper or "⚠" in message:
            level = "WARNING"
        elif any(marker in upper for marker in _TRADE_MARKERS):
            level = "TRADE"
        else:
            level = "INFO"
        rows.append(
            {
                "timestamp_ist": match.group("timestamp") + " IST",
                "level": level,
                "message": message,
            }
        )
    return rows


@router.get("/logs")
def recent_logs(
    lines: int = Query(500, ge=50, le=5000),
    contains: str = Query("", max_length=120),
    level: str = Query("ALL", pattern="^(ALL|INFO|WARNING|ERROR|TRADE)$"),
) -> JSONResponse:
    rows = _parse_rows(_gather_logs(lines, contains=contains, clean=True))
    if level != "ALL":
        rows = [row for row in rows if row["level"] == level]
    return JSONResponse({"rows": rows, "count": len(rows)})


@router.get("/logs.csv", response_class=PlainTextResponse)
def logs_csv(
    trading_date: date = Query(default_factory=date.today),
    start_time: time = Query(time(9, 15)),
    end_time: time = Query(time(15, 30)),
    contains: str = Query("", max_length=120),
    level: str = Query("ALL", pattern="^(ALL|INFO|WARNING|ERROR|TRADE)$"),
) -> PlainTextResponse:
    since = f"{trading_date.isoformat()} {start_time.strftime('%H:%M:%S')}"
    until = f"{trading_date.isoformat()} {end_time.strftime('%H:%M:%S')}"
    rows = _parse_rows(
        _gather_logs(20000, since=since, until=until, contains=contains, clean=True)
    )
    if level != "ALL":
        rows = [row for row in rows if row["level"] == level]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp_ist", "level", "message"])
    writer.writeheader()
    writer.writerows(rows)
    filename = (
        f"niftybot-logs-{trading_date.isoformat()}-"
        f"{start_time.strftime('%H%M')}-{end_time.strftime('%H%M')}.csv"
    )
    return PlainTextResponse(
        output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
