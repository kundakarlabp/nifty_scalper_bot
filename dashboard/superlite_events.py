"""Low-overhead parsing of actionable Nifty bot journal events."""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable

from dashboard.event_buffer import SECRET

STAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} IST)\]")
FIELD = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]*)=([^\s,}]+)")
TRACE = re.compile(r"\btrace_id=([^\s,}]+)")
RESULT = re.compile(r"\baccepted=([^\s]+).*?\breason=([^\s]+)")
BASE_COLUMNS = ["timestamp_ist", "type", "message"]
DETAIL_COLUMNS = [
    "event", "symbol", "reason", "source", "attempt", "required_bars",
    "rows", "incoming_ts", "last_ts", "blocker", "failure_reason",
    "trace_id", "accepted",
]
HISTORY_DIAGNOSTICS = {"CANONICAL_HISTORY_RESULT", "RUNNER_HISTORY_SYNC_RESULT"}
EXPECTED_REJECTIONS = {
    "CANDIDATE_REJECTED",
    "SIGNAL_REJECTED",
    "SIGNAL_EXECUTION_RESULT",
    "ORDER_READINESS_REJECTED",
}
HARD_ERRORS = {
    "TRACEBACK",
    "CRITICAL",
    "UNHANDLED EXCEPTION",
    "RUNNER_ON_TICK_ERROR",
    "ORDER_FAILED",
    "STARTUP_FAILED",
    "HANDLER CRASHED",
    "FATAL",
    "IMPORTERROR",
    "MEMORYERROR",
    "BROKER_AUTH_INVALID",
}
BENIGN = {
    "HEARTBEAT",
    "TELEGRAM_POLLING",
    "NO SIGNAL",
    "MARKET CLOSED",
    "OUTSIDE SESSION",
    "INDICATOR_HISTORY_READY",
    "INDICATOR_HISTORY_RESEEDED",
    "INDICATOR_WARMUP",
    "INDICATOR_HISTORY_MISSING_OFFMARKET",
}
EVENT_WORDS = {
    "SIGNAL", "ORDER", "FILLED", "ENTRY", "EXIT", "TARGET", "STOP", "PNL",
    "POSITION", "BROKER", "AUTH", "READY", "BLOCKER", "RISK", "COOLDOWN",
    "ERROR", "FAIL", "WARN", "DEGRADED", "STARTUP", "SHUTDOWN", "HYDRATION",
    "SUBSCRIPTION", "RECONCIL", "CONTRACT_SSOT", "LIVE_READINESS",
    *HISTORY_DIAGNOSTICS,
}
NULLS = {"", "none", "null", "nil", "false", "0", "unknown", "n/a", "na"}
SOFT_HISTORY_ROLES = {"option_context"}
SOFT_HISTORY_FAILURE_REASONS = {"broker_fetch_not_allowed"}
TRADE_WORDS = {"ORDER_SENT", "ORDER_PLACED", "ORDER_FILLED", "FILLED", "ENTRY", "EXIT", "TARGET_HIT", "STOP_HIT", "STOP_LOSS", "PNL", "POSITION_OPENED", "POSITION_CLOSED", "TRADE_ATTEMPT"}
NON_TRADE_SYSTEM_WORDS = {"READINESS", "BLOCKER", "CANDLE", "HEARTBEAT", "SUMMARY", "SELECTED_OPTION_SUBSCRIPTION_STATE", "RUNNER_EVAL_DECISION", "NO_TRADE", *HISTORY_DIAGNOSTICS}


def fields(message: str) -> dict[str, str]:
    return {key.lower(): value for key, value in FIELD.findall(message)}


def _history_diagnostic(upper: str) -> bool:
    return any(token in upper for token in HISTORY_DIAGNOSTICS)


def _soft_non_gating_history_message(upper: str, values: dict[str, str]) -> bool:
    """Return True for expected non-selected option-context history misses."""

    role = values.get("role", "").strip().lower()
    failure = values.get("failure_reason", "").strip().lower()
    return (
        "CANONICAL_HISTORY_RESULT" in upper
        and role in SOFT_HISTORY_ROLES
        and failure in SOFT_HISTORY_FAILURE_REASONS
    )


def _detail_fields(message: str, values: dict[str, str]) -> dict[str, str]:
    details = {key: values[key] for key in DETAIL_COLUMNS if key in values}
    upper = message.upper()
    if "DATA_INTEGRITY_ERROR" in upper and "reason" not in details:
        details["reason"] = "missing_structured_details"
        details.setdefault("source", "journal_message")
    return details


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    names = list(BASE_COLUMNS)
    for key in DETAIL_COLUMNS:
        if any(row.get(key) for row in rows):
            names.append(key)
    return names


def _explicit_failure(upper: str, values: dict[str, str]) -> bool:
    if _soft_non_gating_history_message(upper, values):
        return False
    if any(token in upper for token in HARD_ERRORS):
        return True
    if any(token in upper for token in EXPECTED_REJECTIONS):
        return False
    if re.search(r"\b[A-Z0-9_]+_(?:ERROR|FAILED|FAILURE)\b", upper):
        return True
    for key in ("error", "failure_reason"):
        if key in values and values[key].strip().lower() not in NULLS:
            return True
    return False


def parse_event(line: str) -> dict[str, str] | None:
    stamp = STAMP.search(line)
    if not stamp:
        return None
    message = SECRET.sub(r"\1=[REDACTED]", line[stamp.end():].strip())
    upper = message.upper()
    values = fields(message)
    history_diagnostic = _history_diagnostic(upper)
    if _explicit_failure(upper, values):
        kind = "ERROR"
    else:
        if any(token in upper for token in BENIGN) and not history_diagnostic:
            return None
        if not history_diagnostic and not any(token in upper for token in EVENT_WORDS):
            return None
        if "WARN" in upper or "DEGRADED" in upper:
            kind = "WARNING"
        elif history_diagnostic or any(token in upper for token in NON_TRADE_SYSTEM_WORDS):
            kind = "SYSTEM"
        elif any(token in upper for token in TRADE_WORDS):
            kind = "TRADE"
        elif "SIGNAL" in upper or "CANDIDATE_REJECTED" in upper:
            kind = "SIGNAL"
        elif "RISK" in upper or "COOLDOWN" in upper:
            kind = "RISK"
        else:
            kind = "SYSTEM"
    row = {"timestamp_ist": stamp.group(1), "type": kind, "message": message}
    row.update(_detail_fields(message, values))
    return row


def _terminal_key(row: dict[str, str]) -> tuple[str, str] | None:
    message = row.get("message", "")
    if "SIGNAL_EXECUTION_RESULT" not in message:
        return None
    trace = TRACE.search(message)
    if not trace:
        return None
    outcome = RESULT.search(message)
    return trace.group(1), outcome.group(0) if outcome else message


def deduplicate_events(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    terminal_seen: set[tuple[str, str]] = set()
    for row in rows:
        identity = (row.get("timestamp_ist", ""), row.get("type", ""), row.get("message", ""))
        if identity in seen:
            continue
        seen.add(identity)
        terminal = _terminal_key(row)
        if terminal and terminal in terminal_seen:
            continue
        if terminal:
            terminal_seen.add(terminal)
        output.append(row)
    return output


def filter_events(rows: list[dict[str, str]], event_type: str, query: str) -> list[dict[str, str]]:
    result = rows
    if event_type != "ALL":
        result = [row for row in result if row.get("type") == event_type]
    needle = query.strip().lower()
    if needle:
        result = [row for row in result if needle in row.get("message", "").lower()]
    return result


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    target = io.StringIO()
    writer = csv.DictWriter(target, fieldnames=_fieldnames(rows), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")
