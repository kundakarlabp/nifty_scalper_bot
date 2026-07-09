"""Bounded, read-only systemd journal event transport for the console."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque
from typing import Iterable

STAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} IST)\]")
SECRET = re.compile(r"(?i)(api[_ -]?key|api[_ -]?secret|access[_ -]?token|request[_ -]?token|password)\s*[:=]\s*\S+")
TRACE = re.compile(r"\btrace_id=([^\s,}]+)")
RESULT = re.compile(r"\baccepted=([^\s]+).*?\breason=([^\s]+)")
FIELD = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]*)=([^\s,}]+)")
HISTORY_DIAGNOSTICS = ("CANONICAL_HISTORY_RESULT", "RUNNER_HISTORY_SYNC_RESULT")
EVENTS = ("SIGNAL","ORDER","FILLED","ENTRY","EXIT","TARGET","STOP","PNL","POSITION",
          "BROKER","AUTH","READY","BLOCKER","RISK","COOLDOWN","ERROR","FAIL","WARN",
          "DEGRADED","OPERATIONAL","STARTUP","SHUTDOWN","READINESS","CANDLE")
TRADE_MARKERS = ("ORDER_SENT", "ORDER_PLACED", "ORDER_FILLED", "FILLED", "ENTRY", "EXIT", "TARGET_HIT", "STOP_HIT", "STOP_LOSS", "PNL", "POSITION_OPENED", "POSITION_CLOSED", "TRADE_ATTEMPT")
NON_TRADE_SYSTEM_MARKERS = ("READINESS", "BLOCKER", "CANDLE", "HEARTBEAT", "SUMMARY", "SELECTED_OPTION_SUBSCRIPTION_STATE", "RUNNER_EVAL_DECISION", "NO_TRADE", *HISTORY_DIAGNOSTICS)
IGNORE = ("HEARTBEAT","POLLING","INDICATOR","NO SIGNAL","MARKET CLOSED","OUTSIDE SESSION")
EXPECTED_REJECTIONS = ("CANDIDATE_REJECTED", "SIGNAL_REJECTED", "SIGNAL_EXECUTION_RESULT", "ORDER_READINESS_REJECTED")
HARD_ERRORS = ("TRACEBACK", "CRITICAL", "UNHANDLED EXCEPTION", "RUNNER_ON_TICK_ERROR", "ORDER_FAILED", "STARTUP_FAILED", "HANDLER CRASHED", "FATAL")
NULLS = {"", "none", "null", "nil", "false", "0", "unknown", "n/a", "na"}
SOFT_HISTORY_ROLES = {"option_context"}
SOFT_HISTORY_FAILURE_REASONS = {"broker_fetch_not_allowed"}


def fields(message: str) -> dict[str, str]:
    return {key.lower(): value for key, value in FIELD.findall(message)}


def _history_diagnostic(upper: str) -> bool:
    return any(token in upper for token in HISTORY_DIAGNOSTICS)


def _soft_non_gating_history_message(upper: str, values: dict[str, str]) -> bool:
    """Return True for expected non-selected option-context history misses.

    The runtime may suppress broker history fetches for non-gating option-context
    strikes, especially outside market hours. Those diagnostics must remain
    visible as SYSTEM events, not ERROR rows, while selected-option failures
    continue to surface as hard blockers.
    """

    role = values.get("role", "").strip().lower()
    failure = values.get("failure_reason", "").strip().lower()
    return (
        "CANONICAL_HISTORY_RESULT" in upper
        and role in SOFT_HISTORY_ROLES
        and failure in SOFT_HISTORY_FAILURE_REASONS
    )


def _explicit_failure(upper: str, values: dict[str, str], *, expected_rejection: bool) -> bool:
    if _soft_non_gating_history_message(upper, values):
        return False
    if any(token in upper for token in HARD_ERRORS):
        return True
    if expected_rejection:
        return False
    if re.search(r"\b[A-Z0-9_]+_(?:ERROR|FAILED|FAILURE)\b", upper):
        return True
    for key in ("error", "failure_reason"):
        if key in values and values[key].strip().lower() not in NULLS:
            return True
    return False


def parse_event(line: str) -> dict[str, str] | None:
    match = STAMP.search(line)
    if not match:
        return None
    message = SECRET.sub(r"\1=[REDACTED]", line[match.end():].strip())
    upper = message.upper()
    values = fields(message)
    expected_rejection = any(x in upper for x in EXPECTED_REJECTIONS)
    soft_non_gating_history = _soft_non_gating_history_message(upper, values)
    history_diagnostic = _history_diagnostic(upper)
    if any(x in upper for x in IGNORE) and not history_diagnostic:
        return None
    if not history_diagnostic and not expected_rejection and not any(x in upper for x in EVENTS):
        return None
    if _explicit_failure(upper, values, expected_rejection=expected_rejection):
        kind = "ERROR"
    elif any(x in upper for x in ("WARN", "DEGRADED")):
        kind = "WARNING"
    elif history_diagnostic or any(x in upper for x in NON_TRADE_SYSTEM_MARKERS):
        kind = "SYSTEM"
    elif any(x in upper for x in TRADE_MARKERS):
        kind = "TRADE"
    elif "SIGNAL" in upper or "CANDIDATE_REJECTED" in upper:
        kind = "SIGNAL"
    elif any(x in upper for x in ("RISK","COOLDOWN")):
        kind = "RISK"
    else:
        kind = "SYSTEM"
    return {"timestamp_ist": match.group(1), "type": kind, "message": message}


def terminal_event_key(row: dict[str, str]) -> tuple[str, str] | None:
    message = row.get("message", "")
    if "SIGNAL_EXECUTION_RESULT" not in message:
        return None
    trace = TRACE.search(message)
    if trace is None:
        return None
    outcome = RESULT.search(message)
    return trace.group(1), outcome.group(0) if outcome else message


def deduplicate_events(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = terminal_event_key(row)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        output.append(row)
    return output


class EventRing:
    def __init__(self, service: str, max_events: int = 3000):
        try:
            configured = int(os.getenv("BOT_EVENT_BUFFER_MAX", str(max_events)) or max_events)
        except (TypeError, ValueError):
            configured = max_events
        self.capacity = max(250, min(configured, 5000))
        self.service = service
        self.rows = deque(maxlen=self.capacity)
        self.lock = threading.RLock()
        self.connected = False
        self.last_event = 0.0
        self.restarts = 0
        self.last_error: str | None = None
        self._terminal_seen: set[tuple[str, str]] = set()
        self._terminal_order: deque[tuple[str, str]] = deque()
        threading.Thread(target=self._run, daemon=True, name="journal-event-tail").start()

    def _run(self) -> None:
        while True:
            proc = None
            try:
                proc = subprocess.Popen(
                    ["journalctl","-u",self.service,"-n","1200","-f","--no-pager","-o","cat"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
                with self.lock:
                    self.connected = True
                    self.last_error = None
                for line in proc.stdout or ():
                    event = parse_event(line)
                    if event:
                        with self.lock:
                            key = terminal_event_key(event)
                            if key is not None and key in self._terminal_seen:
                                continue
                            if key is not None:
                                if len(self._terminal_order) >= self.capacity:
                                    expired = self._terminal_order.popleft()
                                    self._terminal_seen.discard(expired)
                                self._terminal_seen.add(key)
                                self._terminal_order.append(key)
                            if not self.rows or self.rows[-1] != event:
                                self.rows.append(event)
                                self.last_event = time.time()
            except Exception as exc:
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self.lock:
                    self.connected = False
                    self.restarts += 1
                if proc and proc.poll() is None:
                    proc.terminate()
                time.sleep(1.5)

    def snapshot(self) -> list[dict[str, str]]:
        with self.lock:
            return list(self.rows)

    def stats(self) -> dict[str, object]:
        with self.lock:
            return {"connected": self.connected, "last_event": self.last_event,
                    "restarts": self.restarts, "size": len(self.rows),
                    "capacity": self.capacity, "last_error": self.last_error}
