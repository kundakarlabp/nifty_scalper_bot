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
EVENTS = ("SIGNAL","ORDER","FILLED","ENTRY","EXIT","TARGET","STOP","PNL","POSITION",
          "BROKER","AUTH","READY","BLOCKER","RISK","COOLDOWN","ERROR","FAIL","WARN",
          "DEGRADED","OPERATIONAL","STARTUP","SHUTDOWN")
IGNORE = ("HEARTBEAT","POLLING","INDICATOR","NO SIGNAL","MARKET CLOSED","OUTSIDE SESSION")
EXPECTED_REJECTIONS = ("CANDIDATE_REJECTED", "SIGNAL_REJECTED", "SIGNAL_EXECUTION_RESULT", "ORDER_READINESS_REJECTED")
HARD_ERRORS = ("TRACEBACK", "CRITICAL", "UNHANDLED EXCEPTION", "RUNNER_ON_TICK_ERROR", "ORDER_FAILED", "STARTUP_FAILED", "HANDLER CRASHED", "FATAL")


def parse_event(line: str) -> dict[str, str] | None:
    match = STAMP.search(line)
    if not match:
        return None
    message = SECRET.sub(r"\1=[REDACTED]", line[match.end():].strip())
    upper = message.upper()
    if any(x in upper for x in IGNORE) or not any(x in upper for x in EVENTS):
        return None
    expected_rejection = any(x in upper for x in EXPECTED_REJECTIONS)
    if any(x in upper for x in HARD_ERRORS) or (("ERROR" in upper or "FAILED" in upper or "FAILURE" in upper) and not expected_rejection):
        kind = "ERROR"
    elif any(x in upper for x in ("WARN", "DEGRADED")):
        kind = "WARNING"
    elif any(x in upper for x in ("ORDER","FILLED","ENTRY","EXIT","TARGET","STOP","PNL","POSITION")):
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
                                self._terminal_seen.add(key)
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
