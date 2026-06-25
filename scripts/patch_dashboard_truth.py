#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str, *, sentinel: str | None = None) -> None:
    value = read(path)
    if sentinel and sentinel in value:
        return
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    write(path, value.replace(old, new, 1))


def patch_event_buffer() -> None:
    write(
        "dashboard/event_buffer.py",
        '''"""Bounded, read-only systemd journal event transport for the console."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque
from typing import Iterable

STAMP = re.compile(r"\\[(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} IST)\\]")
SECRET = re.compile(r"(?i)(api[_ -]?key|api[_ -]?secret|access[_ -]?token|request[_ -]?token|password)\\s*[:=]\\s*\\S+")
TRACE = re.compile(r"\\btrace_id=([^\\s,}]+)")
RESULT = re.compile(r"\\baccepted=([^\\s]+).*?\\breason=([^\\s]+)")
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
    message = SECRET.sub(r"\\1=[REDACTED]", line[match.end():].strip())
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
''',
    )


def patch_console() -> None:
    path = "dashboard/operations_console.py"
    replace_once(path, "from event_buffer import EventRing, parse_event\n", "from event_buffer import EventRing, deduplicate_events, parse_event\n")
    replace_once(
        path,
        "        response = http_session().get(API + path, timeout=1.2)\n        response.raise_for_status()\n        value = response.json()\n        return value if isinstance(value, dict) else None\n    except Exception:\n        return None\n",
        "        response = http_session().get(API + path, timeout=1.2)\n        value = response.json()\n        if not isinstance(value, dict):\n            return None\n        value = dict(value)\n        value['_http_status'] = response.status_code\n        return value\n    except (requests.RequestException, ValueError):\n        return None\n",
        sentinel="value['_http_status'] = response.status_code",
    )
    replace_once(
        path,
        "    return [\n        event\n        for line in result.stdout.splitlines()\n        if (event := parse_event(line))\n    ], None\n",
        "    rows = [\n        event\n        for line in result.stdout.splitlines()\n        if (event := parse_event(line))\n    ]\n    return deduplicate_events(rows), None\n",
    )
    replace_once(
        path,
        "    broker_ready = bool(broker.get(\"ready\"))\n    auth_invalid = bool(broker.get(\"auth_invalid\"))\n    reconciled = bool(recon.get(\"completed\"))\n",
        "    trading_available = trading is not None\n    broker_available = trading_available and bool(broker)\n    recon_available = trading_available and bool(recon)\n    broker_ready = bool(broker.get(\"ready\")) if broker_available else None\n    auth_invalid = bool(broker.get(\"auth_invalid\")) if broker_available else None\n    reconciled = bool(recon.get(\"completed\")) if recon_available else None\n",
        sentinel="broker_available = trading_available",
    )
    replace_once(
        path,
        "        + state_item(\"Broker\", \"READY\" if broker_ready else \"NOT READY\", \"good\" if broker_ready else \"bad-text\")\n        + state_item(\"Balance\", short_value(broker.get(\"balance\")))\n        + state_item(\"Reconciled\", \"YES\" if reconciled else \"NO\", \"good\" if reconciled else \"warn-text\")\n        + state_item(\"Authentication\", \"INVALID\" if auth_invalid else \"OK\", \"bad-text\" if auth_invalid else \"good\")\n",
        "        + state_item(\"Broker\", \"UNKNOWN\" if broker_ready is None else (\"READY\" if broker_ready else \"NOT READY\"), \"warn-text\" if broker_ready is None else (\"good\" if broker_ready else \"bad-text\"))\n        + state_item(\"Balance\", short_value(broker.get(\"balance\")) if broker_available else \"UNKNOWN\")\n        + state_item(\"Reconciled\", \"UNKNOWN\" if reconciled is None else (\"YES\" if reconciled else \"NO\"), \"warn-text\" if reconciled is None else (\"good\" if reconciled else \"warn-text\"))\n        + state_item(\"Authentication\", \"UNKNOWN\" if auth_invalid is None else (\"INVALID\" if auth_invalid else \"OK\"), \"warn-text\" if auth_invalid is None else (\"bad-text\" if auth_invalid else \"good\"))\n",
        sentinel='state_item("Authentication", "UNKNOWN"',
    )
    replace_once(
        path,
        "        '<div class=\"status-card\"><div class=\"card-title\">Deployment</div>'\n",
        "        '<div class=\"status-card\"><div class=\"card-title\">Deployment</div>'\n        f'<div class=\"deploy-row\"><span class=\"deploy-key\">Platform</span><span class=\"deploy-value\">{html.escape(os.getenv(\"DEPLOYMENT_PLATFORM\", \"aws_lightsail\"))}</span></div>'\n",
        sentinel='deploy-key">Platform',
    )
    replace_once(
        path,
        "        f'<span>Buffer: {stats[\"size\"]:,} / 3,000</span>'\n",
        "        f'<span>Buffer: {stats[\"size\"]:,} / {int(stats.get(\"capacity\") or 3000):,}</span>'\n",
    )


def create_tests() -> None:
    write(
        "tests/dashboard/test_event_buffer_truth.py",
        '''from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "event_buffer.py"
SPEC = importlib.util.spec_from_file_location("dashboard_event_buffer_truth", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_expected_candidate_rejection_is_not_process_error():
    event = MODULE.parse_event("[2026-06-25 15:00:00 IST] CANDIDATE_REJECTED symbol=X reason=tick_stale")
    assert event is not None
    assert event["type"] == "SIGNAL"


def test_actual_runner_error_remains_error():
    event = MODULE.parse_event("[2026-06-25 15:00:00 IST] RUNNER_ON_TICK_ERROR symbol=X error=boom")
    assert event is not None
    assert event["type"] == "ERROR"


def test_duplicate_terminal_result_is_removed():
    rows = [
        {"timestamp_ist": "2026-06-25 15:00:00 IST", "type": "SIGNAL", "message": "SIGNAL_EXECUTION_RESULT accepted=False reason=no_execution_ready_candidate trace_id=t1"},
        {"timestamp_ist": "2026-06-25 15:00:01 IST", "type": "TRADE", "message": "SIGNAL_EXECUTION_RESULT accepted=False reason=no_execution_ready_candidate trace_id=t1"},
    ]
    assert len(MODULE.deduplicate_events(rows)) == 1
''',
    )


if __name__ == "__main__":
    patch_event_buffer()
    patch_console()
    create_tests()
