"""Purpose: Summarize observed market-session startup and recovery log events.

Responsibilities:
- audit_session_startups counts canonical WebSocket, reconciliation and ready events.
- Extract clock times, first-ready/opening-window observations and bounded counts.
- Mark missing readiness markers explicitly; do not infer continuous availability.
- Mark absent evidence explicitly instead of inferring broker or stop safety.

Boundaries: Offline/read-only log parsing; no broker, engine or scheduler imports.
Safety: This audit is not an entry gate, protective-stop proof or restart trigger.
"""

from __future__ import annotations

import re
from datetime import datetime

_TS = re.compile(r"\[(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) IST\]")
_START = "STARTUP_PHASE phase=create_context"
_WS = "STARTUP_WS_START_DONE"
_RECONCILE = "RECONCILE_COMPLETE"
_LIVE_READY = "STARTUP_MODE_TRANSITION"
_ORDER_OK = "BROKER_ORDER_RECONCILE_OK"
# Event counts are retrospective diagnostics, never continuous stop coverage.
_STOP_EVENTS = {
    "EXCHANGE_PROTECTIVE_STOP_ARMED": "stop_armed_event_count",
    "EXCHANGE_PROTECTIVE_STOP_UNVERIFIED": "stop_unverified_event_count",
    "EXCHANGE_PROTECTIVE_STOP_NOT_ARMED": "stop_not_armed_event_count",
    "EXCHANGE_PROTECTIVE_STOP_RATCHET_CONFIRMED": "stop_ratchet_confirmed_event_count",
}


def _event_time(line: str) -> datetime | None:
    match = _TS.search(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def audit_session_startups(log_text: str) -> dict[str, int | str | None]:
    """Compute a bounded observation-only summary of a single session archive."""
    starts: list[datetime] = []
    readiness: list[datetime] = []
    ws_count = 0
    reconciled_count = 0
    order_ok_count = 0
    stop_counts = dict.fromkeys(_STOP_EVENTS.values(), 0)

    for line in log_text.splitlines():
        for event, key in _STOP_EVENTS.items():
            if event in line:
                stop_counts[key] += 1
        if _ORDER_OK in line:
            order_ok_count += 1
        if _WS in line:
            ws_count += 1
        if _RECONCILE in line:
            reconciled_count += 1
        if _START not in line and _LIVE_READY not in line:
            continue
        timestamp = _event_time(line)
        if timestamp is None:
            continue
        if _START in line:
            starts.append(timestamp)
        elif _LIVE_READY in line and "to=LIVE_READY" in line:
            readiness.append(timestamp)

    # Match each observed LIVE_READY to one *unmatched* preceding startup.
    # Ordering is within one archive and unrelated earlier READY events are ignored.
    used: set[int] = set()
    lags: list[int] = []
    for ready in readiness:
        possible = [
            index
            for index, start in enumerate(starts)
            if index not in used and start <= ready
        ]
        if possible:
            index = possible[-1]
            used.add(index)
            lags.append(max(0, int((ready - starts[index]).total_seconds())))

    return {
        "schema_version": 1,
        "observed_startup_count": len(starts),
        "ws_started_count": ws_count,
        "reconcile_complete_count": reconciled_count,
        "live_ready_count": len(readiness),
        "broker_order_reconcile_ok_count": order_ok_count,
        "startups_after_0930_count": sum(
            start.strftime("%H:%M:%S") >= "09:30:00" for start in starts
        ),
        "startups_without_live_ready_count": len(starts) - len(used),
        "observed_repeat_startup_count": max(0, len(starts) - 1),
        "market_window_startup_count": sum(
            "09:30:00" <= start.strftime("%H:%M:%S") <= "15:00:00" for start in starts
        ),
        **stop_counts,
        "first_startup_at_ist": starts[0].strftime("%H:%M:%S") if starts else None,
        "last_startup_at_ist": starts[-1].strftime("%H:%M:%S") if starts else None,
        "first_live_ready_at_ist": (
            min(readiness).strftime("%H:%M:%S") if readiness else None
        ),
        "last_live_ready_at_ist": (
            max(readiness).strftime("%H:%M:%S") if readiness else None
        ),
        # Observed READY transition, not evidence of continuous availability.
        "opening_window_ready_event_observed": any(
            "09:15:00" <= ready.strftime("%H:%M:%S") < "09:30:00"
            for ready in readiness
        ),
        "max_startup_to_ready_seconds": max(lags) if lags else None,
        "protective_stop_verification": "not_established_from_log_markers",
        "restart_cause": "unknown_without_process_and_deployment_events",
    }
