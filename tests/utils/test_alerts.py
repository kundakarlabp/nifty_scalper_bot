from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.utils.alerts import AlertDeduplicator, AlertLogHandler


def test_alert_deduplicator_batches_noncritical() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deduper = AlertDeduplicator(timedelta(seconds=60))

    assert not deduper.should_immediate(
        "log:error", "warning", hint_immediate=False, now=base
    )


def test_alert_deduplicator_respects_hint_immediate() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deduper = AlertDeduplicator(timedelta(seconds=60))

    assert deduper.should_immediate(
        "ops:recovery", "warning", hint_immediate=True, now=base
    )


def test_alert_deduplicator_limits_critical_bursts() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deduper = AlertDeduplicator(
        timedelta(seconds=5), bucket_capacity=1, bucket_refill_seconds=0.01
    )

    assert deduper.should_immediate(
        "svc:outage", "critical", hint_immediate=False, now=base
    )
    # Still within quiet window and token bucket empty -> aggregate
    assert not deduper.should_immediate(
        "svc:outage", "critical", hint_immediate=False, now=base + timedelta(seconds=1)
    )
    # Allow token bucket to refill and quiet window to elapse
    deduper._bucket.tokens = float(deduper._bucket.capacity)  # type: ignore[attr-defined]
    deduper._bucket.last_refill_ts = time.monotonic()  # type: ignore[attr-defined]
    assert deduper.should_immediate(
        "svc:outage", "critical", hint_immediate=False, now=base + timedelta(seconds=7)
    )


def test_alert_deduplicator_suppresses_same_outage_family_during_cooldown() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deduper = AlertDeduplicator(timedelta(seconds=60))

    assert deduper.should_immediate(
        "market_data.starvation:first",
        "critical",
        hint_immediate=True,
        outage_class="market_data.starvation",
        now=base,
    )
    assert not deduper.should_immediate(
        "market_data.starvation:repeat",
        "critical",
        hint_immediate=True,
        outage_class="market_data.starvation",
        now=base + timedelta(seconds=10),
    )


def test_alert_deduplicator_allows_recovery_event_during_same_window() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deduper = AlertDeduplicator(timedelta(seconds=60))

    assert deduper.should_immediate(
        "market_data.empty_quote_map",
        "warning",
        hint_immediate=True,
        outage_class="market_data.empty_quote_map",
        now=base,
    )
    assert deduper.should_immediate(
        "market_data.empty_quote_map",
        "info",
        hint_immediate=True,
        outage_class="market_data.empty_quote_map",
        recovery=True,
        now=base + timedelta(seconds=15),
    )


def test_alert_deduplicator_flood_hold_suppresses_immediate_retries() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deduper = AlertDeduplicator(timedelta(seconds=60))

    deduper.mark_flood_limited("market_data.mapping_empty", now=base)
    assert not deduper.should_immediate(
        "market_data.mapping_empty",
        "critical",
        hint_immediate=True,
        outage_class="market_data.mapping_empty",
        now=base + timedelta(seconds=5),
    )


def _warning_record(message: str, *, func: str = "_watchdog_loop") -> logging.LogRecord:
    return logging.LogRecord(
        name="nifty_scalper_bot.streaming.websocket_manager",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
        func=func,
    )


def test_alert_log_handler_collapses_dynamic_condition_repeats() -> None:
    emitted: list[dict[str, str]] = []
    now = [100.0]
    handler = AlertLogHandler(
        lambda payload: emitted.append(dict(payload)),
        repeat_window_seconds=300.0,
        clock=lambda: now[0],
    )

    handler.emit(
        _warning_record(
            "Condition met: websocket_pong_timeout age=17.22s threshold=10.00s"
        )
    )
    now[0] = 101.0
    handler.emit(
        _warning_record(
            "Condition met: websocket_pong_timeout age=17.29s threshold=10.00s"
        )
    )

    assert len(emitted) == 1
    assert emitted[0]["key"].endswith(":websocket_pong_timeout")
    assert "age=17.22s" in emitted[0]["message"]

    now[0] = 401.0
    handler.emit(
        _warning_record(
            "Condition met: websocket_pong_timeout age=18.01s threshold=10.00s"
        )
    )
    assert len(emitted) == 2


def test_alert_log_handler_keeps_distinct_conditions_from_same_function() -> None:
    emitted: list[dict[str, str]] = []
    handler = AlertLogHandler(
        lambda payload: emitted.append(dict(payload)),
        repeat_window_seconds=300.0,
        clock=lambda: 100.0,
    )

    handler.emit(
        _warning_record(
            "Condition met: websocket_pong_timeout age=17.22s threshold=10.00s"
        )
    )
    handler.emit(
        _warning_record(
            "Condition met: websocket_handshake_timeout age=31.00s threshold=30.00s"
        )
    )

    assert len(emitted) == 2
    assert emitted[0]["key"] != emitted[1]["key"]


def test_alert_log_handler_does_not_hide_generic_warning_repeats() -> None:
    emitted: list[dict[str, str]] = []
    handler = AlertLogHandler(
        lambda payload: emitted.append(dict(payload)),
        repeat_window_seconds=300.0,
        clock=lambda: 100.0,
    )
    record = _warning_record("broker response malformed", func="decode")

    handler.emit(record)
    handler.emit(record)

    assert len(emitted) == 2
    assert emitted[0]["key"].endswith(":decode")
