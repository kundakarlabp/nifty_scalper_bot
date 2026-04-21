from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.utils.alerts import AlertDeduplicator


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
