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
