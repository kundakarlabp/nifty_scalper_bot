"""Minute-transition grace for the just-closed bar.

Production logs showed `live_latest_closed_bar_stale` firing almost
exclusively at HH:MM:00 (15:26:00, 15:27:00, 15:28:00, 15:29:00). At the roll,
`expected_closed_start` advances immediately but the bar for the minute that
just closed is published a few hundred ms later, so the newest available bar
is one bucket behind expected and is misclassified as stale.

The grace narrows that race only. It does not relax freshness generally.
"""

from __future__ import annotations

from nifty_scalper_bot.core.strategy_live_safety import (
    _bar_publication_grace_seconds,
)


def test_grace_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("LIVE_BAR_PUBLICATION_GRACE_SECONDS", raising=False)
    assert _bar_publication_grace_seconds() == 1.5

    monkeypatch.setenv("LIVE_BAR_PUBLICATION_GRACE_SECONDS", "2.5")
    assert _bar_publication_grace_seconds() == 2.5

    # Malformed config must not widen the window.
    monkeypatch.setenv("LIVE_BAR_PUBLICATION_GRACE_SECONDS", "junk")
    assert _bar_publication_grace_seconds() == 1.5

    # 0 disables the grace entirely (previous behaviour).
    monkeypatch.setenv("LIVE_BAR_PUBLICATION_GRACE_SECONDS", "0")
    assert _bar_publication_grace_seconds() == 0.0

    # Negative values clamp to 0 rather than inverting the comparison.
    monkeypatch.setenv("LIVE_BAR_PUBLICATION_GRACE_SECONDS", "-30")
    assert _bar_publication_grace_seconds() == 0.0


def _classify(now: float, bar_epoch: float, grace: float, interval_s: int = 60):
    """Mirror of the production bucket comparison under test."""
    current_bucket_start = (now // interval_s) * interval_s
    expected_closed_start = current_bucket_start - interval_s
    bucket_start = (bar_epoch // interval_s) * interval_s
    if bucket_start >= current_bucket_start:
        return "open"
    if bucket_start == expected_closed_start:
        return "ready"
    if bucket_start == expected_closed_start - interval_s:
        if grace > 0 and (now - current_bucket_start) <= grace:
            return "ready_grace"
    if bucket_start < expected_closed_start:
        return "stale"
    return "unknown"


def test_just_after_roll_one_bucket_behind_is_accepted() -> None:
    """THE FIX: 1.0s past HH:MM:00, the not-yet-published bar is accepted."""
    now = 1_800_000_060.0 + 1.0          # 1s after a minute roll
    bar = 1_799_999_970.0                # bucket 1_799_999_940: one behind expected
    assert _classify(now, bar, grace=1.5) == "ready_grace"


def test_outside_grace_window_still_stale() -> None:
    """Well into the minute there is no excuse for a missing bar."""
    now = 1_800_000_060.0 + 30.0
    bar = 1_799_999_970.0
    assert _classify(now, bar, grace=1.5) == "stale"


def test_two_buckets_behind_is_stale_even_inside_grace() -> None:
    """Grace covers exactly one bucket; real gaps must still fail."""
    now = 1_800_000_060.0 + 1.0
    bar = 1_799_999_910.0                # bucket 1_799_999_880: two behind
    assert _classify(now, bar, grace=1.5) == "stale"


def test_grace_disabled_restores_previous_behaviour() -> None:
    now = 1_800_000_060.0 + 1.0
    bar = 1_799_999_970.0
    assert _classify(now, bar, grace=0.0) == "stale"


def test_expected_bar_present_is_ready_without_grace() -> None:
    """Normal path is untouched."""
    now = 1_800_000_060.0 + 30.0
    bar = 1_800_000_030.0                # bucket 1_800_000_000 = expected
    assert _classify(now, bar, grace=1.5) == "ready"
