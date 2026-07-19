from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from nifty_scalper_bot.core.history_quality import (
    coerce_history_timestamp,
    evaluate_history_quality,
    expected_latest_closed_market_minute,
)

IST = ZoneInfo("Asia/Kolkata")


def trading(day: date) -> bool:
    return day.weekday() < 5 and day != date(2026, 1, 26)


def test_coerce_history_timestamp_units_and_shapes() -> None:
    expected = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    seconds = int(expected.timestamp())
    assert coerce_history_timestamp(seconds) == expected
    assert coerce_history_timestamp(seconds * 1000) == expected
    assert coerce_history_timestamp(seconds * 1_000_000) == expected
    assert coerce_history_timestamp(expected.isoformat()) == expected
    assert coerce_history_timestamp(expected) == expected
    assert coerce_history_timestamp(datetime(2026, 1, 2, 8, 34)) == expected


def test_coerce_history_timestamp_rejects_invalid_values() -> None:
    assert coerce_history_timestamp("not-a-time") is None
    assert coerce_history_timestamp(float("nan")) is None
    assert coerce_history_timestamp(float("inf")) is None
    assert coerce_history_timestamp(datetime(1999, 1, 1, tzinfo=timezone.utc)) is None
    assert coerce_history_timestamp(datetime(2099, 1, 1, tzinfo=timezone.utc)) is None


def test_expected_latest_closed_minute_obeys_publication_grace() -> None:
    now = datetime(2026, 1, 2, 9, 18, 0, tzinfo=IST)
    assert expected_latest_closed_market_minute(
        now=now,
        market_timezone=IST,
        market_open=time(9, 15),
        market_close=time(15, 30),
        trading_day_resolver=trading,
        publication_grace_seconds=90,
    ) == datetime(2026, 1, 2, 3, 45, tzinfo=timezone.utc)


def test_quality_detects_stale_and_gap_for_selected_option() -> None:
    now = datetime(2026, 1, 2, 9, 25, tzinfo=IST)
    bars = [
        {"timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=IST)},
        {"timestamp": datetime(2026, 1, 2, 9, 17, tzinfo=IST)},
    ]
    result = evaluate_history_quality(
        symbol="NFO:XCE",
        role="selected_ce",
        bars=bars,
        required_bars=2,
        now=now,
        market_timezone=IST,
        trading_day_resolver=trading,
        publication_grace_seconds=0,
        allowed_recent_missing_minutes=0,
        continuity_window_bars=2,
    )
    assert result.latest_bar_fresh is False
    assert result.recent_window_contiguous is False
    assert "selected_ce_history_stale" in result.blocker_reasons
    assert "selected_ce_history_gap_detected" in result.blocker_reasons
    assert result.missing_expected_minute_count == 1


def test_quality_ignores_overnight_and_weekend_boundaries() -> None:
    bars = [
        {"timestamp": datetime(2026, 1, 2, 15, 30, tzinfo=IST)},  # Friday
        {"timestamp": datetime(2026, 1, 5, 9, 15, tzinfo=IST)},  # Monday
    ]
    result = evaluate_history_quality(
        symbol="NSE:NIFTY",
        role="spot",
        bars=bars,
        required_bars=2,
        now=datetime(2026, 1, 5, 9, 16, tzinfo=IST),
        market_timezone=IST,
        trading_day_resolver=trading,
        publication_grace_seconds=90,
        allowed_recent_missing_minutes=0,
        continuity_window_bars=2,
    )
    assert result.recent_window_contiguous is True
    assert result.missing_expected_minute_count == 0
