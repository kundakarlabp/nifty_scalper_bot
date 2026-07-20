from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.core.history_readiness import (
    _coerce_history_timestamp_utc,
    _evaluate_recent_history_quality,
    _expected_latest_bar_start_utc,
    build_symbol_hydration_status,
)

IST = ZoneInfo("Asia/Kolkata")


def trading(day: date) -> bool:
    return day.weekday() < 5 and day != date(2026, 1, 26)


def test_coerce_history_timestamp_units_and_shapes() -> None:
    expected = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    seconds = int(expected.timestamp())
    assert (
        _coerce_history_timestamp_utc(
            seconds, now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        == expected
    )
    assert (
        _coerce_history_timestamp_utc(
            seconds * 1000, now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        == expected
    )
    assert (
        _coerce_history_timestamp_utc(
            seconds * 1_000_000, now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        == expected
    )
    assert (
        _coerce_history_timestamp_utc(
            expected.isoformat(),
            now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc),
        )
        == expected
    )
    assert (
        _coerce_history_timestamp_utc(
            expected, now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        == expected
    )
    assert (
        _coerce_history_timestamp_utc(
            datetime(2026, 1, 2, 8, 34),
            now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc),
        )
        == expected
    )


def test_coerce_history_timestamp_rejects_invalid_values() -> None:
    assert (
        _coerce_history_timestamp_utc(
            "not-a-time", now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        is None
    )
    assert (
        _coerce_history_timestamp_utc(
            float("nan"), now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        is None
    )
    assert (
        _coerce_history_timestamp_utc(
            float("inf"), now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        )
        is None
    )
    assert (
        _coerce_history_timestamp_utc(
            datetime(1999, 1, 1, tzinfo=timezone.utc),
            now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc),
        )
        is None
    )
    assert (
        _coerce_history_timestamp_utc(
            datetime(2099, 1, 1, tzinfo=timezone.utc),
            now_utc=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc),
        )
        is None
    )


def test_expected_latest_closed_minute_obeys_publication_grace() -> None:
    now = datetime(2026, 1, 2, 9, 18, 0, tzinfo=IST)
    assert _expected_latest_bar_start_utc(
        now.astimezone(timezone.utc), publication_grace_seconds=90
    ) == datetime(2026, 1, 2, 3, 45, tzinfo=timezone.utc)


def test_quality_detects_stale_and_gap_for_selected_option() -> None:
    now = datetime(2026, 1, 2, 9, 25, tzinfo=IST)
    bars = [
        {"timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=IST)},
        {"timestamp": datetime(2026, 1, 2, 9, 17, tzinfo=IST)},
    ]
    result = _evaluate_recent_history_quality(
        bars,
        role="selected_ce",
        required_bars=2,
        now_utc=now.astimezone(timezone.utc),
        max_lag_minutes=0,
        publication_grace_seconds=0,
        allowed_missing_minutes=0,
        continuity_window_bars=2,
        provider_error=None,
    )
    assert result.latest_bar_fresh is False
    assert result.recent_window_contiguous is False
    assert "selected_ce_history_stale" in result.blockers
    assert "selected_ce_history_gap_detected" in result.blockers
    assert result.missing_minute_count == 1


def test_quality_ignores_overnight_and_weekend_boundaries() -> None:
    bars = [
        {"timestamp": datetime(2026, 1, 2, 15, 30, tzinfo=IST)},  # Friday
        {"timestamp": datetime(2026, 1, 5, 9, 15, tzinfo=IST)},  # Monday
    ]
    result = _evaluate_recent_history_quality(
        bars,
        role="spot",
        required_bars=2,
        now_utc=datetime(2026, 1, 5, 9, 16, tzinfo=IST).astimezone(timezone.utc),
        max_lag_minutes=2,
        publication_grace_seconds=90,
        allowed_missing_minutes=0,
        continuity_window_bars=2,
        provider_error=None,
    )
    assert result.recent_window_contiguous is True
    assert result.missing_minute_count == 0


def test_expected_latest_bar_session_edges_return_utc_1529_final() -> None:
    cases = [
        (
            datetime(2026, 1, 2, 9, 14, tzinfo=IST),
            datetime(2026, 1, 1, 9, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 1, 2, 9, 15, tzinfo=IST),
            datetime(2026, 1, 1, 9, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 1, 2, 9, 17, tzinfo=IST),
            datetime(2026, 1, 2, 3, 46, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 1, 2, 15, 30, tzinfo=IST),
            datetime(2026, 1, 2, 9, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 1, 3, 12, 0, tzinfo=IST),
            datetime(2026, 1, 2, 9, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 1, 26, 12, 0, tzinfo=IST),
            datetime(2026, 1, 23, 9, 59, tzinfo=timezone.utc),
        ),
    ]
    for now_ist, expected in cases:
        got = _expected_latest_bar_start_utc(
            now_ist.astimezone(timezone.utc), publication_grace_seconds=0
        )
        assert got == expected
        assert got.tzinfo is timezone.utc


def test_timestamp_coercion_offsets_nanoseconds_rejects_bool_naive_now() -> None:
    now = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)
    expected = datetime(2026, 1, 2, 3, 45, tzinfo=timezone.utc)
    assert (
        _coerce_history_timestamp_utc("2026-01-02T09:15:00+05:30", now_utc=now)
        == expected
    )
    assert (
        _coerce_history_timestamp_utc(
            int(expected.timestamp() * 1_000_000_000), now_utc=now
        )
        == expected
    )
    assert _coerce_history_timestamp_utc(True, now_utc=now) is None
    with pytest.raises(ValueError):
        _coerce_history_timestamp_utc(expected, now_utc=datetime(2026, 1, 2))
    with pytest.raises(ValueError):
        _expected_latest_bar_start_utc(
            datetime(2026, 1, 2), publication_grace_seconds=0
        )


def test_build_symbol_hydration_status_rejects_naive_injected_now() -> None:
    ctx = SimpleNamespace(
        market_data_manager=SimpleNamespace(get_ohlc_bars=lambda *_a, **_k: []),
        data_hub=None,
        strategy_runner=None,
    )
    with pytest.raises(ValueError):
        build_symbol_hydration_status(
            ctx, "NSE:NIFTY", "spot", 1, now_utc=datetime(2026, 1, 2)
        )


def test_option_context_gap_is_diagnostic_not_role_specific_execution_blocker() -> None:
    bars = [
        {"timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=IST)},
        {"timestamp": datetime(2026, 1, 2, 9, 17, tzinfo=IST)},
    ]
    result = _evaluate_recent_history_quality(
        bars,
        role="option_context",
        required_bars=2,
        now_utc=datetime(2026, 1, 2, 9, 18, tzinfo=IST).astimezone(timezone.utc),
        max_lag_minutes=2,
        publication_grace_seconds=0,
        allowed_missing_minutes=0,
        continuity_window_bars=2,
        provider_error=None,
    )
    assert result.recent_window_contiguous is False
    assert "selected_ce_history_gap_detected" not in result.blockers
    assert "option_context_history_gap_detected" in result.blockers
