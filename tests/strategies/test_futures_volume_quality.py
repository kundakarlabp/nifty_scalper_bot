"""Completed FUT 1-minute volume provenance must never use quote deltas."""

from nifty_scalper_bot.strategies.futures_volume_quality import (
    analyze_completed_futures_volume,
)


def _bar(minute: int, volume: float, *, completed: bool = True) -> dict:
    return {
        "timestamp": f"2026-11-17T09:{minute:02d}:00+05:30",
        "volume": volume,
        "is_complete": completed,
    }


def test_completed_volume_uses_prior_session_bars_only() -> None:
    bars = [_bar(minute, 100) for minute in range(15, 21)]
    bars.append(_bar(21, 250))
    bars.append(_bar(22, 900, completed=False))
    report = analyze_completed_futures_volume(bars)
    assert report["ready"] is True
    assert report["ratio"] == 2.5
    assert report["latest_volume"] == 250
    assert report["baseline_count"] == 6


def test_replayed_bar_does_not_inflate_history_and_day_isolated() -> None:
    bars = [_bar(minute, 10) for minute in range(15, 21)]
    bars.append(_bar(20, 100))
    bars.append(_bar(21, 20))
    report = analyze_completed_futures_volume(bars)
    assert report["baseline_count"] == 6
    assert report["ratio"] < 2


def test_no_completed_bars_is_unknown_not_zero() -> None:
    report = analyze_completed_futures_volume([_bar(15, 100, completed=False)])
    assert report["ready"] is False
    assert report["ratio"] is None


def test_sparse_and_bad_volume_must_not_be_called_ready() -> None:
    bars = [_bar(15, 100), _bar(16, 0), _bar(17, 0), _bar(18, 200)]
    assert analyze_completed_futures_volume(bars)["ready"] is False


def test_extreme_low_completed_volume_is_diagnostic_not_a_synthetic_vote() -> None:
    bars = [_bar(minute, 100) for minute in range(15, 21)]
    bars.append(_bar(21, 1))
    report = analyze_completed_futures_volume(bars)
    assert report["ready"] is True  # Data provenance is valid, activity is low.
    assert report["ratio"] == 0.01
    assert report["latest_volume"] == 1
    assert report["baseline_mean"] == 100
    assert report["baseline_min"] == 100
    assert report["baseline_max"] == 100
    assert report["extreme_low_relative_volume"] is True
    assert report["reason"] == "verified"


def test_normal_completed_volume_does_not_flag_extreme_low() -> None:
    bars = [_bar(minute, 100) for minute in range(15, 21)]
    bars.append(_bar(21, 75))
    report = analyze_completed_futures_volume(bars)
    assert report["ready"] is True
    assert report["ratio"] == 0.75
    assert report["extreme_low_relative_volume"] is False
