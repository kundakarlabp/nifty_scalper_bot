"""Tests for read-only option-volume evidence classification."""

from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.core.option_volume_diagnostics import (
    describe_option_volume_evidence,
)


def _bar(volume: float, *, complete: bool = True, provisional: bool = False):
    return {
        "timestamp": datetime(2026, 10, 27, 8, 30, tzinfo=timezone.utc),
        "volume": volume,
        "is_complete": complete,
        "is_provisional": provisional,
    }


def test_finalized_zero_bar_is_unverified_zero_not_proven_no_trade() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0, "history_source": "data_hub"},
        latest_bar=_bar(0.0),
        quote={"volume_cumulative": 1000.0, "volume_delta": 0.0},
    )
    assert out["state"] == "zero_recorded_bar_volume_unverified"
    assert out["history_bar_complete"] is True
    assert out["history_bar_provisional"] is False
    assert out["quote_volume_cumulative"] == 1000.0
    assert out["quote_volume_delta"] == 0.0
    assert out["history_bar_ts"].startswith("2026-10-27T08:30:00")


def test_provisional_tail_does_not_pretend_to_be_finalized() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0},
        latest_bar=_bar(0.0, complete=False, provisional=True),
    )
    assert out["state"] == "unfinalized_history_tail"
    assert out["history_bar_complete"] is False


def test_positive_final_bar_with_zero_indicator_is_mismatch() -> None:
    out = describe_option_volume_evidence({"volume": 0.0}, latest_bar=_bar(145.0))
    assert out["state"] == "indicator_history_volume_mismatch"
    assert out["history_bar_volume"] == 145.0


def test_missing_history_and_unknown_quote_stay_unknown() -> None:
    out = describe_option_volume_evidence({"volume": 0.0})
    assert out["state"] == "history_tail_unavailable"
    assert out["history_bar_volume"] is None
    assert out["quote_volume_delta"] is None
    assert out["quote_volume_cumulative"] is None


def test_missing_volume_is_not_silently_filled_with_zero() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        latest_bar={"timestamp": "2026-10-27T08:30:00+00:00", "is_complete": True},
    )
    assert out["state"] == "history_volume_missing_or_invalid"
    assert out["history_bar_has_volume_field"] is False


def test_quote_transition_captures_untrusted_tick_baseline() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        latest_bar=_bar(0.0),
        quote={
            "volume_traded_today": 400.0,
            "volume_transition": {"state": "baseline_initialized"},
            "volume_delta_untrusted": True,
            "source": "ws",
            "quote_update_version": 14,
        },
    )
    assert out["quote_volume_transition_state"] == "baseline_initialized"
    assert out["quote_volume_delta_untrusted"] is True
    assert out["quote_source"] == "ws"
    assert out["quote_update_version"] == 14


def test_nonfinite_quote_volume_is_not_logged_as_legitimate_volume() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        latest_bar=_bar(0.0),
        quote={"volume_cumulative": float("inf"), "volume_delta": float("nan")},
    )
    assert out["quote_volume_cumulative"] is None
    assert out["quote_volume_delta"] is None

def test_finalized_candle_parity_matches_aware_timestamps_and_volume() -> None:
    from datetime import timedelta

    indicator = _bar(150.0)
    canonical = {
        "timestamp": datetime(
            2026, 10, 27, 14, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
        ),
        "volume": 150.0,
    }
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        latest_bar=_bar(0.0, complete=False, provisional=True),
        indicator_completed_bar=indicator,
        canonical_completed_bar=canonical,
    )
    assert out["state"] == "unfinalized_history_tail"
    assert out["completed_bar_parity"] == "same_minute_volume_match"
    assert out["indicator_completed_ts"] == "2026-10-27T08:30:00+00:00"
    assert out["canonical_completed_ts"] == out["indicator_completed_ts"]


def test_finalized_candle_volume_mismatch_same_minute_is_detected() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        latest_bar=_bar(0.0),
        indicator_completed_bar=_bar(0.0),
        canonical_completed_bar=_bar(125.0),
    )
    assert out["completed_bar_parity"] == "same_minute_volume_mismatch"
    assert out["indicator_completed_volume"] == 0.0
    assert out["canonical_completed_volume"] == 125.0


def test_mismatched_minutes_are_not_a_volume_mismatch() -> None:
    old = _bar(0.0)
    recent = {
        **_bar(125.0),
        "timestamp": datetime(2026, 10, 27, 8, 31, tzinfo=timezone.utc),
    }
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        indicator_completed_bar=old,
        canonical_completed_bar=recent,
    )
    assert out["completed_bar_parity"] == "different_completed_minutes"


def test_missing_completed_candle_is_indeterminate_not_match() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        latest_bar=_bar(0.0),
        canonical_completed_bar=_bar(0.0),
    )
    assert out["completed_bar_parity"] == "missing_or_invalid_completed_timestamp"
    assert out["indicator_completed_volume"] is None


def test_naive_timestamp_is_not_silently_assumed_utc() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        indicator_completed_bar={
            "timestamp": "2026-10-27T08:30:00",
            "volume": 0.0,
        },
        canonical_completed_bar=_bar(0.0),
    )
    assert out["completed_bar_parity"] == "missing_or_invalid_completed_timestamp"


def test_equal_minute_missing_volume_is_unknown_not_zero() -> None:
    out = describe_option_volume_evidence(
        {"volume": 0.0},
        indicator_completed_bar={
            "timestamp": "2026-10-27T08:30:00Z",
            "volume": None,
        },
        canonical_completed_bar=_bar(0.0),
    )
    assert out["completed_bar_parity"] == "missing_or_invalid_completed_volume"
