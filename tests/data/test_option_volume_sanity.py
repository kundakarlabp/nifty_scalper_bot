"""Option OHLCV normalization must preserve authenticated completed-bar volume."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.data.normalizers import normalize_history_row


@pytest.mark.parametrize("volume", [5_000_000, 6_989_060, 7_854_730])
@pytest.mark.parametrize("suffix", ["CE", "PE"])
def test_option_completed_minute_preserves_volume_above_legacy_cap(
    monkeypatch, volume: int, suffix: str
) -> None:
    monkeypatch.setenv("OPTION_MAX_REASONABLE_1M_VOLUME", "1")
    row = {
        "timestamp": datetime(2026, 11, 3, 7, 2, tzinfo=timezone.utc),
        "open": 110.0,
        "high": 112.0,
        "low": 109.0,
        "close": 111.0,
        "volume": volume,
    }
    out = normalize_history_row(f"NFO:NIFTY26N0323650{suffix}", row)
    assert out is not None
    assert out["volume"] == volume


def test_non_option_history_volume_is_preserved() -> None:
    row = {
        "timestamp": datetime(2026, 11, 3, 7, 2, tzinfo=timezone.utc),
        "open": 10,
        "high": 12,
        "low": 9,
        "close": 11,
        "volume": 120_319_745,
    }
    out = normalize_history_row("NSE:NIFTY", row)
    assert out is not None
    assert out["volume"] == 120_319_745
