from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nifty_scalper_bot.data.source import HistoricalLiveOHLCProvider, is_symbol_valid


def test_get_clean_ohlc_overwrites_last_row_without_appending() -> None:
    base = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 10,
            },
            {
                "timestamp": datetime(2026, 1, 1, 9, 16, tzinfo=timezone.utc),
                "open": 102.0,
                "high": 106.0,
                "low": 101.0,
                "close": 103.0,
                "volume": 11,
            },
        ]
    )

    provider = HistoricalLiveOHLCProvider(
        fetch_historical=lambda _symbol, _timeframe: base,
        get_current_live_candle=lambda _symbol: {
            "open": 103.0,
            "high": 108.0,
            "low": 102.0,
            "close": 107.0,
            "volume": 15,
        },
    )

    result = provider.get_clean_ohlc("NSE:NIFTY")

    assert len(result) == 2
    assert result.iloc[-1]["close"] == 107.0
    assert result.iloc[-1]["high"] == 108.0


def test_is_symbol_valid_rejects_duplicates_and_nans() -> None:
    idx = pd.to_datetime(
        [
            datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc),
        ],
        utc=True,
    )
    frame = pd.DataFrame({"close": [100.0, float("nan")]}, index=idx)

    assert is_symbol_valid(frame, min_required_bars=2) is False
