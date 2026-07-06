from __future__ import annotations

import pandas as pd
import pytest

from nifty_scalper_bot.data.pipeline import Candle, CandleStore
from nifty_scalper_bot.data.source import DataIntegrityError


def _candle(symbol: str, ts: str, close: float = 100.0) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=pd.Timestamp(ts),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def test_hydration_live_overlap_candle_is_quietly_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_CANDLE_OVERLAP_TOLERANCE_SECONDS", "120")
    store = CandleStore()
    symbol = "NFO:NIFTY2670724400CE"

    store.push(_candle(symbol, "2026-07-06T08:35:00Z"))
    store.push(_candle(symbol, "2026-07-06T08:34:00Z"))

    candles = store.get(symbol)
    assert len(candles) == 1
    assert pd.Timestamp(candles[0].timestamp) == pd.Timestamp("2026-07-06T08:35:00Z")


def test_old_out_of_order_candle_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_CANDLE_OVERLAP_TOLERANCE_SECONDS", "120")
    store = CandleStore()
    symbol = "NFO:NIFTY2670724400CE"

    store.push(_candle(symbol, "2026-07-06T08:35:00Z"))

    with pytest.raises(DataIntegrityError, match="monotonic"):
        store.push(_candle(symbol, "2026-07-06T08:20:00Z"))
