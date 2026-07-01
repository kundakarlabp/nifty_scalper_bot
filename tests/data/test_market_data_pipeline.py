
from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.data.pipeline import Candle, CandleStore
from nifty_scalper_bot.data.source import DataIntegrityError


def _candle(ts: datetime, close: float = 100.0) -> Candle:
    return Candle(symbol="NFO:TEST", timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_candle_store_seed_sorts_and_deduplicates() -> None:
    store = CandleStore()
    base = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    store.seed("NFO:TEST", [
        {"timestamp": base + timedelta(minutes=1), "open": 101, "high": 101, "low": 101, "close": 101},
        {"timestamp": base, "open": 100, "high": 100, "low": 100, "close": 100},
        {"timestamp": base, "open": 102, "high": 102, "low": 102, "close": 102},
    ])
    rows = store.get("NFO:TEST")
    assert [row.timestamp.to_pydatetime() for row in rows] == [base, base + timedelta(minutes=1)]
    assert rows[0].close == 102.0


def test_candle_store_push_duplicate_noop_and_older_rejected() -> None:
    store = CandleStore()
    base = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    store.push(_candle(base, 100.0))
    store.push(_candle(base, 101.0))
    assert len(store.get("NFO:TEST")) == 1
    with pytest.raises(DataIntegrityError):
        store.push(_candle(base - timedelta(minutes=1), 99.0))
