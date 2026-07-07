from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.data.pipeline import Candle, CandleStore
from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.validator import validate_tick

IST = ZoneInfo("Asia/Kolkata")


def test_naive_tick_timestamp_is_local_ist() -> None:
    tick = validate_tick({"symbol": "NFO:TEST", "timestamp": "2026-01-02 11:55:00", "ltp": 100})
    assert tick.timestamp.isoformat() == "2026-01-02T11:55:00+05:30"


def test_naive_candle_timestamp_is_local_ist() -> None:
    store = CandleStore()
    store.push(Candle("NFO:TEST", datetime(2026, 1, 2, 11, 55), 100, 100, 100, 100, 1))
    assert store.get("NFO:TEST")[0].timestamp.isoformat() == "2026-01-02T11:55:00+05:30"


def test_future_candle_is_rejected_without_store_write() -> None:
    store = CandleStore()
    ts = datetime.now(IST) + timedelta(minutes=10)
    with pytest.raises(DataIntegrityError):
        store.push(Candle("NFO:TEST", ts, 100, 100, 100, 100, 1))
    assert store.get("NFO:TEST") == []
