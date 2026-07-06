from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.data.pipeline import Candle, CandleStore, MarketDataPipeline
from nifty_scalper_bot.data.source import DataIntegrityError

IST = ZoneInfo("Asia/Kolkata")


def _candle(ts: datetime, close: float = 100.0) -> Candle:
    return Candle(
        symbol="NFO:TEST",
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def test_candle_store_seed_sorts_and_deduplicates() -> None:
    store = CandleStore()
    base = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    store.seed(
        "NFO:TEST",
        [
            {
                "timestamp": base + timedelta(minutes=1),
                "open": 101,
                "high": 101,
                "low": 101,
                "close": 101,
            },
            {"timestamp": base, "open": 100, "high": 100, "low": 100, "close": 100},
            {"timestamp": base, "open": 102, "high": 102, "low": 102, "close": 102},
        ],
    )
    rows = store.get("NFO:TEST")
    assert [row.timestamp.isoformat() for row in rows] == [
        "2026-01-02T14:45:00+05:30",
        "2026-01-02T14:46:00+05:30",
    ]
    assert all(row.timestamp.tzinfo == IST for row in rows)
    assert rows[0].close == 102.0


def test_candle_store_push_duplicate_noop_and_older_rejected() -> None:
    store = CandleStore()
    base = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    store.push(_candle(base, 100.0))
    store.push(_candle(base, 101.0))
    assert len(store.get("NFO:TEST")) == 1
    with pytest.raises(DataIntegrityError):
        store.push(_candle(base - timedelta(minutes=3), 99.0))
    assert len(store.get("NFO:TEST")) == 1


def test_market_data_pipeline_catches_store_integrity_rejection(caplog) -> None:
    pipeline = MarketDataPipeline()
    base = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    pipeline.store.push(_candle(base + timedelta(minutes=3), 101.0))

    assert (
        pipeline.on_tick(
            {"symbol": "NFO:TEST", "timestamp": base, "ltp": 100.0, "volume": 1}
        )
        is None
    )
    candle = pipeline.on_tick(
        {
            "symbol": "NFO:TEST",
            "timestamp": base + timedelta(minutes=1),
            "ltp": 102.0,
            "volume": 1,
        }
    )

    assert candle is None
    assert any(
        getattr(record, "event", "") == "pipeline_candle_store_rejected"
        for record in caplog.records
    )


def test_market_data_pipeline_emits_closed_candle_with_ist_timestamp() -> None:
    pipeline = MarketDataPipeline()
    base = datetime(2026, 1, 2, 9, 15, 42, tzinfo=timezone.utc)

    assert (
        pipeline.on_tick(
            {"symbol": "NFO:TEST", "timestamp": base, "ltp": 100.0, "volume": 1}
        )
        is None
    )
    candle = pipeline.on_tick(
        {
            "symbol": "NFO:TEST",
            "timestamp": base + timedelta(minutes=1),
            "ltp": 102.0,
            "volume": 1,
        }
    )

    assert candle is not None
    assert candle.timestamp.isoformat() == "2026-01-02T14:45:00+05:30"
    assert candle.timestamp.tzinfo == IST
