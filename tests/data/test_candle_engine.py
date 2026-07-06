from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.data.candle_engine import (
    CandleEngine,
    detect_gap,
    ensure_valid_data,
    fetch_historical_safe,
    repair_with_backfill,
    sanitize,
    validate_dataframe,
)

IST = ZoneInfo("Asia/Kolkata")


def _tick(ts: datetime, price: float) -> dict[str, float | datetime]:
    return {"symbol": "NFO:NIFTY", "timestamp": ts, "ltp": price, "volume": 1.0}


def test_candle_engine_finalizes_closed_candle() -> None:
    engine = CandleEngine()
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    engine.on_tick(_tick(ts, 100.0))
    engine.on_tick(_tick(ts + timedelta(seconds=10), 102.0))
    engine.on_tick(_tick(ts + timedelta(minutes=1), 101.0))

    df = engine.get_df()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 102.0
    assert row["low"] == 100.0
    assert row["close"] == 102.0
    assert row["timestamp"].isoformat() == "2026-01-02T14:45:00+05:30"
    assert row["timestamp"].tzinfo == IST
    assert engine.last_candle_close is not None
    assert engine.last_candle_close.isoformat() == "2026-01-02T14:45:00+05:30"


def test_detect_gap_returns_true_for_missing_minute() -> None:
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1},
            {
                "timestamp": ts + timedelta(minutes=1),
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
            },
            {
                "timestamp": ts + timedelta(minutes=3),
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
            },
        ]
    )

    assert detect_gap(df) is True
    assert validate_dataframe(df, min_required=3) is False


def test_fetch_historical_safe_retries_until_minimum() -> None:
    calls = {"count": 0}

    def fetch(_symbol: str) -> pd.DataFrame:
        calls["count"] += 1
        if calls["count"] < 3:
            return pd.DataFrame([{"timestamp": datetime.now(timezone.utc)}])
        ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
        return pd.DataFrame(
            [{"timestamp": ts + timedelta(minutes=i)} for i in range(50)]
        )

    result = fetch_historical_safe("NFO:NIFTY", fetch_historical=fetch, min_required=50)
    assert result is not None
    assert len(result) == 50
    assert calls["count"] == 3


def test_ensure_valid_data_hydrates_when_cache_invalid() -> None:
    engine = CandleEngine()
    engine.df = pd.DataFrame()
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    hist = pd.DataFrame(
        [
            {
                "timestamp": ts + timedelta(minutes=i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }
            for i in range(50)
        ]
    )

    out = ensure_valid_data(
        "NFO:NIFTY",
        engine,
        fetch_historical=lambda _symbol: hist,
        fetch_recent_rest=lambda _symbol: hist,
        min_required=50,
    )

    assert out is not None
    assert len(out) == 50


def test_sanitize_deduplicates_and_drops_missing_close() -> None:
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    dirty = pd.DataFrame(
        [
            {"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": None},
            {
                "timestamp": ts + timedelta(minutes=1),
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
            },
            {
                "timestamp": ts + timedelta(minutes=1),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.6,
            },
        ]
    )

    cleaned = sanitize(dirty)
    assert len(cleaned) == 1
    assert cleaned["close"].isna().sum() == 0


def test_sanitize_converts_timestamps_to_ist() -> None:
    dirty = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-02T09:15:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }
        ]
    )

    cleaned = sanitize(dirty)

    assert cleaned.iloc[0]["timestamp"].isoformat() == "2026-01-02T14:45:00+05:30"
    assert cleaned.iloc[0]["timestamp"].tzinfo == IST


def test_repair_with_backfill_merges_existing_and_recent() -> None:
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    old = pd.DataFrame(
        [
            {
                "timestamp": ts + timedelta(minutes=i),
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
            }
            for i in range(2)
        ]
    )
    recent = pd.DataFrame(
        [
            {
                "timestamp": ts + timedelta(minutes=1),
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.5,
            },
            {
                "timestamp": ts + timedelta(minutes=2),
                "open": 202.0,
                "high": 203.0,
                "low": 201.0,
                "close": 202.5,
            },
        ]
    )

    merged = repair_with_backfill(
        "NFO:NIFTY",
        old,
        fetch_recent_rest=lambda _symbol: recent,
    )
    assert len(merged) == 3
    assert float(merged.iloc[-1]["close"]) == 202.5


def test_candle_engine_flush_is_idempotent() -> None:
    engine = CandleEngine()
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    engine.on_tick(_tick(ts, 100.0))
    first = engine.flush()
    second = engine.flush()
    assert first is not None
    assert second is None
    assert len(engine.get_df()) == 1


def test_candle_engine_equal_duplicate_is_noop_and_older_rejected() -> None:
    engine = CandleEngine()
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    engine.on_tick(_tick(ts, 100.0))
    assert engine.flush() is not None
    engine.current_candle = {
        "timestamp": pd.Timestamp(ts),
        "open": 100.0,
        "high": 100.0,
        "low": 100.0,
        "close": 100.0,
        "volume": 1.0,
    }
    assert engine.flush() is None
    assert engine.current_candle is None
    assert len(engine.get_df()) == 1
    engine.current_candle = {
        "timestamp": pd.Timestamp(ts - timedelta(minutes=1)),
        "open": 99.0,
        "high": 99.0,
        "low": 99.0,
        "close": 99.0,
        "volume": 1.0,
    }
    from nifty_scalper_bot.data.source import DataIntegrityError

    try:
        engine.flush()
    except DataIntegrityError:
        pass
    else:
        raise AssertionError("older candle should be rejected")
