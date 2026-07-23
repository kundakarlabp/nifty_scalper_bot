from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nifty_scalper_bot.data.candle_engine import CandleEngine
from nifty_scalper_bot.data.pipeline import MarketDataPipeline, TickValidator
from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.time_contract import normalize_market_tick_timestamp
from nifty_scalper_bot.data.validator import validate_tick

IST = ZoneInfo("Asia/Kolkata")


def test_numeric_epoch_ms_converts_utc_to_ist() -> None:
    raw = int(pd.Timestamp("2026-01-02T09:15:00Z").timestamp() * 1000)

    normalized = normalize_market_tick_timestamp(
        {"symbol": "NFO:TEST", "exchange_timestamp": raw}
    )

    assert normalized.timestamp.isoformat() == "2026-01-02T14:45:00+05:30"
    assert normalized.source == "exchange_timestamp"


def test_naive_broker_timestamp_is_interpreted_as_ist() -> None:
    normalized = normalize_market_tick_timestamp(
        {"symbol": "NFO:TEST", "timestamp": "2026-01-02 09:15:00"}
    )

    assert normalized.timestamp.isoformat() == "2026-01-02T09:15:00+05:30"
    assert normalized.source == "timestamp"


def test_bad_first_exchange_field_falls_back_to_valid_last_trade_time() -> None:
    normalized = normalize_market_tick_timestamp(
        {
            "exchange_timestamp": "bad",
            "last_trade_time": "2026-07-23T09:30:01+05:30",
        }
    )

    assert normalized.timestamp.isoformat() == "2026-07-23T09:30:01+05:30"
    assert normalized.source == "last_trade_time"
    assert normalized.raw_value == "2026-07-23T09:30:01+05:30"


def test_bad_exchange_fields_fall_back_to_valid_broker_timestamp() -> None:
    normalized = normalize_market_tick_timestamp(
        {
            "exchange_timestamp": "bad",
            "last_trade_time": "also-bad",
            "timestamp": "2026-07-23T09:30:02+05:30",
        }
    )

    assert normalized.timestamp.isoformat() == "2026-07-23T09:30:02+05:30"
    assert normalized.source == "timestamp"


def test_bad_broker_timestamp_falls_back_to_valid_received_at() -> None:
    normalized = normalize_market_tick_timestamp(
        {
            "timestamp": "bad",
            "received_at": "2026-07-23T09:30:03+05:30",
        }
    )

    assert normalized.timestamp.isoformat() == "2026-07-23T09:30:03+05:30"
    assert normalized.source == "received_at"


def test_all_present_market_timestamp_fields_invalid_raises_final_value_error() -> None:
    with pytest.raises(ValueError, match="all present market timestamps are invalid"):
        normalize_market_tick_timestamp(
            {
                "exchange_timestamp": "bad",
                "last_trade_time": "also-bad",
                "timestamp": "still-bad",
                "received_at": "bad-too",
            }
        )


def test_tick_validator_prefers_exchange_time_over_received_at() -> None:
    tick = TickValidator().validate(
        {
            "symbol": "nfo:test",
            "exchange_timestamp": "2026-01-02 09:15:00",
            "received_at": "2026-01-02 09:20:00",
            "ltp": 100,
        }
    )

    assert tick is not None
    assert tick.symbol == "NFO:TEST"
    assert tick.timestamp.isoformat() == "2026-01-02T09:15:00+05:30"
    assert tick.timestamp_source == "exchange_timestamp"


def test_validate_tick_rejects_missing_timestamp_instead_of_using_now() -> None:
    with pytest.raises(DataIntegrityError):
        validate_tick({"symbol": "NFO:TEST", "ltp": 100})


def test_received_at_fallback_is_explicit_not_exchange_time() -> None:
    tick = TickValidator().validate(
        {
            "symbol": "NFO:TEST",
            "received_at": "2026-01-02T09:15:00Z",
            "ltp": 100,
        }
    )

    assert tick is not None
    assert tick.timestamp.isoformat() == "2026-01-02T14:45:00+05:30"
    assert tick.timestamp_source == "received_at"


def test_pipeline_rejects_future_received_at_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = MarketDataPipeline()
    future = datetime.now(IST) + timedelta(minutes=10)

    with caplog.at_level(logging.WARNING, logger="nifty_scalper_bot.data.pipeline"):
        out = pipeline.on_tick(
            {"symbol": "NFO:TEST", "received_at": future, "ltp": 100}
        )

    assert out is None
    record = next(
        rec
        for rec in caplog.records
        if getattr(rec, "event", "") == "future_tick_rejected"
    )
    assert record.symbol == "NFO:TEST"
    assert record.timestamp_source == "received_at"
    assert record.future_by_sec > 0
    assert record.now_ist.endswith("+05:30")


def test_candle_engine_future_drop_has_forensic_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = CandleEngine()
    future = datetime.now(IST) + timedelta(minutes=10)

    with caplog.at_level(
        logging.WARNING, logger="nifty_scalper_bot.data.candle_engine"
    ):
        out = engine.on_tick(
            {"symbol": "NFO:TESTCE", "timestamp": future, "ltp": 100, "volume": 1}
        )

    assert out is None
    record = next(
        rec
        for rec in caplog.records
        if getattr(rec, "reason", "") == "future_timestamp"
    )
    assert record.symbol == "NFO:TESTCE"
    assert record.timestamp_source == "timestamp"
    assert record.tick_ts_ist.endswith("+05:30")
    assert record.now_ist.endswith("+05:30")
    assert record.future_by_sec > 0


def test_candle_engine_missing_symbol_is_dropped_not_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = CandleEngine()
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)

    with caplog.at_level(
        logging.WARNING, logger="nifty_scalper_bot.data.candle_engine"
    ):
        out = engine.on_tick({"timestamp": ts, "ltp": 100, "volume": 1})

    assert out is None
    record = next(
        rec for rec in caplog.records if getattr(rec, "reason", "") == "missing_symbol"
    )
    assert getattr(record, "symbol", None) is None
