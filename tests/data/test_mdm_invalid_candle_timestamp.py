from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        log=lambda *a, **k: None,
    )
    mdm._lock = threading.RLock()
    mdm._symbol_by_token = {1: "NFO:X"}
    mdm._token_to_symbol = {1: "NFO:X"}
    mdm._desired_tokens = {1}
    mdm._candle_metrics = defaultdict(float)
    return mdm


def _tick(**kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {"instrument_token": 1, "last_price": 10.0}
    payload.update(kwargs)
    return payload


def test_malformed_timestamp_rejected_for_candle_processing() -> None:
    mdm = _mdm()
    assert mdm._normalize_ws_tick(_tick(exchange_timestamp="bad")) is None
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_invalid_exchange_timestamp_falls_back_to_valid_timestamp() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(
        _tick(exchange_timestamp="bad", timestamp="2026-01-01T00:00:00Z")
    )
    assert normalized is not None
    assert normalized["timestamp_source"] == "timestamp"


def test_invalid_exchange_timestamp_falls_back_to_last_trade_time() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(
        _tick(exchange_timestamp="bad", last_trade_time="2026-01-01T00:00:00Z")
    )
    assert normalized is not None
    assert normalized["timestamp_source"] == "last_trade_time"


def test_last_trade_time_only_broker_tick_is_accepted() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(_tick(last_trade_time="2026-01-01T00:00:00Z"))
    assert normalized is not None
    assert normalized["timestamp_source"] == "last_trade_time"


def test_missing_timestamp_rejected_for_candle_processing() -> None:
    mdm = _mdm()
    assert mdm._normalize_ws_tick(_tick()) is None
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_ancient_timestamp_rejected_for_candle_processing() -> None:
    mdm = _mdm()
    assert (
        mdm._normalize_ws_tick(_tick(exchange_timestamp="2019-01-01T00:00:00Z")) is None
    )
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_valid_exchange_timestamp_accepted() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(
        _tick(exchange_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    )
    assert normalized is not None
    assert normalized["timestamp_source"] == "exchange_timestamp"
    assert normalized["source_timestamp_valid"] is True


def test_valid_rest_poll_timestamp_accepted_and_marked() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(
        _tick(source="rest_poll", timestamp="2026-01-01T00:00:00Z")
    )
    assert normalized is not None
    assert normalized["timestamp_source"] == "timestamp"
    assert normalized["source_timestamp_valid"] is True


def test_all_source_timestamps_invalid_rejected() -> None:
    mdm = _mdm()
    assert (
        mdm._normalize_ws_tick(
            _tick(
                exchange_timestamp="bad",
                timestamp="2019-01-01T00:00:00Z",
                last_trade_time="bad-too",
            )
        )
        is None
    )
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_invalid_timestamp_does_not_mutate_candle_engine() -> None:
    mdm = _mdm()
    mdm._cache_len = 10
    mdm._ohlc = defaultdict(lambda: __import__("collections").deque(maxlen=10))
    mdm._engines = {}
    mdm._bar_symbol_key = lambda s: str(s)
    mdm._canonical_symbol = lambda s: str(s)
    engine = mdm.get_candle_engine("NFO:X")
    mdm._process_queued_tick(_tick(exchange_timestamp="bad"))
    assert engine.current_candle is None
    assert engine.get_completed_bars() == []


def test_missing_timestamp_logs_missing_all_reason(caplog) -> None:
    mdm = _mdm()
    mdm._logger = logging.getLogger("test_mdm_timestamp_reason")
    mdm._symbol_by_token[2] = "NFO:Y"
    mdm._token_to_symbol[2] = "NFO:Y"
    with caplog.at_level("WARNING", logger="test_mdm_timestamp_reason"):
        assert (
            mdm._normalize_ws_tick({"instrument_token": 2, "last_price": 10.0}) is None
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("timestamp_reason=missing_all" in message for message in messages)


def test_invalid_tick_does_not_stop_later_valid_tick_processing() -> None:
    mdm = MarketDataManager(kite=None)
    mdm._symbol_by_token[1] = "NFO:X"
    mdm._token_to_symbol[1] = "NFO:X"
    mdm._symbol_to_token["NFO:X"] = 1
    mdm._token_by_symbol["NFO:X"] = 1

    mdm._process_queued_tick(_tick(exchange_timestamp="bad"))
    mdm._process_queued_tick(
        _tick(exchange_timestamp="2026-07-23T09:30:04+05:30", last_price=11.0)
    )

    assert "NFO:X" in mdm._latest_ticks
    assert mdm._latest_ticks["NFO:X"]["last_price"] == 11.0
    assert mdm.get_candle_engine("NFO:X").current_candle is not None


def test_nifty_spot_invalid_exchange_timestamp_uses_valid_received_at_fallback() -> (
    None
):
    mdm = MarketDataManager(kite=None)
    mdm._symbol_by_token[256265] = "NSE:NIFTY"
    mdm._token_to_symbol[256265] = "NSE:NIFTY"
    mdm._symbol_to_token["NSE:NIFTY"] = 256265
    mdm._token_by_symbol["NSE:NIFTY"] = 256265

    normalized = mdm._normalize_ws_tick(
        {
            "instrument_token": 256265,
            "last_price": 24500.0,
            "exchange_timestamp": "bad",
            "received_at": "2026-07-23T09:30:05+05:30",
        }
    )

    assert normalized is not None
    assert normalized["symbol"] == "NSE:NIFTY"
    assert normalized["timestamp_source"] == "received_at"


def test_nifty_spot_naive_broker_fallback_is_ist_not_utc_shifted() -> None:
    mdm = MarketDataManager(kite=None)
    mdm._symbol_by_token[256265] = "NSE:NIFTY"
    mdm._token_to_symbol[256265] = "NSE:NIFTY"
    mdm._symbol_to_token["NSE:NIFTY"] = 256265
    mdm._token_by_symbol["NSE:NIFTY"] = 256265

    normalized = mdm._normalize_ws_tick(
        {
            "instrument_token": 256265,
            "last_price": 24500.0,
            "exchange_timestamp": "bad",
            "timestamp": "2026-07-23 09:30:00",
        }
    )

    assert normalized is not None
    assert normalized["timestamp_source"] == "timestamp"
    assert normalized["timestamp"] == "2026-07-23T09:30:00+05:30"
    assert normalized["source_timestamp_valid"] is True


def test_poll_timestamp_with_invalid_broker_timestamp_uses_received_at() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(
        _tick(source="poll", timestamp="bad", received_at=1784788200.0)
    )
    assert normalized is not None
    assert normalized["timestamp_source"] == "received_at"
    assert normalized["timestamp"] == "2026-07-23T12:00:00+05:30"
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 0


def test_poll_timestamp_missing_broker_timestamp_uses_received_at() -> None:
    mdm = _mdm()
    normalized = mdm._normalize_ws_tick(_tick(source="poll", received_at=1784788200.0))
    assert normalized is not None
    assert normalized["timestamp_source"] == "received_at"
    assert normalized["timestamp"] == "2026-07-23T12:00:00+05:30"


def test_poll_numeric_epoch_seconds_and_milliseconds_are_not_double_shifted() -> None:
    mdm = _mdm()
    seconds = mdm._normalize_ws_tick(_tick(source="poll", timestamp=1784788200.0))
    millis = mdm._normalize_ws_tick(_tick(source="poll", timestamp=1784788200000.0))
    assert seconds is not None
    assert millis is not None
    assert seconds["timestamp_source"] == "timestamp"
    assert millis["timestamp_source"] == "timestamp"
    assert seconds["timestamp"] == "2026-07-23T12:00:00+05:30"
    assert millis["timestamp"] == "2026-07-23T12:00:00+05:30"


def test_prepare_rest_tick_continues_to_valid_last_trade_time() -> None:
    mdm = _mdm()
    valid_last_trade = datetime(2026, 7, 23, 9, 30, tzinfo=timezone.utc)

    prepared = mdm._prepare_rest_tick(
        {
            "timestamp": "bad",
            "last_trade_time": valid_last_trade,
            "last_price": 10.0,
        },
        source="poll",
    )
    normalized = mdm._normalize_ws_tick({**prepared, "instrument_token": 1})

    assert prepared["timestamp"] is valid_last_trade
    assert isinstance(prepared["received_at"], float)
    assert normalized is not None
    assert normalized["timestamp_source"] == "last_trade_time"


def test_prepare_rest_tick_continues_to_valid_broker_timestamp() -> None:
    valid_broker = datetime(2026, 7, 23, 9, 31, tzinfo=timezone.utc)

    prepared = MarketDataManager._prepare_rest_tick(
        {
            "timestamp": "bad",
            "last_trade_time": "also-bad",
            "broker_timestamp": valid_broker,
            "last_price": 10.0,
        },
        source="poll",
    )

    normalized = _mdm()._normalize_ws_tick({**prepared, "instrument_token": 1})

    assert prepared["timestamp"] is valid_broker
    assert isinstance(prepared["received_at"], float)
    assert normalized is not None
    assert normalized["timestamp_source"] == "timestamp"


def test_prepare_rest_tick_omits_all_invalid_broker_fields_for_received_at_fallback() -> None:
    mdm = _mdm()
    prepared = mdm._prepare_rest_tick(
        {
            "timestamp": "bad",
            "last_trade_time": "also-bad",
            "broker_timestamp": "still-bad",
            "ts": "bad-ts",
            "ts_ms": "bad-ts-ms",
            "last_price": 10.0,
            "_local_timestamp": 1784788200.0,
        },
        source="poll",
    )
    normalized = mdm._normalize_ws_tick({**prepared, "instrument_token": 1})

    assert "timestamp" not in prepared
    assert prepared["received_at"] == 1784788200.0
    assert normalized is not None
    assert normalized["timestamp_source"] == "received_at"
