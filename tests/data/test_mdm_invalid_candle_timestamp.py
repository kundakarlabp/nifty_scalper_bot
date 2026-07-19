from __future__ import annotations

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
    assert normalized["timestamp_source"] == "rest_poll"
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
