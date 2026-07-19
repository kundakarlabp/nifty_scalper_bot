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
