import logging
from collections import defaultdict

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _make_mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = logging.getLogger("test_mdm")
    mdm._lock = type(
        "L", (), {"__enter__": lambda s: s, "__exit__": lambda s, a, b, c: None}
    )()
    mdm._symbol_by_token = {1: "NSE:NIFTY"}
    mdm._token_to_symbol = {1: "NSE:NIFTY"}
    mdm._desired_tokens = {1}
    mdm._candle_metrics = defaultdict(float)
    return mdm


def test_epoch_exchange_timestamp_rejected_not_synthetic_current() -> None:
    mdm = _make_mdm()
    tick = mdm._normalize_ws_tick(
        {
            "instrument_token": 1,
            "last_price": 10,
            "exchange_timestamp": "1970-01-01 00:00:00",
        }
    )
    assert tick is None
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_none_timestamp_rejected_not_synthetic_current() -> None:
    mdm = _make_mdm()
    tick = mdm._normalize_ws_tick({"instrument_token": 1, "last_price": 10})
    assert tick is None
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_valid_exchange_timestamp_kept() -> None:
    mdm = _make_mdm()
    tick = mdm._normalize_ws_tick(
        {
            "instrument_token": 1,
            "last_price": 10,
            "exchange_timestamp": "2026-04-30T09:15:00Z",
        }
    )
    assert tick["timestamp"] == "2026-04-30T14:45:00+05:30"
