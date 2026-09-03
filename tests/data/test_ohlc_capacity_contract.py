from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import threading

import pandas as pd

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.ohlc_capacity_contract import (
    configured_ohlc_capacity,
    install_mdm_ohlc_capacity_contract,
)


SYMBOL = "NSE:NIFTY"


def _bars(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 9, 3, 3, 45, tzinfo=timezone.utc)  # 09:15 IST
    return [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 24_000.0 + index,
            "high": 24_001.0 + index,
            "low": 23_999.0 + index,
            "close": 24_000.5 + index,
            "volume": 1_000 + index,
        }
        for index in range(count)
    ]


def _seed_and_refresh(mdm: MarketDataManager, count: int) -> list[dict[str, object]]:
    rows = _bars(count)
    engine = mdm.get_candle_engine(SYMBOL)
    engine.import_history(pd.DataFrame(rows), mode="bootstrap", source="historical")
    mdm._refresh_candle_projection(SYMBOL)
    return rows


def test_completed_ohlc_capacity_has_full_session_floor(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "250")

    assert configured_ohlc_capacity() == 400


def test_completed_ohlc_capacity_does_not_exceed_native_engine(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "1000")

    assert configured_ohlc_capacity() == 500


def test_raw_tick_cache_stays_small_while_completed_projection_is_deeper(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)

    _seed_and_refresh(mdm, 100)

    assert mdm._cache_len == 50
    assert mdm._raw_tick_history[SYMBOL].maxlen == 50
    assert mdm._ohlc[SYMBOL].maxlen == 500
    assert len(mdm.get_ohlc_bars(SYMBOL)) == 100
    assert mdm.get_candle_engine(SYMBOL).diagnostics()["candle_store_maxlen"] == 500


def test_canonical_projection_keeps_opening_range_after_250_bar_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)
    rows = _seed_and_refresh(mdm, 400)

    full_session = mdm.get_ohlc_bars(SYMBOL, limit=400)

    assert len(full_session) == 400
    assert pd.Timestamp(full_session[0]["timestamp"]) == pd.Timestamp(rows[0]["timestamp"])
    assert mdm._ohlc[SYMBOL].maxlen == 500
    assert mdm._raw_tick_history[SYMBOL].maxlen == 50


def test_projection_remains_authoritative_for_reads(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)
    _seed_and_refresh(mdm, 100)

    assert mdm.get_latest_closed_bar(SYMBOL) is not None
    with mdm._lock:
        mdm._ohlc.pop(SYMBOL, None)

    assert mdm.get_latest_closed_bar(SYMBOL) is None


def test_runtime_legacy_cache_override_remains_authoritative(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)
    mdm._cache_len = 1

    _seed_and_refresh(mdm, 2)

    assert len(mdm.get_ohlc_bars(SYMBOL)) == 1
    assert mdm._ohlc[SYMBOL].maxlen == 1


def test_explicit_constructor_cache_len_keeps_legacy_combined_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None, cache_len=2)

    _seed_and_refresh(mdm, 3)

    assert len(mdm.get_ohlc_bars(SYMBOL)) == 2
    assert mdm._ohlc[SYMBOL].maxlen == 2


def test_manual_partial_mdm_uses_legacy_combined_cache_limit(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(warning=lambda *a, **k: None)
    mdm._cache_len = 1
    mdm._lock = threading.RLock()
    mdm._ohlc = defaultdict(lambda: deque(maxlen=1))
    mdm._engines = {}
    mdm._candle_metrics = defaultdict(float)
    mdm._candle_projection_diagnostics = {}
    mdm._bar_symbol_key = lambda value: str(value)
    mdm._canonical_symbol = lambda value: str(value)
    rows = _bars(2)
    engine = mdm.get_candle_engine(SYMBOL)
    engine.import_history(pd.DataFrame(rows), mode="bootstrap", source="historical")

    refreshed = mdm._refresh_candle_projection(SYMBOL)

    assert len(refreshed) == 1
    assert mdm._ohlc[SYMBOL].maxlen == 1


def test_small_history_requests_remain_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)
    rows = _seed_and_refresh(mdm, 400)

    result = mdm.get_ohlc_bars(SYMBOL, limit=30)

    assert len(result) == 30
    assert pd.Timestamp(result[0]["timestamp"]) == pd.Timestamp(rows[-30]["timestamp"])
