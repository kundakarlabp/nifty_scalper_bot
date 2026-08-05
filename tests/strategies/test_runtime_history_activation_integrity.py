from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.data import market_data_manager as mdm_module
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.utils.market_hours import MarketState


def _bars(symbol: str, count: int) -> list[dict]:
    start = datetime(2026, 8, 4, 9, 15, tzinfo=timezone.utc)
    return [
        {
            "symbol": symbol,
            "timestamp": start + timedelta(minutes=index),
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.5 + index,
            "volume": 100 + index,
        }
        for index in range(count)
    ]


def _runner_for_reseed() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test.runtime_history_activation")
    runner._normalize_symbol = str
    runner._lock = threading.RLock()
    runner._symbol_history = {}
    runner._indicator_engine = IndicatorEngine()
    runner._last_bar_ts = {}
    runner._active_symbols = set()
    runner._tracked_symbols = set()
    runner._data_phase = {}
    runner._set_symbol_hydration_state = lambda *_a, **_k: None
    runner._seed_pipeline_store = lambda *_a, **_k: None
    runner._seed_candle_engine_from_history = lambda *_a, **_k: None
    runner._maybe_promote_pending_active_basket = lambda **_k: None
    return runner


def test_history_reseed_does_not_activate_deferred_symbol() -> None:
    runner = _runner_for_reseed()
    symbol = "NFO:NIFTY2680424550CE"

    count = runner.reseed_history_from_bars(
        symbol,
        _bars(symbol, 30),
        source="startup_hydration",
        min_bars=30,
    )

    assert count == 30
    assert symbol in runner._symbol_history
    assert symbol not in runner._active_symbols
    assert symbol not in runner._tracked_symbols


def test_fallback_backfill_reseeds_only_cold_active_symbols() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    warm = "NSE:NIFTY"
    cold = "NFO:NIFTY2680424600CE"
    warm_rows = _bars(warm, 30)
    cold_rows = _bars(cold, 20)
    histories = {warm: list(warm_rows), cold: []}
    reseeded: list[str] = []
    requested: list[tuple[str, int]] = []

    runner._logger = logging.getLogger("test.runtime_history_fallback")
    runner._lock = threading.RLock()
    runner._active_symbols = {warm, cold}
    runner._required_candles = 20
    runner._indicator_engine = SimpleNamespace(
        get_history=lambda symbol: list(histories.get(symbol, []))
    )
    runner._required_bars_for_symbol = lambda _symbol: 20
    runner._get_mdm_bars = lambda symbol, _target: (
        list(warm_rows) if symbol == warm else list(cold_rows)
    )
    runner._set_symbol_hydration_state = lambda *_a, **_k: None
    runner._request_mdm_hydration = (
        lambda symbol, target: requested.append((symbol, target))
    )
    runner.ingest_historical_bar = lambda _row: (_ for _ in ()).throw(
        AssertionError("fallback must replace canonical history, not append rows")
    )

    def _reseed(symbol, rows, *, source, min_bars):
        assert source == "runner_fallback_backfill"
        assert min_bars == 20
        reseeded.append(symbol)
        histories[symbol] = list(rows)
        return len(histories[symbol])

    runner.reseed_history_from_bars = _reseed

    asyncio.run(runner._backfill_history())

    assert reseeded == [cold]
    assert len(histories[warm]) == 30
    assert len(histories[cold]) == 20
    assert requested == []


def test_stale_spot_history_cannot_satisfy_live_readiness() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._tick_stale_threshold_ms = 60_000
    mdm._is_symbol_fresh = lambda *_a, **_k: False
    mdm._logger = logging.getLogger("test.stale_spot_readiness")
    spot = "NSE:NIFTY"
    ce = "NFO:NIFTY2680424600CE"
    pe = "NFO:NIFTY2680424600PE"

    state = mdm._readiness_state(
        {spot: 100, ce: 75, pe: 75},
        20,
        {
            "spot": spot,
            "atm_ce": ce,
            "atm_pe": pe,
            "options": [ce, pe],
        },
    )

    assert state["spot_ready"] is False
    assert state["hard_ready"] is False
    assert "fresh_spot_tick_missing" in state["missing_hard"]


def test_wait_until_ready_reports_configured_spot_stale_threshold(
    monkeypatch, caplog
) -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    spot = "NSE:NIFTY"
    ce = "NFO:NIFTY2680424600CE"
    pe = "NFO:NIFTY2680424600PE"
    mdm._lock = threading.RLock()
    mdm._active_subscribed_symbols = {spot, ce, pe}
    mdm._raw_tick_history = {spot: [1], ce: [1], pe: [1]}
    mdm._min_required_bars = 1
    mdm._readiness_requirements = {
        "spot": spot,
        "atm_ce": ce,
        "atm_pe": pe,
        "options": [ce, pe],
    }
    mdm._last_readiness_state = {}
    mdm._spot_ready_logged = False
    mdm._last_tick_source = {spot: "ws"}
    mdm._token_by_symbol = {spot: 256265}
    mdm._tick_stale_threshold_ms = 60_000
    mdm._is_symbol_fresh = lambda *_a, **_k: False
    mdm.symbol_data_age_ms_or_none = lambda _symbol: 61_000
    mdm.hydration_complete = False
    mdm.ready = False
    mdm.degraded = False
    mdm._logger = logging.getLogger("test.wait_until_ready_threshold")
    monkeypatch.setattr(
        mdm_module, "get_market_state", lambda: MarketState.POSTMARKET
    )
    caplog.set_level(logging.INFO, logger="test.wait_until_ready_threshold")

    asyncio.run(mdm.wait_until_ready(timeout=0.01))

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "SPOT_NOT_READY"
    )
    assert record.age_ms == 61_000
    assert record.threshold_ms == 60_000
