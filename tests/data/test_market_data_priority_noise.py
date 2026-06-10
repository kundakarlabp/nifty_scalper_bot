from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.message_bus import Message, MessageType
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.runner import StrategyRunner, SymbolState


class DummyIndicatorEngine:
    def __init__(self, count: int = 0) -> None:
        self.count = count

    def get_history(self, _symbol):
        return [object()] * self.count

    def has_min_bars(self, _symbol, required):
        return self.count >= required


class DummyRisk:
    available_balance = 1.0


def _mdm() -> MarketDataManager:
    manager = MarketDataManager(broker=None, settings={})
    manager._tick_queue = asyncio.Queue(maxsize=2)
    manager._active_contract_basket = {
        "selected_ce": "NFO:NIFTY2660923250CE",
        "selected_pe": "NFO:NIFTY2660923250PE",
        "option_symbols": [
            "NFO:NIFTY2660923250CE",
            "NFO:NIFTY2660923250PE",
            "NFO:NIFTY2660923500CE",
        ],
    }
    return manager


def test_queue_full_drops_low_priority_before_open_position_symbol() -> None:
    manager = _mdm()
    manager.set_open_position_symbols(["NFO:NIFTY2660923250CE"])

    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923500CE", "ltp": 1})
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923500PE", "ltp": 2})
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923250CE", "ltp": 3})

    queued = [manager._tick_queue.get_nowait()["symbol"] for _ in range(manager._tick_queue.qsize())]
    assert "NFO:NIFTY2660923250CE" in queued
    assert len(queued) == 2
    assert sum(manager._tick_queue_priority_drops.values()) >= 1


@pytest.mark.asyncio
async def test_open_position_tick_delivered_under_bus_queue_pressure() -> None:
    manager = _mdm()
    manager.set_open_position_symbols(["NFO:NIFTY2660923250CE"])
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    bus = SimpleNamespace(queues={MessageType.TICK: queue})
    low = Message(MessageType.TICK, datetime.now(timezone.utc), {"symbol": "NFO:NIFTY2660923500CE"}, "test")
    high = Message(MessageType.TICK, datetime.now(timezone.utc), {"symbol": "NFO:NIFTY2660923250CE"}, "test")
    queue.put_nowait(low)

    await manager._publish_tick_message_with_priority(
        bus, high, symbol="NFO:NIFTY2660923250CE", priority=0, bucket="open_position"
    )

    delivered = queue.get_nowait()
    assert delivered.data["symbol"] == "NFO:NIFTY2660923250CE"


def test_volume_delta_clamp_summarized_not_per_tick(caplog, monkeypatch) -> None:
    manager = _mdm()
    symbol = "NFO:NIFTY2660923250CE"
    manager._last_cumulative_volume_by_symbol[symbol] = 100.0
    monkeypatch.setenv("OPTION_MAX_REASONABLE_TICK_VOLUME_DELTA", "10")
    caplog.set_level(logging.WARNING)

    normalized = manager._normalise_tick_volume_delta(symbol, {"volume_traded_today": 1000})

    assert normalized["volume_delta"] == 10.0
    assert normalized["volume_delta_untrusted"] is True
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "OPTION_VOLUME_DELTA_CLAMP_SUMMARY" in text
    assert "OPTION_VOLUME_DELTA_CLAMPED" not in text


def _runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test.runner")
    runner._symbol_states = {}
    runner._symbol_bar_count = {}
    runner._hydration_ready_streak = {}
    runner._required_candles = 2
    runner._session_gap_count = {}
    runner._last_tick_time_by_symbol = {}
    runner._indicator_engine = DummyIndicatorEngine(2)
    runner._symbol_history = {}
    runner._context_required_bars = 2
    runner._market_data = SimpleNamespace()
    runner._active_symbols = set()
    runner._active_selected_ce = "NFO:NIFTY2660923250CE"
    runner._active_selected_pe = "NFO:NIFTY2660923250PE"
    runner._selected_ce_symbol = None
    runner._selected_pe_symbol = None
    runner._active_futures_symbol = "NFO:NIFTY26JUNFUT"
    runner._runtime_execution_ready_by_symbol = {}
    runner._runtime_readiness_reason = None
    runner._runtime_live_orders_armed = True
    runner._runtime_evaluation_ready = True
    runner._last_selected_subscription_state_key = None
    runner._should_log_throttled = lambda *_args, **_kwargs: True
    runner._set_symbol_hydration_state = lambda symbol, state: state
    runner._normalize_symbol = lambda symbol: symbol
    runner._required_bars_for_symbol = lambda _symbol: 2
    runner._history_count_for_symbol = lambda _symbol: 2
    runner._get_mdm_bars = lambda _symbol, _target: [{"timestamp": i} for i in range(2)]
    return runner


def test_duplicate_noop_history_sync_not_emitted_when_all_stores_warm(caplog) -> None:
    runner = _runner()
    symbol = "NFO:NIFTY2660923250CE"
    runner._symbol_history[symbol] = [object(), object()]
    called = False

    def emit(*_args, **_kwargs):
        nonlocal called
        called = True

    runner._emit_history_hydration_trace = emit

    result = runner._sync_history_from_mdm_cache(symbol, required_bars=2, source="pre_option_eval_option_sync")

    assert result == 2
    assert called is False


def test_repeated_missing_candles_suppressed_if_fresh_tick_exists(caplog, monkeypatch) -> None:
    runner = _runner()
    symbol = "NFO:NIFTY2660923250CE"
    runner._has_session_candle_gaps = lambda _symbol: True
    runner._session_gap_count[symbol] = 3
    runner._last_tick_time_by_symbol[symbol] = __import__("time").time()
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("RUNNER_CANDLE_GAP_GRACE_SECONDS", "30")
    old_bar = SimpleNamespace(timestamp=datetime.now(timezone.utc) - timedelta(minutes=5))

    state = runner.update_symbol_hydration(symbol, [old_bar, old_bar], {symbol: {"vwap": 1.0, "cum_volume": 1.0}})

    assert state != SymbolState.DEGRADED
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "CANDLE_GAP_SUPPRESSED_FRESH_TICK" in text
    assert "repeated_missing_candles" not in text


def test_selected_option_depth_missing_not_emitted_for_context_symbol() -> None:
    runner = _runner()
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:NIFTY2660923250CE", selected_pe="NFO:NIFTY2660923250PE"
    )
    runner._selected_option_symbol_for_side = lambda _side, _meta: None
    runner._symbol_role_for_runner = lambda symbol: "spot_context" if symbol == "NSE:NIFTY" else "option_context"
    runner._is_context_symbol_suspended = lambda _symbol: False
    runner._is_option_symbol_tick_fresh = lambda _symbol, max_age_s=60.0: True
    runner._selected_option_has_real_depth = lambda _symbol: False
    runner._market_data = SimpleNamespace(
        _token_by_symbol={"NFO:NIFTY2660923250CE": 1, "NFO:NIFTY2660923250PE": 2},
        _active_subscribed_symbols={"NFO:NIFTY2660923250CE", "NFO:NIFTY2660923250PE"},
        _desired_tokens={1, 2},
        _subscribed_tokens={1, 2},
        _confirmed_subscriptions={1, 2},
        _ws=SimpleNamespace(_tokens={1, 2}),
    )

    ready, reason = runner._emit_live_universe_bootstrap_status(symbol="NSE:NIFTY")

    assert ready is False
    assert reason == "context_symbol_not_tradable"
