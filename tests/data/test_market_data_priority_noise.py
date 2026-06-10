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


def test_selected_option_depth_missing_not_emitted_for_context_symbol(caplog) -> None:
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

    caplog.set_level(logging.INFO)

    ready, reason = runner._emit_live_universe_bootstrap_status(symbol="NSE:NIFTY")

    assert ready is False
    assert reason == "context_symbol_not_tradable"
    record = next(record for record in caplog.records if record.getMessage().startswith("LIVE_UNIVERSE_BOOTSTRAP_STATUS"))
    assert record.evaluated_symbol == "NSE:NIFTY"
    assert record.selected_ce == "NFO:NIFTY2660923250CE"
    assert record.selected_pe == "NFO:NIFTY2660923250PE"
    assert record.selected_pair == ["NFO:NIFTY2660923250CE", "NFO:NIFTY2660923250PE"]
    assert record.symbol_role == "spot_context"
    assert record.reason == "context_symbol_not_tradable"
    assert record.ready is False


def _queued_symbols(manager: MarketDataManager) -> list[str]:
    return [manager._tick_queue.get_nowait()["symbol"] for _ in range(manager._tick_queue.qsize())]


def test_same_symbol_open_position_tick_is_coalesced() -> None:
    manager = _mdm()
    symbol = "NFO:NIFTY2660923250CE"
    manager.set_open_position_symbols([symbol])
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": symbol, "ltp": 1})
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923250PE", "ltp": 2})

    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": symbol, "ltp": 3})

    queued = _queued_symbols(manager)
    assert queued.count(symbol) == 1
    assert manager._tick_queue_priority_coalesced["open_position"] == 1


def test_different_open_position_symbol_forced_drop_logs_critical(caplog) -> None:
    manager = _mdm()
    first = "NFO:NIFTY2660923250CE"
    second = "NFO:NIFTY2660923250PE"
    third = "NFO:NIFTY2660923300CE"
    manager.set_open_position_symbols([first, second, third])
    caplog.set_level(logging.CRITICAL)
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": first, "ltp": 1})
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": second, "ltp": 2})

    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": third, "ltp": 3})

    queued = _queued_symbols(manager)
    assert third in queued
    assert "MDM_OPEN_POSITION_TICK_FORCED_DROP" in "\n".join(r.getMessage() for r in caplog.records)


def test_selected_option_displaces_context_or_far_tick() -> None:
    manager = _mdm()
    selected = "NFO:NIFTY2660923250PE"
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923600CE", "ltp": 1})
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NSE:NIFTY", "ltp": 2})

    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": selected, "ltp": 3})

    queued = _queued_symbols(manager)
    assert selected in queued
    assert len(queued) == 2


def test_context_tick_does_not_displace_selected_or_open_position_tick() -> None:
    manager = _mdm()
    open_symbol = "NFO:NIFTY2660923250CE"
    selected = "NFO:NIFTY2660923250PE"
    manager.set_open_position_symbols([open_symbol])
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": open_symbol, "ltp": 1})
    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": selected, "ltp": 2})

    assert not manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923800CE", "ltp": 3})

    queued = _queued_symbols(manager)
    assert open_symbol in queued
    assert selected in queued


def test_position_manager_exception_does_not_block_tick_enqueue(caplog) -> None:
    class BrokenPositionManager:
        def get_open_positions(self):
            raise RuntimeError("boom")

        def place_order(self):  # must never be called
            raise AssertionError("order placement must not be called")

    manager = _mdm()
    manager.set_position_manager(BrokenPositionManager())
    caplog.set_level(logging.WARNING)

    assert manager._put_priority_tick_nowait(manager._tick_queue, {"symbol": "NFO:NIFTY2660923250CE", "ltp": 1})
    assert "MDM_OPEN_POSITION_PRIORITY_LOOKUP_FAILED" in "\n".join(r.getMessage() for r in caplog.records)


def test_priority_lookup_never_calls_order_or_risk_mutation() -> None:
    class ReadOnlyProbe:
        def __init__(self) -> None:
            self.get_open_positions_called = 0
            self.place_order_called = 0
            self.risk_mutation_called = 0

        def get_open_positions(self):
            self.get_open_positions_called += 1
            return [SimpleNamespace(symbol="NFO:NIFTY2660923250CE")]

        def place_order(self):
            self.place_order_called += 1
            raise AssertionError("must not place orders")

        def update_risk(self):
            self.risk_mutation_called += 1
            raise AssertionError("must not mutate risk")

    probe = ReadOnlyProbe()
    manager = _mdm()
    manager.set_position_manager(probe)

    assert manager._tick_priority("NFO:NIFTY2660923250CE") == (0, "open_position")
    assert probe.get_open_positions_called == 1
    assert probe.place_order_called == 0
    assert probe.risk_mutation_called == 0


def test_set_position_manager_only_used_for_read_only_priority_lookup() -> None:
    from pathlib import Path

    source = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(encoding="utf-8")
    assert "get_open_positions" in source
    assert "place_order(" not in source[source.index("def _open_position_symbol_set"):source.index("def _selected_option_symbol_set")]
    assert "risk_manager" not in source[source.index("def _open_position_symbol_set"):source.index("def _selected_option_symbol_set")]


def _exercise_gap_case(runner: StrategyRunner, symbol: str, *, tick_ts: float, last_bar_age_s: float, caplog, monkeypatch):
    import time

    runner._has_session_candle_gaps = lambda _symbol: True
    runner._session_gap_count[symbol] = 3
    if tick_ts > 0:
        runner._last_tick_time_by_symbol[symbol] = tick_ts
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("RUNNER_CANDLE_GAP_GRACE_SECONDS", "10")
    monkeypatch.setenv("RUNNER_CANDLE_GAP_INTERVAL_SECONDS", "60")
    bar = SimpleNamespace(timestamp=datetime.now(timezone.utc) - timedelta(seconds=last_bar_age_s))
    return runner.update_symbol_hydration(symbol, [bar, bar], {symbol: {"vwap": 1.0, "cum_volume": 1.0}})


def test_candle_gap_no_tick_reason(caplog, monkeypatch) -> None:
    runner = _runner()
    symbol = "NFO:NIFTY2660923300CE"

    _exercise_gap_case(runner, symbol, tick_ts=0.0, last_bar_age_s=30.0, caplog=caplog, monkeypatch=monkeypatch)

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "reason=no_recent_tick_for_gap_assessment" in text


def test_candle_gap_stale_tick_reason_before_bar_grace(caplog, monkeypatch) -> None:
    import time

    runner = _runner()
    symbol = "NFO:NIFTY2660923350CE"

    _exercise_gap_case(runner, symbol, tick_ts=time.time() - 30.0, last_bar_age_s=30.0, caplog=caplog, monkeypatch=monkeypatch)

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "reason=stale_tick_for_gap_assessment" in text


def test_candle_gap_repeated_missing_after_grace_with_stale_tick(caplog, monkeypatch) -> None:
    import time

    runner = _runner()
    symbol = "NFO:NIFTY2660923400CE"

    state = _exercise_gap_case(runner, symbol, tick_ts=time.time() - 30.0, last_bar_age_s=90.0, caplog=caplog, monkeypatch=monkeypatch)

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "reason=repeated_missing_candles" in text
    assert state == SymbolState.DEGRADED


def test_open_position_add_remove_invalidates_priority_cache() -> None:
    manager = _mdm()
    symbol = "NFO:NIFTY2660923450CE"

    assert manager._tick_priority(symbol) != (0, "open_position")
    manager.add_open_position_symbol(symbol)
    assert manager._tick_priority(symbol) == (0, "open_position")
    manager.remove_open_position_symbol(symbol)
    assert manager._tick_priority(symbol) != (0, "open_position")
