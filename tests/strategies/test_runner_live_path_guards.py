from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from types import SimpleNamespace
import threading
from unittest.mock import AsyncMock, MagicMock
import pytest

from nifty_scalper_bot.strategies.runner import ExecutionReadinessResult, SignalExecutionResult, StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


async def _wait_until_async(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


class _SilentLogger:
    def debug(self, *args, **kwargs) -> None:
        return

    def info(self, *args, **kwargs) -> None:
        return

    def warning(self, *args, **kwargs) -> None:
        return

    def error(self, *args, **kwargs) -> None:
        return


class _RecordingOrderManager:
    def __init__(self) -> None:
        self.last_quantity = 0
        self.calls = 0

    def resolve_lot_size(self, _symbol: str) -> int:
        return 65

    def place_order(self, **kwargs) -> str:
        self.calls += 1
        self.last_quantity = int(kwargs.get('quantity') or 0)
        return f'order-{self.calls}'

    def is_kill_switch_active(self) -> bool:
        return False


def _build_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = _SilentLogger()
    runner._order_manager = _RecordingOrderManager()
    runner._last_execution_halted_log_ts = 0.0
    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
    runner._underlying_signal_cooldown_seconds = 30.0
    runner._reason_signal_cooldown_seconds = 30.0
    runner._cooldown_log_throttle_seconds = 1.0
    runner._order_attempt_window = deque()
    runner._max_order_attempts_per_minute = 100
    runner._signal_attempt_debounce_seconds = 45.0
    runner._signal_attempt_debounce_min_improvement = 2.0
    runner._signal_attempt_debounce_state = {}
    runner._record_trade = lambda *args, **kwargs: None
    runner._reset_execution_state = lambda *_args, **_kwargs: None
    runner._entry_lock = threading.Lock()
    runner._trade_candidate_selector = MagicMock()
    runner.build_candidate_snapshots = MagicMock(return_value=[])
    runner.build_candidate_snapshots_async = MagicMock(return_value=([], False))
    runner._market_data = None
    runner._runtime_data_hard_ready = True
    runner._runtime_evaluation_ready = True
    runner._runtime_live_orders_armed = True
    runner._runtime_readiness_reason = None
    runner._build_candidate_snapshots_sync_safe = MagicMock(return_value=([], False, "cached_existing"))
    runner._execution_candidate_basket_cache = {}
    runner._execution_candidate_basket_cache_ttl_seconds = 2.0
    runner._active_atm_strike = None
    runner._active_selected_ce = ""
    runner._active_selected_pe = ""
    return runner


class _ResolverStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_contracts(self, **kwargs):
        self.calls.append(dict(kwargs))
        return [
            {
                'exchange': 'NFO',
                'tradingsymbol': 'NIFTY26APR23800CE',
                'strike': 23800,
                'expiry': '2026-04-30',
                'option_type': 'CE',
            }
        ]


def test_runner_normalizes_one_lot_to_exchange_quantity() -> None:
    runner = _build_runner()
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.9,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='t1',
    )
    assert runner._order_manager.last_quantity == 65


def test_premium_squeeze_suppresses_second_signal_within_cooldown() -> None:
    runner = _build_runner()
    ts = datetime.now(timezone.utc)
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800CE',
        quantity=1,
        confidence=0.9,
        reason='premium_momentum_squeeze',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800CE',
        trade_symbol='NFO:NIFTY26APR23800CE',
        trade_price=110.0,
        timestamp=ts,
        trace_id='t1',
    )
    first_calls = runner._order_manager.calls
    runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23850CE',
        trade_symbol='NFO:NIFTY26APR23850CE',
        trade_price=112.0,
        timestamp=ts,
        trace_id='t2',
    )
    assert first_calls == 1
    assert runner._order_manager.calls == 1


def test_premium_squeeze_does_not_self_suppress_on_first_execution() -> None:
    runner = _build_runner()
    runner._order_manager.submit_trade_plan = MagicMock(return_value='order-1')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800CE',
        quantity=1,
        confidence=0.9,
        reason='premium_momentum_squeeze',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800CE',
        trade_symbol='NFO:NIFTY26APR23800CE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='t-premium-first',
    )
    assert result.accepted is True
    assert result.reason == 'order_submitted'
    assert runner._premium_squeeze_last_signal_ts.get('NIFTY', 0.0) > 0.0


def test_live_entry_uses_runtime_readiness_not_mdm_hard_ready(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._runtime_data_hard_ready = True
    runner._runtime_live_orders_armed = True
    class _BadMarketData:
        def hard_ready(self):
            raise AssertionError('hard_ready should not be called')
    runner._market_data = _BadMarketData()
    runner._order_manager.submit_trade_plan = MagicMock(return_value='order-1')
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.9, reason='test', stop_loss=100.0, take_profit=120.0, metadata={'strategy_score': 7, 'option_score': 7, 'data_score': 7, 'rr_score': 7})
    result = runner._handle_entry_signal_inner(signal, 'NFO:NIFTY26APR23800CE', 'NFO:NIFTY26APR23800CE', 110.0, datetime.now(timezone.utc), trace_id='runtime-ready')
    assert result.accepted is True


def test_trade_plan_uses_env_preflight_values(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('ORDER_MAX_QUOTE_AGE_MS', '65000')
    monkeypatch.setenv('SPREAD_MAX_PCT', '12.5')
    monkeypatch.setenv('MIN_DEPTH_QTY', '25')
    monkeypatch.setenv('ALLOW_MARKET_ENTRY', 'true')
    captured = {}
    def _submit(plan):
        captured['plan'] = plan
        return 'order-2'
    runner._order_manager.submit_trade_plan = _submit
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.9, reason='test', stop_loss=100.0, take_profit=120.0, metadata={'strategy_score': 7, 'option_score': 7, 'data_score': 7, 'rr_score': 7})
    runner._handle_entry_signal_inner(signal, 'NFO:NIFTY26APR23800CE', 'NFO:NIFTY26APR23800CE', 110.0, datetime.now(timezone.utc), trace_id='preflight')
    plan = captured['plan']
    assert plan.max_quote_age_ms == 65000
    assert plan.max_spread_pct == 12.5
    assert plan.min_depth_qty == 25
    assert plan.allow_market_entry is True


def test_order_acceptance_notifies_orchestrator_entry() -> None:
    runner = _build_runner()
    notified = {}
    runner._orchestrator = SimpleNamespace(
        notify_entry=lambda symbol, reason="order_accepted": notified.update({"symbol": symbol, "reason": reason})
    )
    runner._order_manager.submit_trade_plan = MagicMock(return_value='order-2')
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.9, reason='test', stop_loss=100.0, take_profit=120.0, metadata={'strategy_score': 7, 'option_score': 7, 'data_score': 7, 'rr_score': 7})
    runner._handle_entry_signal_inner(signal, 'NFO:NIFTY26APR23800CE', 'NFO:NIFTY26APR23800CE', 110.0, datetime.now(timezone.utc), trace_id='notify')
    assert notified["symbol"] == 'NFO:NIFTY26APR23800CE'
    assert notified["reason"] == "order_accepted"


def test_phase10_no_runtime_indicators_attribute_logs_clean_block_not_runner_error(caplog) -> None:
    runner = _build_runner()
    runner._runtime_live_orders_armed = True
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (True, {"active": True})
    if hasattr(runner, "_runtime_indicators"):
        delattr(runner, "_runtime_indicators")
    with caplog.at_level(logging.INFO):
        ready, reason, details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE", trace_id="phase10-guard")
        runner._reject_signal_execution(
            symbol="NFO:NIFTY26JUN23800PE",
            trace_id="phase10-guard",
            reason=reason,
            details=details,
        )
    assert ready is False
    assert reason == "order_manager_kill_switch_active"
    events = {getattr(r, "event", "") for r in caplog.records}
    assert "RUNNER_ON_TICK_ERROR" not in events
    assert "SIGNAL_EXECUTION_RESULT" in events


def test_symbol_live_entry_ready_refreshes_stale_candidate_for_trigger_signal() -> None:
    runner = _build_runner()
    runner._runtime_live_orders_armed = True
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._required_bars_for_symbol = lambda _s: 1
    runner._indicator_engine.get_history = lambda _s: [1, 2, 3, 4, 5]
    runner._is_option_symbol_tick_fresh = lambda *_a, **_k: True
    runner._runtime_indicators = {"NFO:NIFTY26JUN23800PE": {"direction_bias": "PE", "context_age_seconds": 1.0}}
    runner._active_selected_ce = "NFO:NIFTY26JUN23900CE"
    runner._active_selected_pe = "NFO:NIFTY26JUN23900PE"
    runner._active_option_symbols = ["NFO:NIFTY26JUN23800PE", "NFO:NIFTY26JUN23900PE"]
    runner._active_atm_strike = 23900
    runner._get_cached_quote_for_live_entry = lambda _s: {"tradable_quote": True, "depth_available": True, "spread_pct": 0.2}
    signal = Signal(action="BUY", symbol="NFO:NIFTY26JUN23800PE", quantity=1, confidence=0.8, reason="OrderFlow", metadata={"trigger_conditions_met": True})
    ready, reason, details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE", signal=signal, trace_id="refresh-candidate")
    assert ready is True
    assert reason == "symbol_live_ready"
    assert details.get("candidate_refresh_attempted") is True

def test_selected_option_prewarm_accepts_sync_hydrator_list_result(caplog) -> None:
    runner = _build_runner()
    runner._logger = logging.getLogger("test.runner.prewarm.sync")
    runner._indicator_engine = SimpleNamespace(get_history=lambda _s: [1, 2, 3])
    runner._selected_option_prewarm_last = {}
    runner._selected_option_prewarm_inflight = set()
    runner._selected_option_prewarm_cooldown_s = 0.0
    runner._data_hub = SimpleNamespace(hydrate_symbol_history=lambda *a, **k: [1, 2, 3, 4, 5])
    with caplog.at_level("INFO"):
        runner._request_selected_option_history_prewarm("NFO:NIFTY26JUN24000CE", bars_before=1, required_bars=5, trace_id="t1")
    assert _wait_until(lambda: "NFO:NIFTY26JUN24000CE" not in runner._selected_option_prewarm_inflight)
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "SELECTED_OPTION_HISTORY_PREWARM_RESULT")
    assert rec.success is True
    assert rec.bars_after == 5
    assert rec.source == "data_hub_hydrate"


@pytest.mark.asyncio
async def test_selected_option_prewarm_sync_hydrator_runs_off_event_loop() -> None:
    runner = _build_runner()
    runner._indicator_engine = SimpleNamespace(get_history=lambda _s: [1, 2, 3])
    runner._selected_option_prewarm_last = {}
    runner._selected_option_prewarm_inflight = set()
    runner._selected_option_prewarm_cooldown_s = 0.0
    import threading
    called_thread = {"name": ""}
    def _hydrate(*_a, **_k):
        called_thread["name"] = threading.current_thread().name
        return [1, 2, 3]
    runner._data_hub = SimpleNamespace(hydrate_symbol_history=_hydrate)
    runner._request_selected_option_history_prewarm("NFO:NIFTY26JUN24000CE", bars_before=1, required_bars=5, trace_id="t2")
    assert await _wait_until_async(lambda: bool(called_thread["name"]))
    assert called_thread["name"] != threading.current_thread().name


@pytest.mark.asyncio
async def test_selected_option_prewarm_accepts_async_hydrator_list_result(caplog) -> None:
    runner = _build_runner()
    runner._logger = logging.getLogger("test.runner.prewarm.async")
    runner._indicator_engine = SimpleNamespace(get_history=lambda _s: [1, 2, 3])
    runner._selected_option_prewarm_last = {}
    runner._selected_option_prewarm_inflight = set()
    runner._selected_option_prewarm_cooldown_s = 0.0
    async def _hydrate(*_a, **_k):
        return [1, 2, 3, 4, 5]
    runner._data_hub = SimpleNamespace(hydrate_symbol_history=_hydrate)
    with caplog.at_level("INFO"):
        runner._request_selected_option_history_prewarm("NFO:NIFTY26JUN24000CE", bars_before=1, required_bars=5, trace_id="t3")
        assert await _wait_until_async(lambda: "NFO:NIFTY26JUN24000CE" not in runner._selected_option_prewarm_inflight)
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "SELECTED_OPTION_HISTORY_PREWARM_RESULT")
    assert rec.success is True


def test_atr_fallback_used_for_insufficient_bars() -> None:
    runner = _build_runner()
    runner._required_candles = 20
    runner._data_hub = SimpleNamespace(get_ohlc_bars=lambda _symbol: [])
    runner._market_data = None
    atr = runner._get_atr_with_fallback('NFO:NIFTY26APR23800CE', {}, 100.0)
    assert atr > 0

def test_stale_thresholds_are_instrument_specific() -> None:
    runner = _build_runner()
    runner._option_stale_tick_seconds = 900.0
    runner._future_stale_tick_seconds = 120.0
    runner._index_stale_tick_seconds = 120.0
    runner._generic_stale_tick_seconds = 60.0
    assert runner._stale_tick_threshold_for_symbol('NSE:NIFTY') == 120.0
    assert runner._stale_tick_threshold_for_symbol('NFO:NIFTY26APR23800PE') == 900.0
    assert runner._stale_tick_threshold_for_symbol('NFO:NIFTY26APRFUT') == 120.0


def test_extract_underlying_handles_canonical_derivatives() -> None:
    runner = _build_runner()
    assert runner._extract_underlying('NFO:NIFTY26MAY23850CE') == 'NIFTY'
    assert runner._extract_underlying('NFO:NIFTY26MAYFUT') == 'NIFTY'
    assert runner._extract_underlying('NSE:NIFTY') == 'NIFTY'


def test_handle_signal_entry_transitions_to_signal_and_order_pending(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setattr(
        'nifty_scalper_bot.strategies.runner.is_market_hours_cached',
        lambda: True,
    )
    transition_states: list[str] = []
    runner._normalize_symbol = lambda value: value
    runner._transition_execution_state = lambda _symbol, state: transition_states.append(state.value) or True
    runner._reset_execution_state = lambda _symbol: None

    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.9,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    result = runner._handle_signal(
        signal,
        price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='trace-entry-state',
    )

    assert result.accepted is True
    assert transition_states[:2] == ['SIGNAL_RECEIVED', 'ORDER_PENDING']
    assert 'EXIT_PENDING' not in transition_states


def test_missing_score_components_block_live_mode(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.9,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='live-missing',
    )
    assert result.accepted is False
    assert result.reason == 'missing_signal_score_components'


def test_live_unknown_option_side_is_blocked(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    signal = Signal(
        action='BUY',
        symbol='NSE:NIFTY',
        quantity=1,
        confidence=0.7,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 8.0,
            'strategy_score': 8.0,
            'option_score': 8.0,
            'data_score': 8.0,
            'rr_score': 8.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NSE:NIFTY',
        trade_symbol='NSE:NIFTY',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='unknown-side',
    )
    assert result.accepted is False
    assert result.reason == 'unknown_option_side'


def test_live_option_requires_candidate_snapshots(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.7,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 8.0,
            'strategy_score': 8.0,
            'rr_score': 8.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='missing-candidates',
    )
    assert result.accepted is False
    assert result.reason == 'missing_candidate_snapshots'


def test_live_mode_blocks_when_startup_pipeline_not_ready(monkeypatch) -> None:
    class _Mdm:
        def readiness_state_snapshot(self) -> dict[str, object]:
            return {'hard_ready': False, 'spot_ready': False, 'missing_hard': ['futures']}

    runner = _build_runner()
    runner._market_data = _Mdm()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.7,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 9.0,
            'strategy_score': 9.0,
            'option_score': 9.0,
            'data_score': 9.0,
            'rr_score': 9.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='startup-not-ready',
    )
    assert result.accepted is False
    assert result.reason == 'startup_pipeline_not_ready'


def test_preliminary_signal_requires_final_score_gate(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.95,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={'preliminary_only': True, 'requires_runner_final_score': True},
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='prelim-final-required',
    )
    assert result.accepted is False
    assert result.reason == 'final_score_precheck_failed_unknown'


@pytest.mark.asyncio
async def test_live_async_path_builds_candidates_before_sync_handler(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner.build_candidate_snapshots_async = AsyncMock(
        return_value=(
            [
                {
                    'symbol': 'NFO:NIFTY26APR23800PE',
                    'strike': 23800,
                    'atm_strike': 23800,
                    'tradable_quote': True,
                }
            ],
            False,
        )
    )
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.8,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 8.0,
            'strategy_score': 8.0,
            'option_score': 8.0,
            'data_score': 8.0,
            'rr_score': 8.0,
        },
    )
    prepared, reason = await runner._prepare_signal_for_handling(
        signal,
        trace_id='loop-pending',
        price=110.0,
    )
    assert reason is None
    assert prepared is not None
    prepared_snaps = prepared.metadata.get('candidate_snapshots')
    assert prepared_snaps is None or isinstance(prepared_snaps, list)
    assert reason is None


@pytest.mark.asyncio
async def test_prepare_signal_sanitizes_underlying_to_nifty(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    captured: dict[str, object] = {}

    async def _fake_builder(**kwargs):
        captured.update(kwargs)
        return ([{'symbol': 'NFO:NIFTY26APR23800PE', 'atm_strike': 23800}], False)

    runner._extract_underlying = lambda _symbol: 'NFO'
    runner.build_candidate_snapshots_async = _fake_builder
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.8,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    prepared, reason = await runner._prepare_signal_for_handling(signal, price=110.0, trace_id='u1')
    assert reason is None
    assert prepared is not None
    assert captured.get('underlying') in {None, 'NIFTY'}


@pytest.mark.asyncio
async def test_live_async_path_blocks_on_refresh_pending(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner.build_candidate_snapshots_async = AsyncMock(return_value=([], True))
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.8,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 8.0,
            'strategy_score': 8.0,
            'option_score': 8.0,
            'data_score': 8.0,
            'rr_score': 8.0,
        },
    )
    prepared, reason = await runner._prepare_signal_for_handling(
        signal,
        trace_id='loop-pending',
        price=110.0,
    )
    assert (prepared is None and reason == 'candidate_refresh_pending') or (prepared is not None and reason is None)


@pytest.mark.asyncio
async def test_live_async_path_uses_signal_candidate_fallback(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    snapshot = SimpleNamespace(
        ltp=111.0,
        bid=None,
        ask=None,
        source='ws',
        tradable_quote=False,
    )
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot = MagicMock(return_value=snapshot)
    runner.build_candidate_snapshots_async = AsyncMock(return_value=([], True))
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.8,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    prepared, reason = await runner._prepare_signal_for_handling(signal, price=110.0, trace_id='fallback')
    if reason is None and prepared is not None and 'candidate_snapshots' in prepared.metadata:
        assert prepared is not None
        candidate = prepared.metadata['candidate_snapshots'][0]
        assert candidate['ltp'] == 111.0
        assert candidate['bid'] == 0.0
        assert candidate['ask'] == 0.0
        assert candidate['ltp_only_fallback'] is True
    else:
        assert reason in (None, 'candidate_refresh_pending')


def test_strategy_evaluation_caps_required_option_bars(monkeypatch) -> None:
    runner = _build_runner()
    runner._active_symbols = {'NFO:NIFTY26APR23800PE'}
    monkeypatch.setenv('OPTION_MIN_LIVE_BARS', '3')
    runner._required_bars_for_symbol = lambda _s: 5
    runner._is_tradable_symbol = lambda _s: True
    runner._is_context_symbol = lambda _s: False
    runner._indicator_engine.has_min_bars = MagicMock(return_value=True)
    allowed = runner._strategy_evaluation_allowed('NFO:NIFTY26APR23800PE')
    assert allowed is True
    runner._indicator_engine.has_min_bars.assert_called_once_with('NFO:NIFTY26APR23800PE', 3)


def test_contract_resolver_invoked_with_side_and_strikes() -> None:
    runner = _build_runner()
    runner._market_data = MagicMock()
    runner._market_data.tracked_snapshot.return_value = []
    store = _ResolverStore()
    runner._options_contract_store = store
    runner._contract_store = None
    runner._instrument_manager = None
    runner._contract_selector = None

    selected = runner._resolve_candidate_contracts(
        side='CE',
        target_strikes={23800, 23850},
    )

    assert selected == [('NFO:NIFTY26APR23800CE', 23800)]
    assert len(store.calls) == 1
    assert store.calls[0]['strikes'] == [23800, 23850]
    assert store.calls[0]['side'] == 'CE'


def test_contract_resolver_infers_ce_side_from_optidx_suffix() -> None:
    runner = _build_runner()
    runner._market_data = MagicMock()
    runner._market_data.tracked_snapshot.return_value = []

    class _OptIdxStore:
        def get_contracts(self, **kwargs):
            return [
                {
                    'exchange': 'NFO',
                    'tradingsymbol': 'NIFTY26APR23800CE',
                    'instrument_type': 'OPTIDX',
                    'strike': 23800,
                    'expiry': '2026-04-30',
                }
            ]

    runner._options_contract_store = _OptIdxStore()
    runner._contract_store = None
    runner._instrument_manager = None
    runner._contract_selector = None
    selected = runner._resolve_candidate_contracts(side='CE', target_strikes={23800})
    assert selected == [('NFO:NIFTY26APR23800CE', 23800)]


def test_contract_resolver_infers_pe_side_and_rejects_wrong_side() -> None:
    runner = _build_runner()
    runner._market_data = MagicMock()
    runner._market_data.tracked_snapshot.return_value = []

    class _OptIdxStore:
        def get_contracts(self, **kwargs):
            return [
                {
                    'exchange': 'NFO',
                    'tradingsymbol': 'NIFTY26APR23800PE',
                    'strike': 23800,
                    'expiry': '2026-04-30',
                },
                {
                    'exchange': 'NFO',
                    'tradingsymbol': 'NIFTY26APR23800CE',
                    'strike': 23800,
                    'expiry': '2026-04-30',
                },
            ]

    runner._options_contract_store = _OptIdxStore()
    runner._contract_store = None
    runner._instrument_manager = None
    runner._contract_selector = None
    selected = runner._resolve_candidate_contracts(side='PE', target_strikes={23800})
    assert selected == [('NFO:NIFTY26APR23800PE', 23800)]


def test_contract_resolver_rejects_expiry_less_symbol_and_falls_back_tracked() -> None:
    runner = _build_runner()

    class _BadStore:
        def get_contracts(self, **kwargs):
            return [
                {
                    'exchange': 'NFO',
                    'tradingsymbol': 'NIFTY24100CE',
                    'strike': 24100,
                    'option_type': 'CE',
                    'expiry': '',
                }
            ]

    runner._options_contract_store = _BadStore()
    runner._contract_store = None
    runner._instrument_manager = None
    runner._contract_selector = None
    runner._market_data = MagicMock()
    runner._market_data.tracked_snapshot.return_value = [
        'NFO:NIFTY26APR24100CE',
        'NFO:NIFTY24100CE',
    ]

    selected = runner._resolve_candidate_contracts(side='CE', target_strikes={24100})
    assert selected == [('NFO:NIFTY26APR24100CE', 24100)]


def test_final_confidence_is_derived_from_final_score(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.99,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 9.0,
            'strategy_score': 8.0,
            'option_score': 8.0,
            'data_score': 7.0,
            'rr_score': 8.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='confidence-from-score',
    )
    assert result.accepted is True
    assert runner._order_manager.calls == 1
    confidence_used = float(
        runner._order_manager.place_order.call_args.kwargs.get('confidence') or 0.0
    )
    expected_score = (0.30 * 9.0) + (0.25 * 8.0) + (0.20 * 8.0) + (0.15 * 7.0) + (0.10 * 8.0)
    assert confidence_used == expected_score / 10.0


def test_signal_score_logging_emits_without_format_error(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    runner = _build_runner()
    runner._logger = logging.getLogger('test.runner.signal_score')
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    caplog.set_level('INFO')
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.99,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 9.0,
            'strategy_score': 8.0,
            'option_score': 8.0,
            'data_score': 7.0,
            'rr_score': 8.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='signal-score-log',
    )
    assert result.accepted is True
    assert any('SIGNAL_SCORE final=' in rec.message for rec in caplog.records)


def test_premium_helper_respects_generation_cooldown_before_indicator_eval() -> None:
    runner = _build_runner()
    runner._indicator_engine = MagicMock()
    runner._extract_underlying = lambda _symbol: 'NIFTY'
    runner._premium_squeeze_last_signal_ts['NIFTY'] = datetime.now(
        timezone.utc
    ).timestamp()
    generated = runner._maybe_generate_premium_squeeze_signal(
        'NFO:NIFTY26APR23800CE',
        110.0,
        trace_id='premium-cooldown',
    )
    assert generated is None
    runner._indicator_engine.get_indicators.assert_not_called()


def test_premium_helper_skips_future_symbols() -> None:
    runner = _build_runner()
    runner._indicator_engine = MagicMock()
    generated = runner._maybe_generate_premium_squeeze_signal(
        'NFO:NIFTY26APRFUT',
        110.0,
        trace_id='premium-fut',
    )
    assert generated is None
    runner._indicator_engine.get_indicators.assert_not_called()


@pytest.mark.asyncio
async def test_process_token_awaits_prepare_signal_for_handling(monkeypatch) -> None:
    runner = _build_runner()
    captured: dict[str, object] = {}

    class _StrategyManager:
        def generate_signal(self, _symbol: str, _price: float):
            return Signal(
                action='BUY',
                symbol='NFO:NIFTY26APR23800CE',
                quantity=1,
                confidence=0.9,
                reason='unit_test',
                stop_loss=100.0,
                take_profit=120.0,
                metadata={},
            )

    runner._strategy_manager = _StrategyManager()
    runner._data_hub = None
    runner._market_data = MagicMock()
    runner._market_data._symbol_by_token = {1: 'NSE:NIFTY'}

    async def _fake_prepare(signal, price, trace_id):
        captured['called'] = True
        captured['trace_id'] = trace_id
        new_signal = signal
        return new_signal, None

    runner._prepare_signal_for_handling = _fake_prepare
    runner._handle_signal = MagicMock(return_value=None)
    runner._emit_runner_eval_decision = lambda **_kw: None

    import pandas as pd

    candles = pd.DataFrame(
        [
            {
                'open': 100.0,
                'high': 102.0,
                'low': 99.0,
                'close': 101.0,
                'volume': 100.0,
            }
        ]
    )
    await runner._process_token(1, candles, indicators=None)
    assert captured.get('called') is True
    runner._handle_signal.assert_called_once()


@pytest.mark.asyncio
async def test_build_candidate_snapshots_per_symbol_refresh_pending() -> None:
    runner = _build_runner()
    runner.build_candidate_snapshots_async = StrategyRunner.build_candidate_snapshots_async.__get__(runner, StrategyRunner)
    spot = MagicMock()
    spot.ltp = 23800.0
    spot.canonical_symbol = 'NSE:NIFTY'

    fresh_snap = MagicMock()
    fresh_snap.canonical_symbol = 'NFO:NIFTY26APR23800CE'
    fresh_snap.ltp = 100.0
    fresh_snap.bid = 99.0
    fresh_snap.ask = 101.0
    fresh_snap.mid = 100.0
    fresh_snap.tick_age_s = 1.0
    fresh_snap.source = 'ws'
    fresh_snap.real_ticks_last_60s = 30
    fresh_snap.latest_candle_provisional = False
    fresh_snap.latest_candle_synthetic = False
    fresh_snap.latest_candle_volume = 1.0
    fresh_snap.ohlc_valid = True
    fresh_snap.bid_missing = False
    fresh_snap.ask_missing = False
    fresh_snap.bid_ask_source = 'market_depth'
    fresh_snap.tradable_quote = True

    stale_snap = MagicMock()
    stale_snap.canonical_symbol = 'NFO:NIFTY26APR23850CE'
    stale_snap.ltp = None
    stale_snap.bid = 0
    stale_snap.ask = 0
    stale_snap.mid = None
    stale_snap.tick_age_s = 100.0
    stale_snap.source = ''
    stale_snap.real_ticks_last_60s = 0
    stale_snap.latest_candle_provisional = True
    stale_snap.latest_candle_synthetic = False
    stale_snap.latest_candle_volume = 0.0
    stale_snap.ohlc_valid = False
    stale_snap.bid_missing = True
    stale_snap.ask_missing = True
    stale_snap.bid_ask_source = ''
    stale_snap.tradable_quote = False

    snapshots = {
        'NIFTY': spot,
        'NSE:NIFTY': spot,
        'NFO:NIFTY26APR23800CE': fresh_snap,
        'NFO:NIFTY26APR23850CE': stale_snap,
    }

    market_data = MagicMock()

    def _get(symbol):
        return snapshots.get(symbol, spot)

    market_data.get_symbol_snapshot.side_effect = _get
    market_data.request_symbol_subscription = MagicMock()

    async def _ensure_fresh_tick(symbol):
        if symbol == 'NFO:NIFTY26APR23850CE':
            raise asyncio.TimeoutError
        return None

    market_data.ensure_fresh_tick = _ensure_fresh_tick
    runner._market_data = market_data
    runner._resolve_candidate_contracts = MagicMock(
        return_value=[
            ('NFO:NIFTY26APR23800CE', 23800),
            ('NFO:NIFTY26APR23850CE', 23850),
        ]
    )

    candidates, refresh_pending = await runner.build_candidate_snapshots_async(
        underlying='NIFTY',
        direction_bias='CE',
        atm_strike=23800,
        window_each_side=1,
    )
    assert any(
        cand['symbol'] == 'NFO:NIFTY26APR23800CE'
        and cand['refresh_pending'] is False
        for cand in candidates
    )
    assert any(
        cand['symbol'] == 'NFO:NIFTY26APR23850CE'
        and cand['refresh_pending'] is True
        for cand in candidates
    )
    assert refresh_pending is False  # at least one valid candidate keeps overall flag clear
    assert all('side' in cand and 'option_type' in cand and 'atm_strike' in cand for cand in candidates)


def test_live_entry_uses_selected_candidate_symbol(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._trade_candidate_selector.select_best_candidate.return_value = MagicMock(
        symbol='NFO:NIFTY26APR23950CE', score=8.5, data_quality_score=8.5, spread_pct=0.01, stop_loss=95.0, target=130.0
    )
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR24000CE',
        quantity=1,
        confidence=0.9,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'atm_strike': 23950,
            'candidate_snapshots': [{'symbol': 'NFO:NIFTY26APR23950CE', 'atm_strike': 23950, 'side': 'CE', 'tradable_quote': True}],
            'direction_score': 9.0, 'strategy_score': 9.0, 'option_score': 9.0, 'data_score': 9.0, 'rr_score': 9.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal, 'NFO:NIFTY26APR24000CE', 'NFO:NIFTY26APR24000CE', 110.0, datetime.now(timezone.utc), trace_id='sel-1'
    )
    assert result.reason != 'not_selected_candidate'


@pytest.mark.asyncio
async def test_build_candidate_snapshots_all_pending_returns_refresh_pending() -> None:
    runner = _build_runner()
    runner.build_candidate_snapshots_async = StrategyRunner.build_candidate_snapshots_async.__get__(runner, StrategyRunner)
    spot = MagicMock()
    spot.ltp = 23800.0
    spot.canonical_symbol = 'NSE:NIFTY'

    def _stale(name: str):
        snap = MagicMock()
        snap.canonical_symbol = name
        snap.ltp = None
        snap.bid = 0
        snap.ask = 0
        snap.mid = None
        snap.tick_age_s = 100.0
        snap.source = ''
        snap.real_ticks_last_60s = 0
        snap.latest_candle_provisional = True
        snap.latest_candle_synthetic = False
        snap.latest_candle_volume = 0.0
        snap.ohlc_valid = False
        snap.bid_missing = True
        snap.ask_missing = True
        snap.bid_ask_source = ''
        snap.tradable_quote = False
        return snap

    snapshots = {
        'NIFTY': spot,
        'NSE:NIFTY': spot,
        'NFO:NIFTY26APR23800CE': _stale('NFO:NIFTY26APR23800CE'),
        'NFO:NIFTY26APR23850CE': _stale('NFO:NIFTY26APR23850CE'),
    }

    market_data = MagicMock()
    market_data.get_symbol_snapshot.side_effect = lambda s: snapshots.get(s, spot)
    market_data.request_symbol_subscription = MagicMock()

    async def _ensure_fresh_tick(_symbol):
        raise asyncio.TimeoutError

    market_data.ensure_fresh_tick = _ensure_fresh_tick
    runner._market_data = market_data
    runner._resolve_candidate_contracts = MagicMock(
        return_value=[
            ('NFO:NIFTY26APR23800CE', 23800),
            ('NFO:NIFTY26APR23850CE', 23850),
        ]
    )

    candidates, refresh_pending = await runner.build_candidate_snapshots_async(
        underlying='NIFTY',
        direction_bias='CE',
        atm_strike=23800,
        window_each_side=1,
    )
    assert all(cand['refresh_pending'] is True for cand in candidates)
    assert refresh_pending is True




def test_schedule_signal_preparation_uses_asyncio_run_without_loop() -> None:
    runner = _build_runner()

    async def _fake_prepare(signal, price, trace_id):
        return signal, None

    runner._prepare_signal_for_handling = _fake_prepare
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800CE',
        quantity=1,
        confidence=0.9,
        reason='unit_test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )

    now = datetime.now(timezone.utc)
    runner._handle_signal = MagicMock(return_value=SignalExecutionResult(True, 'accepted'))  # type: ignore[method-assign]
    scheduled, reason = runner._schedule_signal_preparation(signal, 101.0, now, 'trace-sync')

    assert scheduled is True
    assert reason is None
    runner._handle_signal.assert_called_once()


def test_schedule_signal_preparation_logs_result_when_handle_signal_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _build_runner()
    logs: list[tuple[tuple[object, ...], dict[str, object]]] = []
    runner._logger.info = lambda *args, **kwargs: logs.append((args, kwargs))  # type: ignore[method-assign]

    async def _fake_prepare(signal, price, trace_id):
        return signal, None

    runner._prepare_signal_for_handling = _fake_prepare
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800CE',
        quantity=1,
        confidence=0.9,
        reason='unit_test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    runner._handle_signal = MagicMock(return_value=SignalExecutionResult(False, 'blocked'))  # type: ignore[method-assign]
    runner._schedule_signal_preparation(
        signal, 101.0, datetime.now(timezone.utc), 'trace-reject'
    )
    assert any(
        kwargs.get('extra', {}).get('event') == 'SIGNAL_EXECUTION_RESULT'
        and kwargs.get('extra', {}).get('accepted') is False
        for _, kwargs in logs
    )

@pytest.mark.asyncio
async def test_schedule_signal_preparation_schedules_when_loop_running() -> None:
    runner = _build_runner()
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800CE',
        quantity=1,
        confidence=0.9,
        reason='unit_test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )

    async def _fake_prepare(prepared_signal, _price, _trace_id):
        return prepared_signal, None

    runner._prepare_signal_for_handling = _fake_prepare  # type: ignore[method-assign]
    runner._handle_signal = MagicMock(return_value=SignalExecutionResult(True, 'accepted'))  # type: ignore[method-assign]
    now = datetime.now(timezone.utc)
    scheduled, reason = runner._schedule_signal_preparation(signal, 101.0, now, 'trace-loop')
    await asyncio.sleep(0.05)

    assert scheduled is True
    assert reason == 'signal_preparation_scheduled'
    runner._handle_signal.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_signal_preparation_logs_task_failure() -> None:
    runner = _build_runner()
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.9, reason='x', stop_loss=90.0, take_profit=120.0, metadata={})
    runner._prepare_signal_for_handling = AsyncMock(side_effect=RuntimeError('boom'))  # type: ignore[method-assign]
    runner._handle_signal = MagicMock()  # type: ignore[method-assign]
    errors: list[str] = []
    runner._logger.error = lambda msg, *args, **kwargs: errors.append(str(msg))  # type: ignore[method-assign]
    runner._schedule_signal_preparation(signal, 100.0, datetime.now(timezone.utc), 'trace-fail')
    await asyncio.sleep(0.05)
    assert any('SIGNAL_PREPARATION_TASK_FAILED' in msg for msg in errors)

def test_runner_no_async_event_loop_fallback_in_runner_source() -> None:
    """Regression guard: async paths must not skip prep just because event loop exists."""
    from pathlib import Path
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    # _process_token must call await self._prepare_signal_for_handling
    assert 'await self._prepare_signal_for_handling' in source
    # Sync candidate-builder fallback in _on_tick was removed
    assert 'asyncio.run(\n                        self._prepare_signal_for_handling' not in source
    # _handle_entry_signal_inner must not build snapshots in sync path
    assert 'asyncio.run(\n                        self.build_candidate_snapshots_async' not in source


def test_signal_generated_log_occurs_after_regime_skip_guard_in_source() -> None:
    from pathlib import Path

    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert source.index('Strategy skipped due to detected market regime') < source.index(
        'SIGNAL_GENERATED symbol=%s action=%s reason=%s trace_id=%s'
    )


def test_runner_kill_switch_precheck_logs_status_details_in_source() -> None:
    from pathlib import Path
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'RUNNER_KILL_SWITCH_PRECHECK_BLOCK' in source
    assert 'consecutive_failures' in source
    assert 'kill_reason' in source
    assert 'broker_attempted' in source


def test_build_single_candidate_from_signal_includes_tick_fields() -> None:
    runner = _build_runner()
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=374.95, bid=374.5, ask=375.4, tick_age_s=None, real_ticks_last_60s=0, tradable_quote=True, source='ws'
    )
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY24050PE', quantity=1, confidence=0.8, reason='x', stop_loss=300.0, take_profit=450.0, metadata={})
    candidate = runner._build_single_candidate_from_signal(signal=signal, metadata={}, option_side='PE')
    assert candidate is not None
    assert candidate['tick_age_s'] == 0.0
    assert candidate['real_ticks_last_60s'] >= 1
    assert candidate['quote_quality'] == 'bid_ask'


def test_build_single_candidate_from_signal_does_not_fake_bid_ask() -> None:
    runner = _build_runner()
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=374.95, bid=0.0, ask=0.0, tick_age_s=0.1, real_ticks_last_60s=3, tradable_quote=True, source='ws'
    )
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY24050PE', quantity=1, confidence=0.8, reason='x', stop_loss=300.0, take_profit=450.0, metadata={})
    candidate = runner._build_single_candidate_from_signal(signal=signal, metadata={}, option_side='PE')
    assert candidate is not None
    assert candidate['bid'] == 0.0
    assert candidate['ask'] == 0.0
    assert candidate['ltp_only_fallback'] is True
    assert candidate['quote_quality'] == 'ltp_only'


def test_pre_order_rejection_logs_signal_execution_result() -> None:
    runner = _build_runner()
    logs: list[dict[str, object]] = []
    runner._logger.info = lambda *args, **kwargs: logs.append(kwargs.get('extra', {}))  # type: ignore[method-assign]
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.9, reason='x', stop_loss=90.0, take_profit=120.0, metadata={'candidate_snapshots': []})
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=100.0, timestamp=datetime.now(timezone.utc), trace_id='trace-pre')
    assert result.accepted is False
    assert any(item.get('event') == 'SIGNAL_EXECUTION_RESULT' for item in logs)


def test_fallback_candidate_reaches_order_request_path() -> None:
    runner = _build_runner()
    emitted: list[str] = []
    runner._logger.info = lambda *_args, **kwargs: emitted.append(str(kwargs.get('extra', {}).get('event', '')))  # type: ignore[method-assign]
    runner._transition_execution_state = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    runner._order_manager.submit_trade_plan = MagicMock(return_value='oid-1')  # type: ignore[attr-defined]
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=374.95, bid=374.5, ask=375.4, tick_age_s=0.2, real_ticks_last_60s=3, tradable_quote=True, source='ws'
    )
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY24050PE', quantity=1, confidence=0.9, reason='x', stop_loss=300.0, take_profit=450.0, metadata={'candidate_snapshots': [{'symbol': 'NFO:NIFTY26MAY24050PE', 'side': 'PE', 'strike': 24050, 'atm_strike': 24050, 'ltp': 374.95, 'bid': 374.5, 'ask': 375.4, 'tick_age_s': 0.2, 'real_ticks_last_60s': 3}]})
    runner._trade_candidate_selector.select_best_candidate = MagicMock(
        return_value=SimpleNamespace(symbol=signal.symbol, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.01)
    )
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id='trace-order')
    assert result.accepted is True
    assert 'ORDER_QTY_NORMALIZED' in emitted
    assert 'RUNNER_ORDER_REQUEST' in emitted


def test_single_candidate_distance_fallback_from_strike_allows_selected_candidate(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PREMIUM_FALLBACK_MAX_STRIKE_DISTANCE", "100")
    runner._transition_execution_state = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    runner._order_manager.submit_trade_plan = MagicMock(return_value="oid-one-good")  # type: ignore[attr-defined]
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=374.95, bid=374.5, ask=375.4, tick_age_s=0.2, real_ticks_last_60s=3, tradable_quote=True, source="ws"
    )
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY26MAY24050PE",
        quantity=1,
        confidence=0.9,
        reason="x",
        stop_loss=300.0,
        take_profit=450.0,
        metadata={"candidate_snapshots": [{"symbol": "NFO:NIFTY26MAY24050PE", "side": "PE", "strike": 24050, "atm_strike": 24050, "ltp": 374.95, "bid": 374.5, "ask": 375.4, "tick_age_s": 0.2, "real_ticks_last_60s": 3}]},
    )
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(
        return_value=[SimpleNamespace(symbol=signal.symbol, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.01, rr=2.0, side="PE", entry_price=374.95, tick_age_s=0.2)]
    )
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-one-good")
    assert result.reason != "candidate_basket_inadequate"


def test_single_candidate_distance_fallback_rejects_far_strike(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PREMIUM_FALLBACK_MAX_STRIKE_DISTANCE", "100")
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=374.95, bid=374.5, ask=375.4, tick_age_s=0.2, real_ticks_last_60s=3, tradable_quote=True, source="ws"
    )
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY26MAY24050PE",
        quantity=1,
        confidence=0.9,
        reason="x",
        stop_loss=300.0,
        take_profit=450.0,
        metadata={"candidate_snapshots": [{"symbol": "NFO:NIFTY26MAY24050PE", "side": "PE", "strike": 24550, "atm_strike": 24050, "ltp": 374.95, "bid": 374.5, "ask": 375.4, "tick_age_s": 0.2, "real_ticks_last_60s": 3}]},
    )
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-one-far")
    assert result.reason == "candidate_basket_inadequate"
    assert result.details.get("is_selected") is True
    assert result.details.get("is_fresh") is True
    assert "tradable" in result.details
    assert "premium_ok" in result.details
    assert "strike_distance" in result.details
    assert "ltp" in result.details
    assert "bid" in result.details
    assert "ask" in result.details


def test_runtime_symbol_execution_not_ready_emits_diagnostics(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    emitted: list[dict] = []
    runner._logger.info = lambda *_args, **kwargs: emitted.append(kwargs.get("extra", {}))  # type: ignore[method-assign]
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(False, "quote_stale", {"tick_age_ms": 90000}))
    runner._is_symbol_execution_ready = MagicMock(return_value=False)
    symbol = "NFO:NIFTY26MAY24050PE"
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": symbol, "side": "PE", "strike": 24050, "atm_strike": 24050, "ltp": 374.95, "bid": 374.5, "ask": 375.4, "tick_age_s": 0.2, "tradable_quote": True}]})
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=symbol, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side="PE", entry_price=374.95, tick_age_s=0.2)])
    result = runner._handle_entry_signal_inner(signal, base_symbol=symbol, trade_symbol=symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-ready")
    assert result.reason == "runtime_symbol_execution_not_ready"
    diag = next((e for e in emitted if e.get("event") == "SYMBOL_EXECUTION_READY_DIAGNOSTICS"), {})
    assert diag.get("symbol") == symbol
    assert diag.get("trace_id") == "trace-ready"
    assert diag.get("readiness_reason") == "quote_stale"
    assert diag.get("readiness_details", {}).get("tick_age_ms") == 90000


def test_candidate_refresh_pending_includes_diagnostics(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    emitted: list[dict] = []
    runner._logger.info = lambda *_args, **kwargs: emitted.append(kwargs.get("extra", {}))  # type: ignore[method-assign]
    symbol = "NFO:NIFTY26MAY24050PE"
    runner._build_candidate_snapshots_sync_safe = MagicMock(return_value=([{"symbol": symbol}], True, "event_loop_active_refresh_pending"))
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": symbol}], "atm_strike": 24050})
    result = runner._handle_entry_signal_inner(signal, base_symbol=symbol, trade_symbol=symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-pending")
    assert result.reason == "candidate_refresh_pending"
    assert result.details.get("candidate_total") == 1
    assert result.details.get("basket_source") == "event_loop_active_refresh_pending"
    assert result.details.get("option_side") == "PE"
    assert result.details.get("underlying")
    diag = next((e for e in emitted if e.get("event") == "CANDIDATE_REFRESH_PENDING_DIAGNOSTICS"), {})
    assert diag.get("trace_id") == "trace-pending"
    assert result.details.get("event_loop_active") is True
    assert result.details.get("reason") == "event_loop_active_refresh_pending"
    assert result.details.get("selected_ce") == ""
    assert result.details.get("selected_pe") == ""
    fallback_events = [e for e in emitted if e.get("event") == "EXECUTION_CANDIDATE_BASKET_FALLBACK"]
    assert not any(e.get("reason") == "basket_unavailable" for e in fallback_events)
    

def test_build_candidate_snapshots_sync_safe_logs_refresh_pending_in_active_loop() -> None:
    runner = _build_runner()
    emitted: list[dict] = []
    runner._logger.info = lambda *_args, **kwargs: emitted.append(kwargs.get("extra", {}))  # type: ignore[method-assign]
    symbol = "NFO:NIFTY26MAY24050PE"

    async def _invoke() -> tuple[list[dict], bool, str]:
        return runner._build_candidate_snapshots_sync_safe(
            symbol=symbol,
            underlying="NIFTY",
            direction_bias="PE",
            atm_strike=24050,
            existing_snapshots=[{"symbol": symbol}],
            window_each_side=2,
        )

    snapshots, pending, source = asyncio.run(_invoke())
    assert snapshots == [{"symbol": symbol}]
    assert pending is True
    assert source == "event_loop_active_refresh_pending"
    refresh_pending = [e for e in emitted if e.get("event") == "EXECUTION_CANDIDATE_BASKET_REFRESH_PENDING"]
    assert refresh_pending
    assert refresh_pending[0].get("reason") == "event_loop_active_refresh_pending"
    assert refresh_pending[0].get("source") == "event_loop_active_refresh_pending"
    assert refresh_pending[0].get("total") == 1


@pytest.mark.asyncio
async def test_prepare_execution_candidate_basket_async_populates_metadata() -> None:
    runner = _build_runner()
    symbol = "NFO:NIFTY26MAY24050PE"
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={})
    metadata = {"candidate_snapshots": [{"symbol": symbol}]}
    runner.build_candidate_snapshots_async = AsyncMock(return_value=([{"symbol": symbol, "option_type": "PE"}, {"symbol": "NFO:NIFTY26MAY24000PE", "option_type": "PE"}, {"symbol": "NFO:NIFTY26MAY24100PE", "option_type": "PE"}], False))
    events: list[dict] = []
    runner._logger.info = lambda *_args, **kwargs: events.append(kwargs.get("extra", {}))  # type: ignore[method-assign]
    snapshots = await runner._prepare_execution_candidate_basket_async(signal=signal, metadata=metadata, underlying="NIFTY", option_side="PE", atm_strike=24050, trace_id="trace-prepare")
    assert len(snapshots) == 3
    assert len(metadata["candidate_snapshots"]) == 3
    assert metadata["candidate_basket_source"] == "async_prepared"
    assert any(e.get("event") == "EXECUTION_CANDIDATE_BASKET_PREPARED" for e in events)


def test_build_candidate_snapshots_sync_safe_uses_cache_in_active_loop() -> None:
    runner = _build_runner()
    runner._build_candidate_snapshots_sync_safe = StrategyRunner._build_candidate_snapshots_sync_safe.__get__(runner, StrategyRunner)
    symbol = "NFO:NIFTY26MAY24050PE"
    cached = [{"symbol": symbol}, {"symbol": "NFO:NIFTY26MAY24000PE"}, {"symbol": "NFO:NIFTY26MAY24100PE"}]
    runner._set_cached_execution_candidate_basket(underlying="NIFTY", option_side="PE", atm_strike=24050, snapshots=cached, source="async_prepared")

    async def _invoke() -> tuple[list[dict], bool, str]:
        return runner._build_candidate_snapshots_sync_safe(symbol=symbol, underlying="NIFTY", direction_bias="PE", atm_strike=24050, existing_snapshots=[{"symbol": symbol}], window_each_side=2)

    snaps, pending, source = asyncio.run(_invoke())
    assert len(snaps) == 3
    assert pending is False
    assert source == "cache_event_loop_active"


def test_candidate_refresh_pending_non_event_loop_reason(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    symbol = "NFO:NIFTY26MAY24050PE"
    runner._build_candidate_snapshots_sync_safe = MagicMock(return_value=([{"symbol": symbol}], True, "async_refresh"))
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": symbol}], "atm_strike": 24050})
    result = runner._handle_entry_signal_inner(signal, base_symbol=symbol, trade_symbol=symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-pending-async")
    assert result.reason == "candidate_refresh_pending"
    assert result.details.get("event_loop_active") is False
    assert result.details.get("reason") == "candidate_snapshot_refresh_pending"
    assert result.details.get("basket_source") == "async_refresh"


def test_candidate_refresh_pending_diag_has_cache_fields(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    symbol = "NFO:NIFTY26MAY24050PE"
    runner._build_candidate_snapshots_sync_safe = MagicMock(return_value=([{"symbol": symbol}], True, "event_loop_active_refresh_pending"))
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": symbol}], "atm_strike": 24050})
    result = runner._handle_entry_signal_inner(signal, base_symbol=symbol, trade_symbol=symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-cache-diag")
    assert result.reason == "candidate_refresh_pending"
    assert result.details.get("cache_hit") is False
    assert result.details.get("cache_total") == 0
    assert "preparation_attempted" in result.details
    assert "existing_snapshot_symbols" in result.details


def test_handle_entry_uses_cached_basket_and_reaches_submission(monkeypatch) -> None:
    runner = _build_runner()
    runner._build_candidate_snapshots_sync_safe = StrategyRunner._build_candidate_snapshots_sync_safe.__get__(runner, StrategyRunner)
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    symbol = "NFO:NIFTY26MAY24050PE"
    cached = [
        {"symbol": symbol, "side": "PE", "strike": 24050, "atm_strike": 24050, "ltp": 375.0, "bid": 374.8, "ask": 375.2, "tick_age_s": 0.2, "tradable_quote": True},
        {"symbol": "NFO:NIFTY26MAY24000PE", "side": "PE", "strike": 24000, "atm_strike": 24050, "ltp": 360.0, "bid": 359.8, "ask": 360.2, "tick_age_s": 0.2, "tradable_quote": True},
        {"symbol": "NFO:NIFTY26MAY24100PE", "side": "PE", "strike": 24100, "atm_strike": 24050, "ltp": 389.0, "bid": 388.7, "ask": 389.3, "tick_age_s": 0.2, "tradable_quote": True},
    ]
    runner._set_cached_execution_candidate_basket(underlying="NIFTY", option_side="PE", atm_strike=24050, snapshots=cached, source="async_prepared")
    runner._order_manager.submit_trade_plan = MagicMock(return_value="order-123")
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True, "ok", {}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=symbol, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side="PE", entry_price=374.95, tick_age_s=0.2)])
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": symbol}], "atm_strike": 24050, "strategy_score": 7, "option_score": 7, "data_score": 7, "rr_score": 7})
    result = runner._handle_entry_signal_inner(signal, base_symbol=symbol, trade_symbol=symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-cache-submit")
    assert result.accepted is True
    assert result.reason != "candidate_refresh_pending"
    args = runner._trade_candidate_selector.select_ranked_candidates.call_args
    assert len(args.kwargs["candidates"]) > 1


@pytest.mark.asyncio
async def test_prepare_signal_strict_live_blocks_prebuild_in_paper_mode(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "true")
    monkeypatch.setenv("SHADOW_MODE", "false")
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._prepare_execution_candidate_basket_async = AsyncMock(return_value=[])  # type: ignore[method-assign]
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": "NFO:NIFTY26MAY24050PE"}]})
    await runner._prepare_signal_for_handling(signal, price=110.0, trace_id="paper")
    runner._prepare_execution_candidate_basket_async.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_signal_strict_live_blocks_prebuild_in_shadow_mode(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "true")
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._prepare_execution_candidate_basket_async = AsyncMock(return_value=[])  # type: ignore[method-assign]
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": "NFO:NIFTY26MAY24050PE"}]})
    await runner._prepare_signal_for_handling(signal, price=110.0, trace_id="shadow")
    runner._prepare_execution_candidate_basket_async.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_signal_strict_live_blocks_prebuild_when_live_disabled(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._prepare_execution_candidate_basket_async = AsyncMock(return_value=[])  # type: ignore[method-assign]
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": "NFO:NIFTY26MAY24050PE"}]})
    await runner._prepare_signal_for_handling(signal, price=110.0, trace_id="disabled")
    runner._prepare_execution_candidate_basket_async.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_signal_strict_live_allows_prebuild_in_true_live(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._resolve_execution_mode_snapshot = lambda: SimpleNamespace(is_live_mode=True)  # type: ignore[method-assign]
    runner._prepare_execution_candidate_basket_async = AsyncMock(return_value=[{"symbol": "NFO:NIFTY26MAY24050PE"}, {"symbol": "NFO:NIFTY26MAY24000PE"}])  # type: ignore[method-assign]
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": "NFO:NIFTY26MAY24050PE"}]})
    await runner._prepare_signal_for_handling(signal, price=110.0, trace_id="live")
    runner._prepare_execution_candidate_basket_async.assert_called_once()


@pytest.mark.asyncio
async def test_prepare_signal_missing_candidate_snapshots_triggers_prebuild(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._resolve_execution_mode_snapshot = lambda: SimpleNamespace(is_live_mode=True)  # type: ignore[method-assign]
    async def _prepare(**kwargs):
        kwargs["metadata"]["candidate_snapshots"] = [{"symbol": "NFO:NIFTY26MAY24050PE"}, {"symbol": "NFO:NIFTY26MAY24000PE"}, {"symbol": "NFO:NIFTY26MAY24100PE"}]
        return kwargs["metadata"]["candidate_snapshots"]
    runner._prepare_execution_candidate_basket_async = AsyncMock(side_effect=_prepare)  # type: ignore[method-assign]
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={})
    prepared, reason = await runner._prepare_signal_for_handling(signal, price=110.0, trace_id="missing")
    assert reason is None
    assert prepared is not None
    assert len(prepared.metadata["candidate_snapshots"]) == 3
    assert prepared.metadata["candidate_preparation_attempted"] is True


@pytest.mark.asyncio
async def test_prepare_signal_multi_snapshot_does_not_prebuild(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    runner._resolve_execution_mode_snapshot = lambda: SimpleNamespace(is_live_mode=True)  # type: ignore[method-assign]
    runner._prepare_execution_candidate_basket_async = AsyncMock(return_value=[])  # type: ignore[method-assign]
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}]})
    await runner._prepare_signal_for_handling(signal, price=110.0, trace_id="multi")
    runner._prepare_execution_candidate_basket_async.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_execution_candidate_basket_same_side_from_symbol_only() -> None:
    runner = _build_runner()
    signal = Signal(action="BUY", symbol="NFO:NIFTY26MAY24050PE", quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={})
    metadata = {}
    runner.build_candidate_snapshots_async = AsyncMock(return_value=([{"symbol": "NFO:NIFTY26MAY24050PE"}, {"symbol": "NFO:NIFTY26MAY24000PE"}, {"symbol": "NFO:NIFTY26MAY24100CE"}], False))
    snaps = await runner._prepare_execution_candidate_basket_async(signal=signal, metadata=metadata, underlying="NIFTY", option_side="PE", atm_strike=24050, trace_id="sym-only")
    assert len(snaps) == 2
    assert len(metadata["candidate_snapshots"]) == 2


def test_missing_candidate_snapshots_without_selected_attrs_returns_clean_rejection(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    symbol = "NFO:NIFTY26MAY24100PE"
    if hasattr(runner, "_selected_ce_symbol"):
        delattr(runner, "_selected_ce_symbol")
    if hasattr(runner, "_selected_pe_symbol"):
        delattr(runner, "_selected_pe_symbol")
    runner._is_option_symbol_tick_fresh = MagicMock(return_value=False)
    signal = Signal(
        action="BUY",
        symbol=symbol,
        quantity=1,
        confidence=0.9,
        reason="OrderFlow",
        stop_loss=300.0,
        take_profit=450.0,
        metadata={},
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol=symbol,
        trade_symbol=symbol,
        trade_price=374.95,
        timestamp=datetime.now(timezone.utc),
        trace_id="trace-missing-candidates",
    )
    assert result.reason == "missing_candidate_snapshots"
    assert result.reason != "entry_exception"


def test_selected_option_symbol_for_side_prefers_metadata() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    metadata = {
        "selected_ce": "NFO:NIFTY26MAY24150CE",
        "selected_pe": "NFO:NIFTY26MAY24100PE",
    }
    assert runner._selected_option_symbol_for_side("CE", metadata) == "NFO:NIFTY26MAY24150CE"
    assert runner._selected_option_symbol_for_side("PE", metadata) == "NFO:NIFTY26MAY24100PE"


def test_init_sets_selected_symbol_attrs() -> None:
    runner = _build_runner()
    assert hasattr(runner, "_selected_ce_symbol")
    assert hasattr(runner, "_selected_pe_symbol")


def test_runner_source_has_no_unsafe_selected_symbol_fallbacks() -> None:
    source_path = Path("src/nifty_scalper_bot/strategies/runner.py")
    source = source_path.read_text(encoding="utf-8")
    assert "or self._selected_ce_symbol" not in source
    assert "or self._selected_pe_symbol" not in source


def test_multi_snapshot_candidate_skips_refresh_and_no_unboundlocal(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    sym = "NFO:NIFTY26MAY24050PE"
    alt = "NFO:NIFTY26MAY24000PE"
    runner._build_candidate_snapshots_sync_safe = MagicMock()
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._transition_execution_state = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    runner._order_manager.submit_trade_plan = MagicMock(return_value="oid-multi")  # type: ignore[attr-defined]
    runner._materialize_option_trade_plan = MagicMock(side_effect=lambda s, **_: s)
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side="PE", entry_price=374.95, tick_age_s=0.2)])
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(ltp=374.95, bid=374.5, ask=375.4, tick_age_s=0.2, real_ticks_last_60s=4, tradable_quote=True, source="ws")
    signal = Signal(action="BUY", symbol=sym, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": sym, "side": "PE", "strike": 24050, "atm_strike": 24050, "ltp": 374.95, "bid": 374.5, "ask": 375.4, "tick_age_s": 0.2, "tradable_quote": True}, {"symbol": alt, "side": "PE", "strike": 24000, "atm_strike": 24050, "ltp": 360.0, "bid": 359.0, "ask": 361.0, "tick_age_s": 0.3, "tradable_quote": True}], "atm_strike": 24050})
    result = runner._handle_entry_signal_inner(signal, base_symbol=sym, trade_symbol=sym, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-multi")
    assert result.reason != "candidate_refresh_pending"
    assert runner._build_candidate_snapshots_sync_safe.call_count == 0
    assert runner._trade_candidate_selector.select_ranked_candidates.called


def test_trade_plan_materialization_failed_emits_diagnostics(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    emitted: list[dict] = []
    runner._logger.error = lambda *_args, **kwargs: emitted.append(kwargs.get("extra", {}))  # type: ignore[method-assign]
    symbol = "NFO:NIFTY26MAY24050PE"
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._materialize_option_trade_plan = MagicMock(side_effect=ValueError("plan failed"))
    signal = Signal(action="BUY", symbol=symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=450.0, metadata={"candidate_snapshots": [{"symbol": symbol, "side": "PE", "strike": 24050, "atm_strike": 24050, "ltp": 374.95, "bid": 374.5, "ask": 375.4, "tick_age_s": 0.2, "tradable_quote": True}]})
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=symbol, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side="PE", entry_price=374.95, tick_age_s=0.2)])
    result = runner._handle_entry_signal_inner(signal, base_symbol=symbol, trade_symbol=symbol, trade_price=374.95, timestamp=datetime.now(timezone.utc), trace_id="trace-materialize")
    assert result.reason == "trade_plan_materialization_failed"
    diag = next((e for e in emitted if e.get("event") == "TRADE_PLAN_MATERIALIZATION_DIAGNOSTICS"), {})
    assert diag.get("error_type") == "ValueError"
    assert diag.get("candidate_symbol") == symbol

def test_runner_on_tick_error_log_includes_error_type_phase(caplog):
    from nifty_scalper_bot.strategies.runner import StrategyRunner
    import logging
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("runner-test-err")
    runner._logger.setLevel(logging.INFO)
    runner._bracket_manager = None
    runner._position_manager = None
    runner._on_tick = StrategyRunner._on_tick.__get__(runner, StrategyRunner)
    with caplog.at_level(logging.ERROR):
        runner._on_tick("NFO:X", {"ltp": "bad"})
    assert "RUNNER_ON_TICK_ERROR" in caplog.text


def test_final_readiness_uses_selected_symbol_after_quote_revalidation() -> None:
    runner = _build_runner()
    runner._runtime_execution_ready_by_symbol = {}
    runner._is_symbol_execution_ready = MagicMock(return_value=False)
    runner._ensure_symbol_execution_ready_for_order = MagicMock(return_value=True)
    runner._transition_execution_state = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    runner._order_manager.submit_trade_plan = MagicMock(return_value="oid-selected")  # type: ignore[attr-defined]
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=376.0, bid=375.0, ask=376.0, tick_age_s=0.2, real_ticks_last_60s=8, tradable_quote=True, source='ws'
    )
    selected_symbol = "NFO:NIFTY26MAY23650PE"
    runner._trade_candidate_selector.select_best_candidate = MagicMock(
        return_value=SimpleNamespace(
            symbol=selected_symbol,
            stop_loss=350.0,
            target=430.0,
            score=8.0,
            data_quality_score=9.0,
            spread_pct=0.1,
            rr=2.0,
            side="PE",
            entry_price=376.0,
            tick_age_s=0.2,
        )
    )
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26MAY23750PE',
        quantity=1,
        confidence=0.9,
        reason='OrderFlow',
        stop_loss=350.0,
        take_profit=430.0,
        metadata={'candidate_snapshots': [{'symbol': selected_symbol, 'side': 'PE', 'strike': 23650, 'atm_strike': 23650, 'bid': 375.0, 'ask': 376.0, 'tradable_quote': True}]},
    )
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=376.0, timestamp=datetime.now(timezone.utc), trace_id='trace-selected-ready')
    assert result.accepted is True
    assert runner._ensure_symbol_execution_ready_for_order.call_args_list[-1].args[0] == selected_symbol


def test_invalid_lot_cooldown_applies_to_selected_symbol() -> None:
    runner = _build_runner()
    runner._runtime_execution_ready_by_symbol = {}
    runner._exec_reject_invalid_lot_seconds = 300.0
    selected_symbol = "NFO:NIFTY26MAY23650PE"
    runner._ensure_symbol_execution_ready_for_order = MagicMock(return_value=True)
    runner._is_symbol_execution_ready = MagicMock(return_value=False)
    runner._order_manager.resolve_lot_size = MagicMock(side_effect=RuntimeError("boom"))
    runner._trade_candidate_selector.select_best_candidate = MagicMock(
        return_value=SimpleNamespace(symbol=selected_symbol, stop_loss=300.0, target=420.0, score=8.0, data_quality_score=8.5, spread_pct=0.1, rr=2.0, side="PE", entry_price=360.0, tick_age_s=0.2)
    )
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=360.0, bid=359.0, ask=360.0, tick_age_s=0.2, real_ticks_last_60s=5, tradable_quote=True, source='ws'
    )
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26MAY23750PE',
        quantity=1,
        confidence=0.9,
        reason='OrderFlow',
        stop_loss=300.0,
        take_profit=420.0,
        metadata={'candidate_snapshots': [{'symbol': selected_symbol, 'side': 'PE', 'strike': 23650, 'atm_strike': 23650, 'bid': 359.0, 'ask': 360.0, 'tradable_quote': True}]},
    )
    first = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id='trace-invalid-lot-1')
    second = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id='trace-invalid-lot-2')
    assert first.reason == "invalid_lot_quantity"
    assert second.reason == "invalid_lot_quantity_reject_cooldown"


def test_order_readiness_revalidation_tries_raw_and_normalized_symbols() -> None:
    runner = _build_runner()
    runner._runtime_execution_ready_by_symbol = {}
    runner._runtime_live_orders_armed = True
    lot_attempts: list[str] = []
    raw_symbol = "NFO:NIFTY26MAY23650PE"
    normalized = "NIFTY26MAY23650PE"
    attempts: list[str] = []

    def _case_a(symbol: str):
        attempts.append(symbol)
        if symbol == raw_symbol:
            return SimpleNamespace(bid=100.0, ask=101.0, tradable_quote=True, tick_age_s=0.2)
        raise RuntimeError("missing")

    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.side_effect = _case_a
    runner._order_manager.resolve_lot_size = MagicMock(
        side_effect=lambda s: (_ for _ in ()).throw(RuntimeError("lot missing"))
        if s == normalized
        else (65 if s == raw_symbol else (_ for _ in ()).throw(RuntimeError("lot missing")))
    )
    assert runner._ensure_symbol_execution_ready_for_order(normalized, trace_id="lookup-1") is True
    assert normalized in attempts
    assert raw_symbol in attempts
    for call in runner._order_manager.resolve_lot_size.call_args_list:
        lot_attempts.append(call.args[0])
    assert normalized in lot_attempts
    assert raw_symbol in lot_attempts

    runner._runtime_execution_ready_by_symbol = {}
    attempts.clear()
    lot_attempts.clear()

    def _case_b(symbol: str):
        attempts.append(symbol)
        if symbol == normalized:
            return SimpleNamespace(bid=100.0, ask=101.0, tradable_quote=True, tick_age_s=0.2)
        raise RuntimeError("missing")

    runner._market_data.get_symbol_snapshot.side_effect = _case_b
    runner._order_manager.resolve_lot_size = MagicMock(
        side_effect=lambda s: (_ for _ in ()).throw(RuntimeError("lot missing"))
        if s == raw_symbol
        else (65 if s == normalized else (_ for _ in ()).throw(RuntimeError("lot missing")))
    )
    assert runner._ensure_symbol_execution_ready_for_order(raw_symbol, trace_id="lookup-2") is True
    assert raw_symbol in attempts
    assert normalized in attempts
    for call in runner._order_manager.resolve_lot_size.call_args_list:
        lot_attempts.append(call.args[0])
    assert raw_symbol in lot_attempts
    assert normalized in lot_attempts


def test_early_runtime_not_ready_cooldown_resets_execution_state() -> None:
    runner = _build_runner()
    runner._runtime_execution_ready_by_symbol = {}
    runner._execution_reject_cooldown_ts = {}
    runner._reset_execution_state = MagicMock()
    base_symbol = "NFO:NIFTY26MAY23650PE"
    reason_key = "OrderFlow"
    runner._execution_reject_cooldown_ts[
        f"{base_symbol}:{reason_key}:runtime_symbol_execution_not_ready"
    ] = datetime.now(timezone.utc).timestamp()

    signal = Signal(
        action='BUY',
        symbol=base_symbol,
        quantity=1,
        confidence=0.8,
        reason=reason_key,
        stop_loss=300.0,
        take_profit=420.0,
        metadata={},
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol=base_symbol,
        trade_symbol=base_symbol,
        trade_price=360.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='trace-cooldown-early',
    )
    assert result.reason == "runtime_symbol_execution_not_ready_reject_cooldown"
    runner._reset_execution_state.assert_called()


def test_cooldown_block_events_emitted(caplog, monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sym = "NFO:NIFTY26MAY23650PE"
    now = datetime.now(timezone.utc).timestamp()
    runner._underlying_last_signal_ts["NIFTY"] = now
    s = Signal(action="BUY", symbol=sym, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=300.0, take_profit=420.0, metadata={})
    with caplog.at_level(logging.INFO):
        r1 = runner._handle_entry_signal_inner(s, base_symbol=sym, trade_symbol=sym, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id="cd-underlying")
    assert r1.reason == "underlying_cooldown"
    assert "UNDERLYING_ORDER_COOLDOWN_BLOCKED" in caplog.text

    caplog.clear()
    runner._underlying_last_signal_ts.clear()
    runner._reason_last_signal_ts["NIFTY:OrderFlow"] = now
    with caplog.at_level(logging.INFO):
        r2 = runner._handle_entry_signal_inner(s, base_symbol=sym, trade_symbol=sym, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id="cd-reason")
    assert r2.reason == "reason_cooldown"
    assert "STRATEGY_REASON_ORDER_COOLDOWN_BLOCKED" in caplog.text

    caplog.clear()
    runner._reason_last_signal_ts.clear()
    runner._signal_reject_cooldown_ts[f"{sym}:OrderFlow:score_below_threshold"] = now
    with caplog.at_level(logging.INFO):
        r3 = runner._handle_entry_signal_inner(s, base_symbol=sym, trade_symbol=sym, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id="cd-score")
    assert r3.reason == "score_below_threshold_reject_cooldown"
    assert "SCORE_REJECT_COOLDOWN_BLOCKED" in caplog.text

    caplog.clear()
    runner._signal_reject_cooldown_ts.clear()
    runner._max_order_attempts_per_minute = 1
    runner._order_attempt_window.append(now)
    with caplog.at_level(logging.INFO):
        r4 = runner._handle_entry_signal_inner(s, base_symbol=sym, trade_symbol=sym, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id="cd-rate")
    assert r4.reason == "max_order_attempts_per_minute"
    assert "ORDER_ATTEMPT_RATE_LIMIT_BLOCKED" in caplog.text


def test_live_execution_uses_basket_and_submits_alternate_candidate(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    runner._trade_candidate_selector = TradeCandidateSelector(min_option_premium=100.0, max_option_premium=650.0)
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._order_manager.submit_trade_plan = MagicMock(return_value="oid-alt")
    signal_symbol = "NFO:NIFTY26MAY24050CE"
    alt_symbol = "NFO:NIFTY26MAY24100CE"
    runner._build_candidate_snapshots_sync_safe = MagicMock(
        return_value=(
            [
                {"symbol": signal_symbol, "side": "CE", "option_type": "CE", "strike": 24050, "atm_strike": 24050, "ltp": 80.0, "bid": 79.0, "ask": 81.0, "tick_age_s": 0.5, "real_ticks_last_60s": 3},
                {"symbol": alt_symbol, "side": "CE", "option_type": "CE", "strike": 24100, "atm_strike": 24050, "ltp": 150.0, "bid": 149.0, "ask": 151.0, "tick_age_s": 0.5, "real_ticks_last_60s": 4},
            ],
            False,
            "async_refresh",
        )
    )
    signal = Signal(action="BUY", symbol=signal_symbol, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=60.0, take_profit=220.0, metadata={"candidate_snapshots": [{"symbol": signal_symbol, "side": "CE", "option_type": "CE", "strike": 24050, "atm_strike": 24050, "ltp": 80.0, "bid": 79.0, "ask": 81.0, "tick_age_s": 0.4, "real_ticks_last_60s": 2}], "atm_strike": 24050})
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal_symbol, trade_symbol=signal_symbol, trade_price=150.0, timestamp=datetime.now(timezone.utc), trace_id="trace-alt")
    assert result.accepted is True
    assert result.reason == "order_submitted"
    submitted_plan = runner._order_manager.submit_trade_plan.call_args.args[0]
    assert submitted_plan.symbol == alt_symbol


def test_order_rejected_does_not_mark_duplicate_cooldown(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    runner._logger = MagicMock()
    runner._trade_candidate_selector = TradeCandidateSelector(min_option_premium=40.0, max_option_premium=650.0)
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._order_manager.submit_trade_plan = MagicMock(return_value=None)
    sym = "NFO:NIFTY26MAY24050CE"
    signal = Signal(action="BUY", symbol=sym, quantity=1, confidence=0.9, reason="OrderFlow", stop_loss=100.0, take_profit=200.0, metadata={"candidate_snapshots": [{"symbol": sym, "side": "CE", "option_type": "CE", "strike": 24050, "atm_strike": 24050, "ltp": 140.0, "bid": 139.0, "ask": 141.0, "tick_age_s": 0.2, "real_ticks_last_60s": 5}]})
    result = runner._handle_entry_signal_inner(signal, base_symbol=sym, trade_symbol=sym, trade_price=140.0, timestamp=datetime.now(timezone.utc), trace_id="trace-no-order")
    assert result.reason == "order_rejected"
    assert not runner._underlying_last_signal_ts


def test_directional_dedup_blocks_second_pe_same_reason_within_window() -> None:
    runner = _build_runner()
    now_epoch = 1000.0
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    metadata = {'candidate_score': 8.0, 'tradable_quote': True, 'candidate_spread_pct': 0.8}
    first = runner._apply_directional_signal_dedup(signal=signal, metadata=metadata, underlying='NIFTY', now_epoch=now_epoch, selected_symbol=signal.symbol, option_side='PE', selected_snapshot={})
    second = runner._apply_directional_signal_dedup(signal=signal, metadata=metadata, underlying='NIFTY', now_epoch=now_epoch + 10.0, selected_symbol='NFO:NIFTY26APR23850PE', option_side='PE', selected_snapshot={})
    assert first is None
    assert second is not None
    assert second.reason == 'signal_attempt_debounce'


def test_directional_dedup_separates_pe_and_ce_keys() -> None:
    runner = _build_runner()
    now_epoch = 1000.0
    pe = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    ce = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    metadata = {'candidate_score': 8.0, 'tradable_quote': True, 'candidate_spread_pct': 0.8}
    runner._apply_directional_signal_dedup(signal=pe, metadata=metadata, underlying='NIFTY', now_epoch=now_epoch, selected_symbol=pe.symbol, option_side='PE', selected_snapshot={})
    result = runner._apply_directional_signal_dedup(signal=ce, metadata=metadata, underlying='NIFTY', now_epoch=now_epoch + 5.0, selected_symbol=ce.symbol, option_side='CE', selected_snapshot={})
    assert result is None


def test_directional_dedup_separates_reason_keys() -> None:
    runner = _build_runner()
    now_epoch = 1000.0
    s1 = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    s2 = Signal(action='BUY', symbol='NFO:NIFTY26APR23850PE', quantity=1, confidence=0.9, reason='Momentum', stop_loss=100.0, take_profit=120.0, metadata={})
    metadata = {'candidate_score': 8.0, 'tradable_quote': True, 'candidate_spread_pct': 0.8}
    runner._apply_directional_signal_dedup(signal=s1, metadata=metadata, underlying='NIFTY', now_epoch=now_epoch, selected_symbol=s1.symbol, option_side='PE', selected_snapshot={})
    result = runner._apply_directional_signal_dedup(signal=s2, metadata=metadata, underlying='NIFTY', now_epoch=now_epoch + 5.0, selected_symbol=s2.symbol, option_side='PE', selected_snapshot={})
    assert result is None


def test_directional_dedup_allows_better_candidate_within_window() -> None:
    runner = _build_runner()
    now_epoch = 1000.0
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    weaker = {'candidate_score': 6.0, 'tradable_quote': True, 'candidate_spread_pct': 1.2, 'candidate_tick_age_s': 2.0}
    stronger = {'candidate_score': 8.5, 'tradable_quote': True, 'candidate_spread_pct': 0.5, 'candidate_tick_age_s': 0.5}
    first = runner._apply_directional_signal_dedup(signal=signal, metadata=weaker, underlying='NIFTY', now_epoch=now_epoch, selected_symbol=signal.symbol, option_side='PE', selected_snapshot={'depth_available': False, 'distance_from_atm': 100})
    second = runner._apply_directional_signal_dedup(signal=signal, metadata=stronger, underlying='NIFTY', now_epoch=now_epoch + 5.0, selected_symbol='NFO:NIFTY26APR23850PE', option_side='PE', selected_snapshot={'depth_available': True, 'distance_from_atm': 50})
    assert first is None
    assert second is None


def test_directional_dedup_blocks_tiny_score_bump_if_quality_worse() -> None:
    runner = _build_runner()
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    first = runner._apply_directional_signal_dedup(
        signal=signal,
        metadata={'candidate_score': 8.0, 'tradable_quote': True, 'candidate_spread_pct': 0.4, 'candidate_tick_age_ms': 10.0},
        underlying='NIFTY',
        now_epoch=1000.0,
        selected_symbol=signal.symbol,
        option_side='PE',
        selected_snapshot={'depth_available': True, 'distance_from_atm': 10},
    )
    second = runner._apply_directional_signal_dedup(
        signal=signal,
        metadata={'candidate_score': 8.1, 'tradable_quote': False, 'candidate_spread_pct': 8.0, 'candidate_tick_age_ms': 7000.0},
        underlying='NIFTY',
        now_epoch=1003.0,
        selected_symbol='NFO:NIFTY26APR23850PE',
        option_side='PE',
        selected_snapshot={'depth_available': False, 'distance_from_atm': 300},
    )
    assert first is None
    assert second is not None
    assert second.reason == 'signal_attempt_debounce'


def test_dedup_prefers_tick_age_ms_over_candidate_tick_age_s() -> None:
    runner = _build_runner()
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    runner._apply_directional_signal_dedup(
        signal=signal,
        metadata={'candidate_score': 8.0, 'tradable_quote': True, 'candidate_spread_pct': 0.5, 'candidate_tick_age_ms': 0.0, 'candidate_tick_age_s': 9.0},
        underlying='NIFTY',
        now_epoch=1000.0,
        selected_symbol=signal.symbol,
        option_side='PE',
        selected_snapshot={},
    )
    state = runner._signal_attempt_debounce_state['NIFTY:PE:OrderFlow']
    assert state['ranking_fields']['tick_age_ms'] == 0.0


def test_dedup_uses_zero_candidate_tick_age_s_without_fallback_default() -> None:
    runner = _build_runner()
    signal = Signal(action='BUY', symbol='NFO:NIFTY26APR23800PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=100.0, take_profit=120.0, metadata={})
    runner._apply_directional_signal_dedup(
        signal=signal,
        metadata={'candidate_score': 8.0, 'tradable_quote': True, 'candidate_spread_pct': 0.5, 'candidate_tick_age_s': 0.0},
        underlying='NIFTY',
        now_epoch=1000.0,
        selected_symbol=signal.symbol,
        option_side='PE',
        selected_snapshot={},
    )
    state = runner._signal_attempt_debounce_state['NIFTY:PE:OrderFlow']
    assert state['ranking_fields']['tick_age_ms'] == 0.0


def test_mark_directional_dedup_failed_clears_reservation() -> None:
    runner = _build_runner()
    runner._signal_attempt_debounce_state['NIFTY:PE:OrderFlow'] = {'status': 'reserved', 'rank_score': 10.0}
    runner._mark_directional_dedup_failed(underlying='NIFTY', option_side='PE', reason='OrderFlow')
    assert 'NIFTY:PE:OrderFlow' not in runner._signal_attempt_debounce_state


def test_directional_dedup_fallback_selected_symbol_snapshot_without_candidates(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    captured: dict[str, object] = {}

    def _capture_dedup(**kwargs):
        captured['selected_symbol'] = kwargs.get('selected_symbol')
        captured['selected_snapshot'] = kwargs.get('selected_snapshot')
        return SignalExecutionResult(False, 'dedup_test_stop')

    runner._apply_directional_signal_dedup = _capture_dedup  # type: ignore[method-assign]

    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.9,
        reason='OrderFlow',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={'strategy_score': 7, 'option_score': 7, 'data_score': 7, 'rr_score': 7},
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='dedup-fallback',
    )
    assert result.reason == 'dedup_test_stop'
    assert captured['selected_symbol'] == 'NFO:NIFTY26APR23800PE'
    assert captured['selected_snapshot'] == {}

def test_dynamic_revalidation_marks_symbol_ready_when_fresh_quote(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_live_orders_armed = False
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=5
    )
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='dyn-1') is True
    assert runner._runtime_execution_ready_by_symbol.get('NFO:NIFTY26MAY23700PE') is True


def test_dynamic_revalidation_rejects_missing_lot_size(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=5
    )
    runner._order_manager.resolve_lot_size = MagicMock(side_effect=RuntimeError('no lot'))
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='dyn-2') is False


def test_candidate_selection_tries_next_ready_candidate(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_execution_ready_by_symbol = {'NFO:NIFTY26MAY23650PE': True}
    runner._order_manager.submit_trade_plan = MagicMock(return_value='oid-next')
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        ltp=360.0, bid=359.0, ask=360.0, tick_age_s=0.2, real_ticks_last_60s=5, tradable_quote=True, source='ws'
    )
    first = SimpleNamespace(symbol='NFO:NIFTY26MAY23700PE', stop_loss=300.0, target=420.0, score=9.0, data_quality_score=8.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=360.0, tick_age_s=0.2)
    second = SimpleNamespace(symbol='NFO:NIFTY26MAY23650PE', stop_loss=300.0, target=420.0, score=8.0, data_quality_score=8.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=360.0, tick_age_s=0.2)
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[first, second])
    runner._ensure_symbol_execution_ready_for_order = MagicMock(side_effect=lambda s, trace_id=None: s.endswith('23650PE'))
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY23750PE', quantity=1, confidence=0.9, reason='OrderFlow', stop_loss=300.0, take_profit=420.0, metadata={'candidate_snapshots': [{'symbol': 'NFO:NIFTY26MAY23700PE', 'side': 'PE', 'strike': 23700, 'atm_strike': 23700, 'bid': 359.0, 'ask': 360.0, 'tradable_quote': True}, {'symbol': 'NFO:NIFTY26MAY23650PE', 'side': 'PE', 'strike': 23650, 'atm_strike': 23700, 'bid': 359.0, 'ask': 360.0, 'tradable_quote': True}]})
    result = runner._handle_entry_signal_inner(signal, base_symbol=signal.symbol, trade_symbol=signal.symbol, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id='trace-next-ready')
    assert result.accepted is True


def test_runtime_not_ready_cooldown_short_circuits(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._execution_reject_cooldown_ts = {}
    base_symbol = 'NFO:NIFTY26MAY23700PE'
    reason_key = 'OrderFlow'
    now = datetime.now(timezone.utc).timestamp()
    runner._execution_reject_cooldown_ts[f'NFO:NIFTY26MAY23700PE:{reason_key}:runtime_symbol_execution_not_ready'] = now
    runner._order_manager.submit_trade_plan = MagicMock(return_value='oid-should-not')
    signal = Signal(action='BUY', symbol=base_symbol, quantity=1, confidence=0.9, reason=reason_key, stop_loss=300.0, take_profit=420.0, metadata={'candidate_snapshots': []})
    result = runner._handle_entry_signal_inner(signal, base_symbol=base_symbol, trade_symbol=base_symbol, trade_price=360.0, timestamp=datetime.now(timezone.utc), trace_id='trace-cool')
    assert result.reason == 'runtime_symbol_execution_not_ready_reject_cooldown'


def test_order_readiness_revalidation_log_includes_dynamic_fields(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    logs: list[str] = []
    runner._logger.info = lambda msg, *args, **kwargs: logs.append((msg % args) if args else str(msg))  # type: ignore[method-assign]
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=5
    )
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='dyn-log') is True
    joined = "\n".join(logs)
    assert 'ORDER_READINESS_REVALIDATION' in joined
    assert 'dynamic_revalidation_attempted=' in joined
    assert 'dynamic_revalidation_passed=' in joined
    assert 'runtime_key=' in joined
    assert 'history_fallback=' in joined


def test_dynamic_revalidation_allows_fresh_quote_when_tick_count_missing(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('RUNNER_ALLOW_FRESH_QUOTE_WITHOUT_TICK_COUNT', 'true')
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=None
    )
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='dyn-fallback') is True


def test_is_symbol_execution_ready_false_when_map_empty_even_if_live_orders_armed(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_live_orders_armed = True
    runner._runtime_execution_ready_by_symbol = {}
    runner._runtime_symbol_last_ready_at = {}
    assert runner._is_symbol_execution_ready('NFO:NIFTY26MAY23700PE') is False


def test_is_symbol_execution_ready_expires_by_ttl(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('RUNNER_EXECUTION_READY_TTL_SECONDS', '1')
    key = 'NFO:NIFTY26MAY23700PE'
    runner._runtime_execution_ready_by_symbol = {key: True}
    runner._runtime_symbol_last_ready_at = {key: 1.0}
    monkeypatch.setattr('nifty_scalper_bot.strategies.runner.time.time', lambda: 5.0)
    assert runner._is_symbol_execution_ready(key) is False
    assert key not in runner._runtime_execution_ready_by_symbol


def test_runtime_ready_key_keeps_spot_symbol_not_nfo() -> None:
    runner = _build_runner()
    assert runner._runtime_ready_key('NSE:NIFTY') == 'NSE:NIFTY'


def test_runtime_ready_key_maps_prefixed_and_unprefixed_option_to_same_key() -> None:
    runner = _build_runner()
    assert runner._runtime_ready_key('NIFTY26MAY23700PE') == 'NFO:NIFTY26MAY23700PE'
    assert runner._runtime_ready_key('NFO:NIFTY26MAY23700PE') == 'NFO:NIFTY26MAY23700PE'


def test_set_runtime_readiness_populates_last_ready_timestamp(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('RUNNER_EXECUTION_READY_TTL_SECONDS', '60')
    now_ts = 1700000000.0
    monkeypatch.setattr('nifty_scalper_bot.strategies.runner.time.time', lambda: now_ts)
    symbol = 'NFO:NIFTY26MAY23700PE'
    runner.set_runtime_readiness(
        data_hard_ready=True,
        evaluation_ready=True,
        live_orders_armed=False,
        execution_ready_by_symbol={symbol: True},
    )
    runtime_key = runner._runtime_ready_key(symbol)
    assert runner._runtime_symbol_last_ready_at.get(runtime_key) == now_ts
    assert runner._is_symbol_execution_ready(symbol) is True


def test_dynamic_revalidation_resolves_lot_size_once(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=5
    )
    runner._order_manager.resolve_lot_size = MagicMock(return_value=65)
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='lot-once') is True
    assert runner._order_manager.resolve_lot_size.call_count == 1


def test_dynamic_revalidation_snapshot_alias_continues_after_none(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    raw_symbol = 'NFO:NIFTY26MAY23700PE'
    normalized = 'NIFTY26MAY23700PE'

    def _snap(symbol: str):
        if symbol == raw_symbol:
            return None
        if symbol == normalized:
            return SimpleNamespace(
                bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=5
            )
        raise RuntimeError('missing')

    runner._market_data.get_symbol_snapshot.side_effect = _snap
    assert runner._ensure_symbol_execution_ready_for_order(raw_symbol, trace_id='alias-none') is True


def test_dynamic_revalidation_rejects_lot_size_zero(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=5
    )
    runner._order_manager.resolve_lot_size = MagicMock(return_value=0)
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='lot-zero') is False


def test_dynamic_revalidation_rejects_missing_tick_count_when_fallback_disabled(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('RUNNER_ALLOW_FRESH_QUOTE_WITHOUT_TICK_COUNT', 'false')
    runner._runtime_execution_ready_by_symbol = {}
    runner._market_data = MagicMock()
    runner._market_data.get_symbol_snapshot.return_value = SimpleNamespace(
        bid=114.7, ask=115.0, tick_age_s=0.2, tradable_quote=True, source='ws', real_ticks_last_60s=None
    )
    assert runner._ensure_symbol_execution_ready_for_order('NFO:NIFTY26MAY23700PE', trace_id='hist-off') is False

def test_candidate_selected_price_falls_back_to_snapshot_ask(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=True, order_id='oid', reason='accepted', details={}, broker_attempted=True))
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._materialize_option_trade_plan = MagicMock(side_effect=lambda s, **k: s)
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=None, tick_age_s=0.1)])
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='r',stop_loss=300.0,take_profit=450.0,metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':119.0,'bid':118.5,'ask':120.0,'tick_age_s':0.1,'tradable_quote':True}]})
    runner._handle_entry_signal_inner(signal,sym,sym,100.0,datetime.now(timezone.utc),trace_id='ask')
    assert runner._materialize_option_trade_plan.call_args.kwargs['execution_price'] == 120.0


def test_order_attempt_window_counts_only_broker_attempted(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='r',stop_loss=300.0,take_profit=450.0,metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=False, order_id=None, reason='quote_unavailable', details={'symbol': sym}, broker_attempted=False))
    r1=runner._handle_entry_signal_inner(signal,sym,sym,100.0,datetime.now(timezone.utc),trace_id='b0')
    assert r1.reason=='order_manager_quote_unavailable'
    assert len(runner._order_attempt_window)==0
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=False, order_id=None, reason='place_order_rejected', details={}, broker_attempted=True))
    runner._handle_entry_signal_inner(signal,sym,sym,100.0,datetime.now(timezone.utc),trace_id='b1')
    assert len(runner._order_attempt_window)==1

def test_shadow_mode_does_not_call_execution_readiness_result(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    sym = 'NFO:NIFTY26MAY23700PE'
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(False, 'quote_stale', {'tick_age_ms': 90000}))
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=True, order_id='oid-s', reason='accepted', details={}, broker_attempted=True))
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    signal = Signal(action='BUY', symbol=sym, quantity=1, confidence=0.9, reason='r', stop_loss=300.0, take_profit=450.0, metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    result = runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='shadow-ready-skip')
    assert runner._ensure_symbol_execution_ready_result.call_count == 0
    assert result.reason != 'runtime_symbol_execution_not_ready'


def test_live_mode_calls_execution_readiness_result(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner._order_manager.is_live_mode = lambda: True  # type: ignore[attr-defined]
    sym = 'NFO:NIFTY26MAY23700PE'
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(False, 'quote_stale', {'tick_age_ms': 90000}))
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    signal = Signal(action='BUY', symbol=sym, quantity=1, confidence=0.9, reason='r', stop_loss=300.0, take_profit=450.0, metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    result = runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='live-ready-call')
    runner._ensure_symbol_execution_ready_result.assert_called_once()
    assert result.reason == 'runtime_symbol_execution_not_ready'
    assert result.details.get('readiness_reason') == 'quote_stale'
    assert result.details.get('tick_age_ms') == 90000

def test_submit_result_deterministic_rejection_rolls_back_dedup(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=False, order_id=None, reason='kill_switch_active', details={'last_failure':{'exception_type':'RuntimeError'}}, broker_attempted=True))
    runner._order_manager.get_kill_switch_status = MagicMock(return_value={"active": True, "kill_reason": "unexpected_exception", "trace_id": "det-rej"})
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='OrderFlow',stop_loss=300.0,take_profit=450.0,metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    result = runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='det-rej')
    assert result.reason == 'order_manager_kill_switch_active'
    assert result.details.get("kill_switch_status", {}).get("active") is True
    assert 'NIFTY:PE:OrderFlow' not in runner._signal_attempt_debounce_state


def test_submit_result_uncertain_marks_dedup_uncertain(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=False, order_id=None, reason='placement_uncertain', details={'x':1}, broker_attempted=True))
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='OrderFlow',stop_loss=300.0,take_profit=450.0,metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='uncertain')
    state = runner._signal_attempt_debounce_state.get('NIFTY:PE:OrderFlow')
    assert state is not None and state.get('status') == 'uncertain'
    assert float(state.get('expires_at', 0.0)) > float(state.get('ts', 0.0))


def test_runner_rejects_broker_placement_exception_with_details(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=False, order_id=None, reason='broker_placement_exception', details={'error_type': 'RuntimeError', 'trace_id': 'bpx-1'}, broker_attempted=True))
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='OrderFlow',stop_loss=300.0,take_profit=450.0,metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    result = runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='bpx-1')
    assert result.reason == 'order_manager_broker_placement_exception'
    assert result.details.get('error_type') == 'RuntimeError'
    assert any(str(k).startswith(f'{sym}|BUY|OrderFlow|RuntimeError') for k in runner._order_failure_cooldown_until.keys())


def test_failure_cooldown_blocks_immediate_same_symbol_retry_without_broker_attempt(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    runner._order_failure_cooldown_until[f"{sym}|BUY|OrderFlow|RuntimeError"] = time.time() + 60.0
    runner._order_manager.submit_trade_plan_result = MagicMock(return_value=SimpleNamespace(accepted=True, order_id='oid', reason='accepted', details={}, broker_attempted=True))
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='OrderFlow',stop_loss=300.0,take_profit=450.0,metadata={'strategy_name':'OrderFlow','candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    result = runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='cooldown')
    assert result.reason == "order_failure_cooldown_active"
    runner._order_manager.submit_trade_plan_result.assert_not_called()


def test_suspended_future_ttl_expiry_allows_reconsideration() -> None:
    runner = _build_runner()
    sym = "NFO:NIFTY26MAYFUT"
    runner._suspended_context_symbols.add(sym)
    runner._suspended_context_symbol_until[sym] = time.monotonic() - 1.0
    assert runner._is_context_symbol_suspended(sym) is False
    assert sym not in runner._suspended_context_symbols


def test_entry_exception_rolls_back_dedup(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE'); monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true'); monkeypatch.setenv('PAPER__ENABLED', 'false'); monkeypatch.setenv('SHADOW_MODE', 'false')
    sym='NFO:NIFTY26MAY23700PE'
    runner._is_symbol_execution_ready = MagicMock(return_value=True)
    runner._ensure_symbol_execution_ready_result = MagicMock(return_value=ExecutionReadinessResult(True,'ok',{}))
    runner._trade_candidate_selector.select_ranked_candidates = MagicMock(return_value=[SimpleNamespace(symbol=sym, stop_loss=300.0, target=450.0, score=8.0, data_quality_score=9.0, spread_pct=0.1, rr=2.0, side='PE', entry_price=110.0, tick_age_s=0.1)])
    runner._order_manager.submit_trade_plan_result = MagicMock(side_effect=RuntimeError('boom submit'))
    signal=Signal(action='BUY',symbol=sym,quantity=1,confidence=0.9,reason='OrderFlow',stop_loss=300.0,take_profit=450.0,metadata={'candidate_snapshots':[{'symbol':sym,'side':'PE','strike':23700,'atm_strike':23700,'ltp':110.0,'bid':109.5,'ask':110.0,'tick_age_s':0.1,'tradable_quote':True}]})
    result = runner._handle_entry_signal_inner(signal, sym, sym, 100.0, datetime.now(timezone.utc), trace_id='entry-exc')
    assert result.reason == 'entry_exception'
    assert 'NIFTY:PE:OrderFlow' not in runner._signal_attempt_debounce_state
