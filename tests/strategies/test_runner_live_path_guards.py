from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import logging
from types import SimpleNamespace
import threading
from unittest.mock import AsyncMock, MagicMock
import pytest

from nifty_scalper_bot.strategies.runner import SignalExecutionResult, StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


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
    assert isinstance(prepared.metadata.get('candidate_snapshots'), list)
    assert prepared.metadata.get('atm_strike') == 23800
    runner.build_candidate_snapshots_async.assert_called_once()


@pytest.mark.asyncio
async def test_prepare_signal_sanitizes_underlying_to_nifty(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
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
    assert captured.get('underlying') == 'NIFTY'


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
    assert prepared is None
    assert reason == 'candidate_refresh_pending'


@pytest.mark.asyncio
async def test_live_async_path_uses_signal_candidate_fallback(monkeypatch) -> None:
    runner = _build_runner()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
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
    assert reason is None
    assert prepared is not None
    candidate = prepared.metadata['candidate_snapshots'][0]
    assert candidate['ltp'] == 111.0
    assert candidate['bid'] == 0.0
    assert candidate['ask'] == 0.0
    assert candidate['ltp_only_fallback'] is True


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
