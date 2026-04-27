from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import logging
import threading
from unittest.mock import AsyncMock, MagicMock
import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner
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


def test_stale_thresholds_are_instrument_specific() -> None:
    runner = _build_runner()
    runner._option_stale_tick_seconds = 900.0
    runner._future_stale_tick_seconds = 120.0
    runner._index_stale_tick_seconds = 120.0
    runner._generic_stale_tick_seconds = 60.0
    assert runner._stale_tick_threshold_for_symbol('NSE:NIFTY') == 120.0
    assert runner._stale_tick_threshold_for_symbol('NFO:NIFTY26APR23800PE') == 900.0
    assert runner._stale_tick_threshold_for_symbol('NFO:NIFTY26APRFUT') == 120.0


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
    assert result.reason == 'final_score_required'


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
    runner.build_candidate_snapshots_async.assert_called_once()


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


@pytest.mark.asyncio
async def test_build_candidate_snapshots_all_pending_returns_refresh_pending() -> None:
    runner = _build_runner()
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
