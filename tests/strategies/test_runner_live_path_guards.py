from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
from unittest.mock import MagicMock

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
    runner._entry_lock = threading.Lock()
    return runner


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
