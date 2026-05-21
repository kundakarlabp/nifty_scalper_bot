from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass


class _OrderManagerNoMode:
    pass


class _OrderManagerWithBrokenMode:
    is_live_mode = True


class _OrderManagerWithLiveMode:
    def is_live_mode(self) -> bool:
        return True


def _build_runner(order_manager: object) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = order_manager
    runner._runtime_live_orders_armed = True
    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
    runner._underlying_signal_cooldown_seconds = 0.0
    runner._reason_signal_cooldown_seconds = 0.0
    runner._cooldown_log_throttle_seconds = 1.0
    runner._order_attempt_window = deque()
    runner._max_order_attempts_per_minute = 10
    runner._signal_reject_cooldown_ts = {}
    runner._reset_execution_state = lambda *_args, **_kwargs: None
    runner._extract_underlying = lambda symbol: symbol
    return runner


def _build_signal() -> Signal:
    return Signal(
        action='BUY',
        symbol='NSE:NIFTY',
        quantity=1,
        confidence=0.9,
        reason='OrderFlow',
        stop_loss=1.0,
        take_profit=2.0,
        metadata={},
    )


def test_entry_mode_resolution_without_mode_source_no_unbound_local(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = _build_runner(_OrderManagerNoMode())

    result = runner._handle_entry_signal_inner(
        _build_signal(),
        'NSE:NIFTY',
        'NSE:NIFTY',
        100.0,
        datetime.now(timezone.utc),
        trace_id='no-mode-source',
    )

    assert result.reason == 'non_option_symbol'


def test_entry_mode_resolution_non_callable_mode_source_no_unbound_local(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = _build_runner(_OrderManagerWithBrokenMode())

    result = runner._handle_entry_signal_inner(
        _build_signal(),
        'NSE:NIFTY',
        'NSE:NIFTY',
        100.0,
        datetime.now(timezone.utc),
        trace_id='broken-mode-source',
    )

    assert result.reason == 'non_option_symbol'


def test_entry_mode_resolution_callable_live_mode_no_unbound_local(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    runner = _build_runner(_OrderManagerWithLiveMode())

    result = runner._handle_entry_signal_inner(
        _build_signal(),
        'NSE:NIFTY',
        'NSE:NIFTY',
        100.0,
        datetime.now(timezone.utc),
        trace_id='callable-mode-source',
    )

    assert result.reason == 'non_option_symbol'
