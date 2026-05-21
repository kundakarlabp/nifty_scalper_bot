from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


OPTION_SYMBOL = 'NFO:NIFTY26MAY23750CE'


class _Logger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []

    def debug(self, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        try:
            rendered = msg % args if args else str(msg)
        except Exception:
            rendered = str(msg)
        self.info_messages.append(rendered)

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass


class _OrderManagerNoMode:
    pass


class _OrderManagerWithLiveMode:
    def is_live_mode(self) -> bool:
        return True


def _build_runner(order_manager: object) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = order_manager
    runner._runtime_live_orders_armed = True
    runner._runtime_execution_ready_by_symbol = {}
    runner._runtime_data_hard_ready = True
    runner._runtime_evaluation_ready = True
    runner._runtime_readiness_reason = 'ready'
    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
    runner._underlying_signal_cooldown_seconds = 0.0
    runner._reason_signal_cooldown_seconds = 0.0
    runner._cooldown_log_throttle_seconds = 1.0
    runner._order_attempt_window = deque([1.0] * 11)
    runner._max_order_attempts_per_minute = 10
    runner._signal_reject_cooldown_ts = {}
    runner._reset_execution_state = lambda *_args, **_kwargs: None
    runner._extract_underlying = lambda symbol: symbol
    runner._is_tradable_symbol = lambda _symbol: True
    runner._is_symbol_execution_ready = lambda _symbol: False
    return runner


def _build_signal() -> Signal:
    return Signal(
        action='BUY',
        symbol=OPTION_SYMBOL,
        quantity=1,
        confidence=0.9,
        reason='OrderFlow',
        stop_loss=1.0,
        take_profit=2.0,
        metadata={'candidate_snapshots': []},
    )


def _run_and_get_mode_log(runner: StrategyRunner) -> str:
    result = runner._handle_entry_signal_inner(
        _build_signal(),
        OPTION_SYMBOL,
        OPTION_SYMBOL,
        100.0,
        datetime.now(timezone.utc),
        trace_id='mode-check',
    )
    assert result.reason in {'runtime_symbol_execution_not_ready', 'max_order_attempts_per_minute'}
    mode_logs = [m for m in runner._logger.info_messages if 'ENTRY_EXECUTION_MODE_RESOLVED' in m]
    assert mode_logs
    return mode_logs[0]


def test_shadow_mode_with_live_flag_stays_non_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    log_line = _run_and_get_mode_log(_build_runner(_OrderManagerNoMode()))
    assert 'is_live_mode=False' in log_line
    assert 'execution_mode=SHADOW' in log_line


def test_live_mode_without_live_flag_stays_non_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'false')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    log_line = _run_and_get_mode_log(_build_runner(_OrderManagerNoMode()))
    assert 'is_live_mode=False' in log_line


def test_live_mode_with_paper_enabled_stays_non_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'true')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    log_line = _run_and_get_mode_log(_build_runner(_OrderManagerNoMode()))
    assert 'is_live_mode=False' in log_line


def test_live_mode_with_shadow_enabled_stays_non_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'true')
    log_line = _run_and_get_mode_log(_build_runner(_OrderManagerNoMode()))
    assert 'is_live_mode=False' in log_line


def test_live_mode_requires_callable_true_and_env_alignment(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner = _build_runner(_OrderManagerWithLiveMode())
    log_line = _run_and_get_mode_log(runner)
    assert 'is_live_mode=True' in log_line
    assert 'execution_mode=LIVE' in log_line
    assert any('runtime_symbol_execution_not_ready' in (m or '') for m in runner._logger.info_messages)


def test_live_mode_uses_enable_live_fallback_with_callable_true(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'false')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('PAPER_MODE', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner = _build_runner(_OrderManagerWithLiveMode())
    log_line = _run_and_get_mode_log(runner)
    assert 'is_live_mode=True' in log_line
    assert 'execution_mode=LIVE' in log_line


def test_live_mode_uses_enable_live_fallback_without_callable_source(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'false')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('PAPER_MODE', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner = _build_runner(_OrderManagerNoMode())
    log_line = _run_and_get_mode_log(runner)
    assert 'is_live_mode=True' in log_line
    assert 'execution_mode=LIVE' in log_line


def test_paper_mode_true_overrides_paper_enabled_false(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('ENABLE_LIVE', 'false')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('PAPER_MODE', 'true')
    monkeypatch.setenv('SHADOW_MODE', 'false')
    runner = _build_runner(_OrderManagerNoMode())
    log_line = _run_and_get_mode_log(runner)
    assert 'is_live_mode=False' in log_line


def test_shadow_mode_y_forces_non_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('ENABLE_LIVE', 'false')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('PAPER_MODE', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'y')
    runner = _build_runner(_OrderManagerNoMode())
    log_line = _run_and_get_mode_log(runner)
    assert 'is_live_mode=False' in log_line
