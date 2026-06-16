from __future__ import annotations

import logging
from pathlib import Path
import time
from collections import defaultdict

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_set_runtime_readiness_propagates_selected_option_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'selected_ce: str | None = None' in source
    assert 'self.set_active_option_context(' in source


def test_set_active_option_context_normalizes_and_stores_values() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'RUNNER_ACTIVE_OPTION_CONTEXT' in source
    assert 'normalize_symbol(str(selected_ce))' in source


def test_extract_strike_from_symbol_parses_options() -> None:
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAY24250CE') == 24250
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAY24250PE') == 24250
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAYFUT') is None
    assert StrategyRunner._extract_strike_from_symbol('NSE:NIFTY') is None


def test_premium_squeeze_uses_active_atm_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'reason=outside_selected_strike_window symbol=%s selected_ce=%s selected_pe=%s atm_strike=%s' in source


def test_premium_squeeze_missing_context_logs_missing_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'reason=missing_active_option_context' in source


def test_signal_evaluation_failure_logs_error_type() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'SIGNAL_EVALUATION_FAILURE symbol=%s phase=%s error_type=%s error=%s trace_id=%s' in source


def test_runtime_readiness_preserves_execution_map_when_not_supplied() -> None:
    runner = object.__new__(StrategyRunner)
    runner._runtime_execution_ready_by_symbol = {}
    runner._runtime_data_hard_ready = False
    runner._runtime_evaluation_ready = False
    runner._runtime_live_orders_armed = False
    runner._runtime_readiness_reason = None
    runner._runtime_startup_ready = False
    runner._active_selected_ce = None
    runner._active_selected_pe = None
    runner._active_atm_strike = None
    runner._active_option_symbols = set()
    runner._logger = logging.getLogger('test')
    runner.set_runtime_readiness(
        data_hard_ready=True,
        evaluation_ready=True,
        live_orders_armed=True,
        execution_ready_by_symbol={'NFO:XCE': True},
    )
    runner.set_runtime_readiness(
        data_hard_ready=True,
        evaluation_ready=True,
        live_orders_armed=False,
    )
    assert runner._runtime_execution_ready_by_symbol.get('NFO:XCE') is True


def test_set_runtime_readiness_logs_without_typeerror(caplog) -> None:
    runner = object.__new__(StrategyRunner)
    runner._runtime_execution_ready_by_symbol = {}
    runner._runtime_data_hard_ready = False
    runner._runtime_evaluation_ready = False
    runner._runtime_live_orders_armed = False
    runner._runtime_readiness_reason = None
    runner._runtime_startup_ready = False
    runner._active_selected_ce = None
    runner._active_selected_pe = None
    runner._active_atm_strike = None
    runner._active_option_symbols = set()
    runner._logger = logging.getLogger('test.readiness.log')
    with caplog.at_level(logging.INFO):
        runner.set_runtime_readiness(
            data_hard_ready=True,
            evaluation_ready=True,
            live_orders_armed=True,
            reason="ok",
            selected_ce="NFO:NIFTY26MAY24200CE",
            selected_pe="NFO:NIFTY26MAY24200PE",
        )
    assert any("RUNNER_STARTUP_READINESS_UPDATE" in rec.message for rec in caplog.records)


def test_strategy_eval_stall_recomputes_readiness_once(monkeypatch) -> None:
    calls: list[str] = []
    starts: list[bool] = []
    runner = object.__new__(StrategyRunner)
    runner.ready = True
    runner._last_tick_seen_ts = time.monotonic()
    runner._last_global_eval_ts = time.monotonic() - 91.0
    runner._eval_stall_recovery_attempted = False
    runner._runtime_readiness_recompute_callback = lambda reason: calls.append(reason)
    runner._logger = logging.getLogger('test.stall.recovery')
    runner._candle_engines = {}
    runner._active_symbols = set()
    runner._last_eval_ts = defaultdict(float)
    runner._last_periodic_eval_at_by_symbol = {'NFO:XCE': time.monotonic()}
    runner.start = lambda: starts.append(True)

    monkeypatch.setattr('nifty_scalper_bot.strategies.runner.is_market_open_now', lambda: True)

    runner._health_watchdog()
    runner._health_watchdog()

    assert calls == ['strategy_eval_stall_watchdog']
    assert starts == [True]
