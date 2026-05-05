from __future__ import annotations

from nifty_scalper_bot.strategies.runner import RunnerState, StrategyRunner


class StubRiskManager:
    def suggest_position_size(self, **kwargs) -> int:
        return int(kwargs.get('requested_quantity', 0))

    def validate_new_position(self, **kwargs):
        return True, None

    def validate_close_position(self, **kwargs):
        return True, None


class StubStrategyManager:
    pass


class StubIndicatorEngine:
    pass


class StubMarketDataManager:
    pass


def _build_runner() -> StrategyRunner:
    return StrategyRunner(
        market_data_manager=StubMarketDataManager(),
        indicator_engine=StubIndicatorEngine(),
        strategy_manager=StubStrategyManager(),
        risk_manager=StubRiskManager(),
        order_manager=None,
        position_manager=None,
    )


def test_runner_start_preserves_execution_enabled_state_and_warmup() -> None:
    runner = _build_runner()
    runner._active_symbols = {'NSE:NIFTY'}
    runner._runner_state = RunnerState.EXECUTION_ENABLED
    runner.ready = True
    runner._vwap_state = {'NSE:NIFTY': {'vwap': 123.45}}
    runner._symbol_bar_count = {'NSE:NIFTY': 42}
    runner._hydration_ready_streak = {'NSE:NIFTY': 5}
    runner._history_ready_by_symbol = {'NSE:NIFTY': True}

    runner.start()

    assert runner._runner_state == RunnerState.EXECUTION_ENABLED
    assert runner._vwap_state == {'NSE:NIFTY': {'vwap': 123.45}}
    assert runner._symbol_bar_count == {'NSE:NIFTY': 42}
    assert runner._hydration_ready_streak == {'NSE:NIFTY': 5}
    assert runner._history_ready_by_symbol == {'NSE:NIFTY': True}


def test_runner_start_with_no_active_symbols_keeps_execution_state() -> None:
    runner = _build_runner()
    runner._active_symbols = set()
    runner._runner_state = RunnerState.EXECUTION_ENABLED

    runner.start()

    assert runner._running is False
    assert runner._runner_state == RunnerState.EXECUTION_ENABLED
