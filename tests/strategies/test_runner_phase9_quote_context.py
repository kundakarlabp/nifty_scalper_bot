from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


def _runner() -> StrategyRunner:
    return StrategyRunner(
        market_data_manager=MagicMock(),
        indicator_engine=MagicMock(),
        strategy_manager=MagicMock(),
        risk_manager=MagicMock(),
        order_manager=MagicMock(),
        position_manager=MagicMock(),
        config=StrategyRunnerConfig(max_trade_history=5),
        data_hub=MagicMock(),
    )


def test_runner_imports_suppress_for_phase9_quote_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')

    assert 'from contextlib import suppress' in source
    assert 'with suppress(Exception):' in source


def test_strategy_eval_allowed_reports_integrity_failure_reason() -> None:
    runner = _runner()
    symbol = 'NFO:NIFTY26MAY25000FUT'
    runner._active_symbols = {symbol}
    runner._required_bars_for_symbol = MagicMock(return_value=5)
    runner._indicator_engine.get_history.return_value = [1] * 50
    runner._indicator_engine.has_min_bars.return_value = False
    emitted: list[str] = []
    runner._emit_runner_eval_decision = lambda **kwargs: emitted.append(kwargs.get('reason'))

    allowed = runner._strategy_evaluation_allowed(symbol, trace_id='t1')

    assert allowed is False
    assert emitted[-1] == 'indicator_history_integrity_failed'


def test_strategy_eval_allowed_reports_insufficient_bar_count_reason() -> None:
    runner = _runner()
    symbol = 'NFO:NIFTY26MAY25000CE'
    runner._active_symbols = {symbol}
    runner._required_bars_for_symbol = MagicMock(return_value=20)
    runner._indicator_engine.get_history.return_value = [1, 2]
    runner._indicator_engine.has_min_bars.return_value = False
    emitted: list[str] = []
    runner._emit_runner_eval_decision = lambda **kwargs: emitted.append(kwargs.get('reason'))

    allowed = runner._strategy_evaluation_allowed(symbol, trace_id='t2')

    assert allowed is False
    assert emitted[-1] == 'insufficient_indicator_bar_count'
