from __future__ import annotations

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


def test_validate_symbol_universe_rejects_missing_option_symbols() -> None:
    runner = _runner()
    runner._active_symbols = {'NSE:NIFTY 50'}

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_rejects_index_count_mismatch() -> None:
    runner = _runner()
    runner._active_symbols = {
        'NSE:NIFTY 50',
        'NSE:NIFTY BANK',
        'NIFTY25JAN25000CE',
        'NIFTY25JAN25000PE',
    }

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_accepts_index_and_options() -> None:
    runner = _runner()
    runner._active_symbols = {
        'NSE:NIFTY 50',
        'NIFTY25JAN25000CE',
        'NIFTY25JAN25000PE',
    }

    assert runner._validate_symbol_universe() is True
