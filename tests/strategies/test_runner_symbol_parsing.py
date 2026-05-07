from __future__ import annotations

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_extract_strike_from_symbol_parses_nfo_options() -> None:
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAY24350CE') == 24350
    assert StrategyRunner._extract_strike_from_symbol('nfo:nifty26may24200pe') == 24200


def test_extract_strike_from_symbol_returns_none_for_fut_and_spot() -> None:
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAY24FUT') is None
    assert StrategyRunner._extract_strike_from_symbol('NSE:NIFTY') is None


def test_premium_squeeze_near_atm_filter_does_not_raise_missing_helper() -> None:
    assert hasattr(StrategyRunner, '_extract_strike_from_symbol')


def test_strategy_runner_required_helpers_exist() -> None:
    required_helpers = [
        '_extract_underlying',
        '_extract_strike_from_symbol',
        '_get_atr_with_fallback',
        '_handle_signal',
        '_handle_entry_signal_inner',
    ]
    for helper_name in required_helpers:
        assert hasattr(StrategyRunner, helper_name)
