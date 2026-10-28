from __future__ import annotations

from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    _strict_replacement_expiry_block_reason,
)


def test_extract_strike_from_symbol_parses_nfo_options() -> None:
    assert StrategyRunner._extract_strike_from_symbol("NFO:NIFTY26MAY24350CE") == 24350
    assert StrategyRunner._extract_strike_from_symbol("nfo:nifty26may24200pe") == 24200


def test_extract_strike_from_symbol_returns_none_for_fut_and_spot() -> None:
    assert StrategyRunner._extract_strike_from_symbol("NFO:NIFTY26MAY24FUT") is None
    assert StrategyRunner._extract_strike_from_symbol("NSE:NIFTY") is None


def test_extract_strike_helper_remains_available() -> None:
    assert hasattr(StrategyRunner, "_extract_strike_from_symbol")


def test_strategy_runner_required_helpers_exist() -> None:
    required_helpers = [
        "_extract_underlying",
        "_extract_strike_from_symbol",
        "_handle_signal",
        "_handle_entry_signal_inner",
    ]
    for helper_name in required_helpers:
        assert hasattr(StrategyRunner, helper_name)


def test_strict_replacement_requires_both_matching_expiry_identities() -> None:
    check = _strict_replacement_expiry_block_reason
    assert check("NIFTY26OCT", "NIFTY26OCT", strict=True) is None
    assert check("NIFTY26OCT", "NIFTY26NOV", strict=True) == "expiry_mismatch"
    assert check(None, "NIFTY26OCT", strict=True) == "expiry_unverified"
    assert check("NIFTY26OCT", None, strict=True) == "expiry_unverified"
    assert check("", "", strict=True) == "expiry_unverified"
    # Explicitly non-strict compatibility is not secretly rewritten.
    assert check(None, "NIFTY26OCT", strict=False) is None
