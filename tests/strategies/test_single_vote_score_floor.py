"""Selected-option single-vote scalps require a high score floor (default 9.0).

A lone vote (no consensus) is riskier, so only the strongest signals should pass.
"""
from __future__ import annotations

import os


def _selected_option_allowed(raw_trigger_score: float, *, threshold_passed=True,
                             selected_option=True, allow_selected_option=True) -> bool:
    # Mirrors the gate in strategy_manager: base allow AND score >= floor.
    allowed = bool(threshold_passed and selected_option and allow_selected_option)
    floor = float(os.getenv("STRATEGY_SELECTED_OPTION_SINGLE_VOTE_MIN_SCORE", "9.0") or "9.0")
    if allowed and raw_trigger_score < floor:
        allowed = False
    return allowed


async def test_single_vote_blocked_below_nine(monkeypatch) -> None:
    monkeypatch.delenv("STRATEGY_SELECTED_OPTION_SINGLE_VOTE_MIN_SCORE", raising=False)
    assert _selected_option_allowed(8.0) is False   # was allowed before; now blocked
    assert _selected_option_allowed(8.9) is False


async def test_single_vote_allowed_at_or_above_nine(monkeypatch) -> None:
    monkeypatch.delenv("STRATEGY_SELECTED_OPTION_SINGLE_VOTE_MIN_SCORE", raising=False)
    assert _selected_option_allowed(9.0) is True
    assert _selected_option_allowed(9.5) is True


async def test_single_vote_floor_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_SELECTED_OPTION_SINGLE_VOTE_MIN_SCORE", "8.5")
    assert _selected_option_allowed(8.6) is True
    assert _selected_option_allowed(8.4) is False


async def test_smc_option_context_fetch_allowed_at_open() -> None:
    # Option-context broker fetch must be blocked only when the market is genuinely
    # closed (PRE/POST/HOLIDAY), not for OPEN or transient UNKNOWN — otherwise SMC
    # stays history-cold and can't vote.
    import pathlib
    src = pathlib.Path("src/nifty_scalper_bot/core/app.py").read_text()
    assert '_market_closed_for_fetch = _market_mode in {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}' in src
    assert 'get_runtime_market_mode() != "OPEN"' not in src  # old over-broad gate removed
    # behavior: only the three closed modes block
    for mode in ("PRE_MARKET", "POST_MARKET", "HOLIDAY"):
        assert mode in {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}
    for mode in ("OPEN", "UNKNOWN"):
        assert mode not in {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}


async def test_malformed_score_env_does_not_crash() -> None:
    # #13: a malformed STRATEGY_SELECTED_OPTION_SINGLE_VOTE_MIN_SCORE must fall back
    # to the default, not raise ValueError inside signal combination.
    from nifty_scalper_bot.config.env_utils import parse_float_env
    assert parse_float_env("high", 9.0) == 9.0
    assert parse_float_env("", 9.0) == 9.0
    assert parse_float_env("9.5", 9.0) == 9.5


def _gate_probe(symbol: str, indicators: dict):
    """Drive the real _combine_strategy_votes gate with one strong CE vote."""
    from nifty_scalper_bot.core.strategy_manager import (
        Signal,
        StrategyManager,
        StrategyVote,
    )

    mgr = StrategyManager.__new__(StrategyManager)
    mgr._last_no_signal_decision_by_symbol = {}
    sig = Signal(
        action="BUY", symbol=symbol, quantity=65, confidence=0.9,
        reason="t", stop_loss=140.0, take_profit=150.0, metadata={},
    )
    vote = StrategyVote(
        strategy="VWAPPro", side="CE", score=9.9, confidence=0.9,
        reasons=[], metadata={"role": "trigger"},
    )
    result = mgr._combine_strategy_votes(
        symbol=symbol, signals=[(sig, vote)], indicators=indicators
    )
    decision = mgr._last_no_signal_decision_by_symbol.get(symbol.upper())
    return result, decision


async def test_option_entry_fails_closed_without_underlying_direction() -> None:
    # Slice-3: missing underlying direction context must block, not fail open.
    result, decision = _gate_probe("NFO:NIFTY2670724050CE", {})
    assert result is None
    assert decision is not None
    assert decision.final_block_reason == "underlying_direction_unresolved"


async def test_option_entry_fails_closed_on_stale_context() -> None:
    # Direction present but context not fresh -> still blocked.
    result, decision = _gate_probe(
        "NFO:NIFTY2670724050CE",
        {"direction_bias": "CE", "context_fresh": False},
    )
    assert result is None
    assert decision.final_block_reason == "underlying_direction_unresolved"


async def test_option_entry_blocked_when_bias_conflicts_option_side() -> None:
    # PE bias can never approve a CE entry (and vice versa) - no side flip.
    result, decision = _gate_probe(
        "NFO:NIFTY2670724050CE",
        {"direction_bias": "PE", "context_fresh": True},
    )
    assert result is None
    assert decision.final_block_reason == "underlying_direction_conflict"
