"""OrderFlow adaptive direction-conflict tests.

A stale directional bias must not veto fresh context after persistent live
microstructure independently confirms the candidate option side.
"""

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


def _ind(bias, tick, buy, sell, **kw):
    d = {
        "bid": 100.0,
        "ask": 100.25,
        "spread_pct": 0.24,
        "depth": {"buy": [{"quantity": buy}], "sell": [{"quantity": sell}]},
        "tick_direction": tick,
        "direction_bias": bias,
        "atr": 2.0,
        "data_age_seconds": 0.1,
        "context_age_seconds": 1.0,
        "tick_age_ms": 100,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "is_selected_option": True,
        "strike_distance_from_atm": 0,
        "quote_update_version": 1,
        "stale_data_used": False,
    }
    d.update(kw)
    return d


@pytest.fixture
def strat():
    return OrderFlowStrategy(
        OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None
    )


def _eval(strat, sym, ind):
    return strat._evaluate_signal(sym, ind, current_price=100.1)


# OrderFlow is context-only: microstructure may corroborate the canonical
# underlying side, but it can never reverse that direction or authorize entry.


def test_opposing_underlying_direction_remains_conflict(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=400, sell=80))

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
    assert sig.metadata["context_quality_eligible"] is True
    assert sig.metadata["effective_context_conflict"] is True
    assert sig.metadata["effective_context_alignment"] is False


def test_persistent_microstructure_cannot_override_underlying_direction(
    monkeypatch, strat
):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")

    results = [
        _eval(
            strat,
            "NFO:NIFTY26MAY24000PE",
            _ind("CE", "UP", buy=400, sell=80, quote_update_version=version),
        )
        for version in (1, 2, 3)
    ]

    assert all(r.metadata["trigger_conditions_met"] is False for r in results)
    assert all(
        r.metadata["trigger_block_reason"] == "context_only_role" for r in results
    )
    assert all(r.metadata["effective_context_conflict"] is True for r in results)
    assert all(r.metadata["effective_context_alignment"] is False for r in results)


def test_aligned_direction_and_microstructure_publish_confirmation(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("CE", "UP", buy=400, sell=80))

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
    assert sig.metadata["context_quality_eligible"] is True
    assert sig.metadata["effective_context_alignment"] is True
    assert sig.metadata["effective_context_conflict"] is False


def test_weak_microstructure_does_not_create_alignment(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("CE", "UP", buy=210, sell=180))

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["context_quality_eligible"] is True
    assert sig.metadata["effective_context_alignment"] is False
    assert sig.metadata["effective_context_conflict"] is False


def test_missing_underlying_direction_makes_context_ineligible(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("", "UP", buy=400, sell=80))

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
    assert sig.metadata["context_quality_eligible"] is False
    assert sig.metadata["effective_context_alignment"] is False
    assert sig.metadata["effective_context_conflict"] is False
