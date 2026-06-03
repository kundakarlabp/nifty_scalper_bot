"""OrderFlow adaptive direction-conflict tests.

A stale directional bias must not veto a fresh signal when live microstructure
(tick + same-side depth imbalance) independently confirms the candidate side.
"""
import os
import pytest
from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


def _ind(bias, tick, buy, sell, **kw):
    d = {
        "bid": 100.0, "ask": 100.25, "spread_pct": 0.24,
        "depth": {"buy": [{"quantity": buy}], "sell": [{"quantity": sell}]},
        "tick_direction": tick, "direction_bias": bias, "atr": 2.0,
        "data_age_seconds": 0.1, "context_age_seconds": 1.0, "tick_age_ms": 100,
        "quote_depth_valid": True, "tradable_quote": True,
        "is_selected_option": True, "strike_distance_from_atm": 0,
    }
    d.update(kw)
    return d


@pytest.fixture
def strat():
    return OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)


def _eval(strat, sym, ind):
    return strat._evaluate_signal(sym, ind, current_price=100.1)


# A. Stale PE bias + CE candidate WITHOUT confirming microstructure -> blocked
def test_stale_pe_weak_micro_ce_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=150, sell=140))
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "direction_bias_conflict"
    assert sig.metadata["bias_invalidated_by_microstructure"] is False


# B. Stale PE bias + CE candidate WITH confirming microstructure -> allowed
def test_stale_pe_strong_micro_ce_allowed(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=400, sell=80))
    assert sig.metadata["trigger_conditions_met"] is True
    assert sig.metadata["bias_invalidated_by_microstructure"] is True


# C. Reversal down: stale CE bias + PE candidate with confirming PE demand -> allowed
def test_stale_ce_strong_micro_pe_allowed(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000PE", _ind("CE", "UP", buy=400, sell=80))
    assert sig.metadata["trigger_conditions_met"] is True
    assert sig.metadata["bias_invalidated_by_microstructure"] is True


# D. Tick contradicts candidate side -> not invalidated -> blocked
def test_tick_contradicts_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "DOWN", buy=400, sell=80))
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["bias_invalidated_by_microstructure"] is False


# E. Aligned bias (CE bias, CE candidate) -> allowed normally, not via invalidation
def test_aligned_bias_allowed_normally(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("CE", "UP", buy=400, sell=80))
    assert sig.metadata["trigger_conditions_met"] is True
    assert sig.metadata["bias_invalidated_by_microstructure"] is False


# F. Below default imbalance threshold -> not confirmed -> blocked
def test_below_imbalance_threshold_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    # imbalance (210-180)/390 = 0.077 < 0.20 default -> not confirmed
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=210, sell=180))
    assert sig.metadata["bias_invalidated_by_microstructure"] is False
    assert sig.metadata["trigger_conditions_met"] is False


# G. No directional bias at all -> not gated by conflict
def test_no_bias_not_conflict_gated(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("", "UP", buy=400, sell=80))
    assert sig.metadata["trigger_block_reason"] != "direction_bias_conflict"
