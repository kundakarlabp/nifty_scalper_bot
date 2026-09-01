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


# A. Stale PE bias + CE candidate WITHOUT confirming microstructure -> blocked
def test_stale_pe_weak_micro_ce_blocked(monkeypatch, strat, caplog):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    caplog.set_level("INFO")
    sig = _eval(
        strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=150, sell=140)
    )
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
    assert sig.metadata["bias_invalidated_by_microstructure"] is False
    assert sig.metadata["context_bonus_score"] == 0.0
    assert sig.metadata["context_veto_score"] > 0.0


# B. One strong snapshot cannot invalidate directional context
def test_stale_pe_single_strong_micro_ce_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", "0")
    sig = _eval(
        strat,
        "NFO:NIFTY26MAY24000CE",
        _ind("PE", "UP", buy=400, sell=80, quote_update_version=1),
    )
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["bias_invalidated_by_microstructure"] is False
    assert sig.metadata["context_bonus_score"] == 0.0
    assert sig.metadata["context_veto_score"] > 0.0


# C. Reversal becomes eligible only after distinct persistent updates
def test_stale_ce_strong_micro_pe_requires_persistence(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", "0")
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
    assert results[0].metadata["context_veto_score"] > 0.0
    assert results[1].metadata["context_veto_score"] > 0.0
    assert results[0].metadata["context_bonus_score"] == 0.0
    assert results[1].metadata["context_bonus_score"] == 0.0

    assert results[2].metadata["bias_invalidated_by_microstructure"] is True
    assert results[2].metadata["reversal_persistence_confirmed"] is True
    assert results[2].metadata["context_quality_eligible"] is True
    assert results[2].metadata["context_veto_score"] == 0.0
    assert results[2].metadata["context_bonus_score"] > 0.0


# D. Tick contradicts candidate side -> not invalidated -> blocked
def test_tick_contradicts_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "DOWN", buy=400, sell=80)
    )
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["bias_invalidated_by_microstructure"] is False


# E. Aligned bias remains context only
def test_aligned_bias_allowed_normally(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat, "NFO:NIFTY26MAY24000CE", _ind("CE", "UP", buy=400, sell=80)
    )
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["bias_invalidated_by_microstructure"] is False
    assert sig.metadata["context_quality_eligible"] is True
    assert sig.metadata["context_bonus_score"] > 0.0
    assert sig.metadata["context_veto_score"] == 0.0


# F. Below default imbalance threshold -> not confirmed -> blocked
def test_below_imbalance_threshold_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=210, sell=180)
    )
    assert sig.metadata["bias_invalidated_by_microstructure"] is False
    assert sig.metadata["trigger_conditions_met"] is False


# G. No directional bias at all -> not gated by conflict
def test_no_bias_not_conflict_gated(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat, "NFO:NIFTY26MAY24000CE", _ind("", "UP", buy=400, sell=80)
    )
    assert sig.metadata["trigger_block_reason"] != "direction_bias_conflict"


def test_no_bias_without_spot_or_futures_live_proof_still_blocks(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat, "NFO:NIFTY26MAY24000CE", _ind("", "UP", buy=400, sell=80)
    )

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
    assert sig.metadata.get("direction_context_live_proof") is not True


def test_no_bias_with_fresh_spot_live_proof_stays_context(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat,
        "NFO:NIFTY26MAY24000CE",
        _ind(
            "",
            "UP",
            buy=400,
            sell=80,
            spot_fresh=True,
            spot_tick_age_s=0.25,
        ),
    )

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
    assert sig.metadata["direction_context_ok"] is False


def test_no_bias_with_stale_spot_and_futures_proof_still_blocks(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(
        strat,
        "NFO:NIFTY26MAY24000CE",
        _ind(
            "",
            "UP",
            buy=400,
            sell=80,
            spot_fresh=False,
            fut_fresh=False,
            spot_tick_age_s=30.0,
            futures_tick_age_s=30.0,
        ),
    )

    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["trigger_block_reason"] == "context_only_role"
