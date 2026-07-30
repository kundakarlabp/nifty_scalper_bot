from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies import (
    order_flow_live_context_patch as live_context,
)
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


def _metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "trigger_block_reason": "direction_context_missing_live",
        "strategy_score": 8.5,
        "trigger_min_score": 8.0,
        "spread_pct": 0.2,
        "trigger_max_spread_pct": 0.75,
        "quote_readiness_allowed": True,
        "quote_depth_valid": True,
        "depth_available": True,
        "tradable_quote": True,
        "selected_or_near_atm": True,
    }
    metadata.update(overrides)
    return metadata


def test_live_context_upgrade_accepts_canonical_quote_age_seconds(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVE_MAX_TICK_AGE_MS", "2500")
    indicators = {
        "context_fresh": True,
        "spot_fresh": True,
        "futures_fresh": True,
    }

    assert live_context._can_upgrade_direction_context_block(
        _metadata(quote_age_s=0.10),
        indicators,
    )


def test_orderflow_evaluation_is_not_replaced_at_package_import() -> None:
    assert OrderFlowStrategy._evaluate_signal.__module__.endswith(".order_flow")
    assert not hasattr(
        OrderFlowStrategy,
        "_live_direction_context_proof_patch_installed",
    )
