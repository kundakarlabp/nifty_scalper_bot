from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.runtime_context_contract import (
    live_direction_context_has_proof,
)


def test_fresh_live_context_is_proof_not_trigger_authority() -> None:
    context = {
        "context_fresh": True,
        "spot_fresh": True,
        "futures_fresh": True,
        "quote_age_s": 0.10,
    }

    assert live_direction_context_has_proof(context)
    # Fresh underlying context is evidence only. OrderFlow entry authority is
    # intentionally absent; no sidecar may promote the strategy to a trigger.
    assert OrderFlowStrategy._evaluate_signal.__module__.endswith(".order_flow")
    assert not hasattr(
        OrderFlowStrategy,
        "_live_direction_context_proof_patch_installed",
    )


def test_stale_live_context_fails_closed() -> None:
    context = {
        "spot_age_seconds": 10.0,
        "futures_age_seconds": 10.0,
        "context_age_seconds": 10.0,
    }

    assert not live_direction_context_has_proof(context, max_age_seconds=5.0)
