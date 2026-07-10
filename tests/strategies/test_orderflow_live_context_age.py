from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.order_flow_live_context_patch import (
    _can_upgrade_direction_context_block,
)


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

    assert _can_upgrade_direction_context_block(
        _metadata(quote_age_s=0.10),
        indicators,
    )
