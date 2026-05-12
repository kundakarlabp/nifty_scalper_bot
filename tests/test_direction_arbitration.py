from __future__ import annotations

from nifty_scalper_bot.strategies.orchestrator import StrategyOrchestrator


def test_direction_state_fields_exist_for_arbitration() -> None:
    orchestrator = StrategyOrchestrator(None)
    assert hasattr(orchestrator, '_active_direction')
    assert hasattr(orchestrator, '_direction_lock_time')
