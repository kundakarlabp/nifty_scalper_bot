from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.strategies.orchestrator import StrategyOrchestrator


CE = "NFO:NIFTY2690823950CE"
PE = "NFO:NIFTY2690823950PE"


class _Positions:
    def get_position(self, _symbol):
        return None

    def get_all_positions(self):
        return []


def _signal(symbol: str):
    return SimpleNamespace(symbol=symbol, action="BUY", confidence=0.90, metadata={})


def test_direction_lock_survives_exit_and_blocks_fast_opposite_reentry(monkeypatch) -> None:
    from nifty_scalper_bot.utils import market_hours

    monkeypatch.setattr(market_hours, "get_time_status_cached", lambda: (True, "open"))
    monkeypatch.setenv("DIRECTION_LOCK_SECONDS", "60")
    monkeypatch.setenv("SIGNAL_COOLDOWN_S", "0")
    monkeypatch.setenv("UNDERLYING_SIGNAL_COOLDOWN", "0")

    orchestrator = StrategyOrchestrator(
        risk_manager=SimpleNamespace(current_balance=100_000.0)
    )
    positions = _Positions()

    orchestrator.notify_entry(CE, reason="order_accepted")
    orchestrator.notify_exit("NIFTY")

    assert orchestrator._active_direction == "CE"
    assert orchestrator.filter_signal(_signal(CE), {}, positions) is not None

    blocked = orchestrator.filter_signal(_signal(PE), {}, positions)
    assert blocked is None
    assert orchestrator.get_skip_reason() == "direction_conflict"

    orchestrator._direction_lock_time -= 61.0
    allowed_after_expiry = orchestrator.filter_signal(_signal(PE), {}, positions)
    assert allowed_after_expiry is not None
    assert orchestrator._active_direction is None
