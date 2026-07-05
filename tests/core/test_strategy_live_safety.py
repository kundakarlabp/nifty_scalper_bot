from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core import strategy_live_safety as guard
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Engine:
    def __init__(self, bars: int) -> None:
        self.bars = bars

    def get_history(self, _symbol: str):
        return [{} for _ in range(self.bars)]

    def get_indicators(self, _symbol: str, _names):
        return {}


class _Positions:
    def get_position(self, _symbol: str):
        return None


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")


def _signal(**metadata) -> Signal:
    return Signal(
        action="BUY",
        symbol="NFO:NIFTY2662324050CE",
        quantity=1,
        confidence=0.9,
        reason="test",
        stop_loss=90.0,
        take_profit=120.0,
        metadata=dict(metadata),
    )


def test_live_strategy_manager_fails_closed_on_cold_history(monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("OPTION_EVAL_MIN_BARS", "10")
    manager = StrategyManager([], _Engine(0), _Positions())
    manager._required_candles = 10

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0, trace_id="cold-history")

    assert result is None
    decision = manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE")
    assert decision is not None
    assert decision.reason == "live_indicators_not_ready"
    assert decision.category == "live_safety"


def test_live_strategy_manager_fails_closed_when_hub_not_ready(monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("OPTION_EVAL_MIN_BARS", "2")
    manager = StrategyManager([], _Engine(5), _Positions())
    manager._required_candles = 2
    manager._data_hub = SimpleNamespace(indicators_ready=False)

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0, trace_id="hub-not-ready")

    assert result is None
    decision = manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE")
    assert decision is not None
    assert decision.reason == "live_hub_indicators_not_ready"


def test_live_approved_signal_must_pass_final_filter(monkeypatch) -> None:
    _live_env(monkeypatch)
    manager = SimpleNamespace(
        _filter_signal=lambda _signal: False,
        _orchestrator=None,
        _position_manager=None,
        _last_no_signal_decision_by_symbol={},
    )

    result = guard._final_filter(manager, _signal(is_approved=True, timestamp="2026-07-05T09:20:00"), "approved")

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol["NFO:NIFTY2662324050CE"]
    assert decision.reason == "live_signal_final_filter_block"


def test_live_signal_identity_is_required_and_enriched() -> None:
    missing = _signal(strategy="SMC")
    present = _signal(strategy="SMC", timestamp="2026-07-05T09:20:00")

    assert guard._has_identity(missing) is False
    assert guard._has_identity(present) is True
    enriched = guard._add_identity(present)
    assert enriched.metadata["deterministic_signal_id"] == present.deterministic_id
