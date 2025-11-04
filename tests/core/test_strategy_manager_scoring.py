"""Tests for strategy performance scoring and dynamic allocation."""

from __future__ import annotations

import typing as t

import pytest

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyPerformance
from nifty_scalper_bot.strategies.signal_generator import Signal


class _DummyStrategy:
    """Minimal strategy stub used for score verification."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_required_indicators(self) -> list[str]:
        """Return no required indicators for the stub."""

        return []

    def generate_signal(
        self,
        symbol: str,
        indicators: t.Mapping[str, t.Any],
        current_price: float,
        position: t.Any,
    ) -> Signal | None:
        """Return no signal for deterministic tests."""

        return None


class _DummyIndicatorEngine:
    """Indicator engine stub returning empty indicator snapshots."""

    def get_indicators(
        self, symbol: str, names: t.Iterable[str]
    ) -> dict[str, float | None]:
        """Return indicators as zeroed values for requested names."""

        return {str(name): 0.0 for name in names}


class _DummyPositionManager:
    """Position manager stub always reporting no positions."""

    def get_position(self, symbol: str) -> None:
        """Return ``None`` to indicate no active position."""

        return None


def test_strategy_performance_metrics() -> None:
    """StrategyPerformance aggregates sharpe, hit rate, and drawdown."""

    perf = StrategyPerformance()
    perf.record(120.0)
    perf.record(-30.0)
    perf.record(-70.0)

    assert perf.total_pnl == pytest.approx(20.0)
    assert perf.rolling_pnl() == pytest.approx(20.0)
    assert perf.hit_rate() == pytest.approx(1.0 / 3.0)
    assert perf.max_drawdown() == pytest.approx(100.0)
    assert perf.sharpe_ratio() > 0.0


def test_strategy_manager_dynamic_allocation_and_disable() -> None:
    """StrategyManager weights strategies by regime and disable state."""

    strategies = [_DummyStrategy("Alpha"), _DummyStrategy("Beta")]
    indicator_engine = _DummyIndicatorEngine()
    position_manager = _DummyPositionManager()

    def _regime_snapshot() -> dict[str, t.Any]:
        """Return fixed bull regime snapshot used in tests."""

        return {"regime": "bull", "confidence": 0.5}

    manager = StrategyManager(
        strategies,
        indicator_engine,
        position_manager,
        regime_signal_getter=_regime_snapshot,
        regime_bias_map={"bull": {"Alpha": 1.5, "Beta": 0.5}},
    )

    manager.record_trade_result("Alpha", 80.0)
    manager.record_trade_result("Alpha", 40.0)
    manager.record_trade_result("Beta", -30.0)
    manager.record_trade_result("Beta", -20.0)

    score_map = {entry.strategy: entry for entry in manager.get_strategy_scores()}
    alpha_entry = score_map["Alpha"]
    beta_entry = score_map["Beta"]

    assert alpha_entry.enabled is True
    assert beta_entry.enabled is True
    assert alpha_entry.drawdown == pytest.approx(0.0)
    assert beta_entry.drawdown == pytest.approx(50.0)
    assert alpha_entry.weight == pytest.approx(alpha_entry.score * 1.25)
    assert beta_entry.weight == pytest.approx(beta_entry.score * 0.75)

    allocations = manager.get_allocation_snapshot()
    assert set(allocations) == {"Alpha", "Beta"}
    assert sum(allocations.values()) == pytest.approx(1.0)

    assert manager.disable_strategy("Beta") is True
    disabled_snapshot = manager.disabled_strategies()
    assert "Beta" in disabled_snapshot
    scores_after = manager.get_strategy_scores()
    beta_after = next(entry for entry in scores_after if entry.strategy == "Beta")
    assert beta_after.enabled is False
    assert beta_after.weight == pytest.approx(0.0)
    allocations_after = manager.get_allocation_snapshot()
    assert "Beta" not in allocations_after

    assert manager.enable_strategy("Beta") is True
    assert manager.is_strategy_enabled("Beta") is True
    final_allocations = manager.get_allocation_snapshot()
    assert "Beta" in final_allocations
    assert sum(final_allocations.values()) == pytest.approx(1.0)
