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

        values = {str(name): 0.0 for name in names}
        values["vwap"] = 100.0
        values["exchange_vwap"] = 100.0
        values["volume"] = 10_000.0
        values["avg_volume"] = 8_000.0
        return values


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


class _SignalStrategy(_DummyStrategy):
    """Strategy stub that always returns a BUY signal."""

    def generate_signal(
        self,
        symbol: str,
        indicators: t.Mapping[str, t.Any],
        current_price: float,
        position: t.Any,
    ) -> Signal | None:
        return Signal(
            action='BUY',
            symbol=symbol,
            quantity=2,
            confidence=0.7,
            reason='test',
            stop_loss=current_price - 1,
            take_profit=current_price + 2,
            metadata={},
        )


class _FixedSymbolSignalStrategy(_DummyStrategy):
    """Strategy stub returning a BUY signal for a fixed option symbol side."""

    def __init__(self, name: str, option_symbol: str, confidence: float = 0.8) -> None:
        super().__init__(name)
        self._option_symbol = option_symbol
        self._confidence = confidence

    def generate_signal(
        self,
        symbol: str,
        indicators: t.Mapping[str, t.Any],
        current_price: float,
        position: t.Any,
    ) -> Signal | None:
        return Signal(
            action='BUY',
            symbol=self._option_symbol,
            quantity=1,
            confidence=self._confidence,
            reason='vote',
            stop_loss=current_price - 1,
            take_profit=current_price + 2,
            metadata={
                'direction_score': 8.0,
                'data_score': 8.0,
                'option_score': 8.0,
                'strategy_score': 8.0,
            },
        )


class _BlockedRegimeManager:
    def can_trade(self, context: t.Mapping[str, t.Any] | None = None) -> bool:
        return False

    def get_filter_reasons(self) -> list[str]:
        return ['regime_block_volatile']

    def get_latest_snapshot(self) -> t.Any:
        return None


def test_regime_block_scales_signal_when_adaptive_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('USE_REGIME_ADAPTIVE', 'true')
    manager = StrategyManager(
        [_SignalStrategy('Alpha')],
        _DummyIndicatorEngine(),
        _DummyPositionManager(),
        market_regime_manager=_BlockedRegimeManager(),
    )

    signal = manager.generate_signal('NFO:NIFTY26FEB22500CE', 100.0)

    assert signal is not None
    assert signal.action == 'BUY'
    assert signal.metadata.get('regime_scale') == pytest.approx(1.0)


def test_regime_block_can_still_block_when_adaptive_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('USE_REGIME_ADAPTIVE', 'false')
    manager = StrategyManager(
        [_SignalStrategy('Alpha')],
        _DummyIndicatorEngine(),
        _DummyPositionManager(),
        market_regime_manager=_BlockedRegimeManager(),
    )

    signal = manager.generate_signal('NFO:NIFTY26FEB22500CE', 100.0)

    assert signal is None


def test_strategy_manager_conflicting_ce_pe_votes_return_no_trade() -> None:
    manager = StrategyManager(
        [
            _FixedSymbolSignalStrategy('Alpha', 'NFO:NIFTY26FEB22500CE', 0.82),
            _FixedSymbolSignalStrategy('Beta', 'NFO:NIFTY26FEB22500PE', 0.84),
        ],
        _DummyIndicatorEngine(),
        _DummyPositionManager(),
    )

    signal = manager.generate_signal('NFO:NIFTY26FEB22500CE', 100.0)
    assert signal is None


def test_strategy_manager_same_side_votes_increase_strategy_score() -> None:
    manager = StrategyManager(
        [
            _FixedSymbolSignalStrategy('Alpha', 'NFO:NIFTY26FEB22500CE', 0.82),
            _FixedSymbolSignalStrategy('Beta', 'NFO:NIFTY26FEB22500CE', 0.86),
        ],
        _DummyIndicatorEngine(),
        _DummyPositionManager(),
    )

    signal = manager.generate_signal('NFO:NIFTY26FEB22500CE', 100.0)
    assert signal is not None
    assert float(signal.metadata.get('strategy_score') or 0.0) > 8.0
