"""Unit tests for market regime manager indicator integration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from nifty_scalper_bot.core.market_regime import MarketRegimeDetector, RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager


@dataclass(slots=True)
class DummyIndicators:
    """Indicator provider that speaks the production get_indicators contract."""

    def get_indicators(self, symbol: str, names=None) -> dict[str, float]:
        del symbol, names
        return {
            "ema_fast": 102.0,
            "ema_slow": 100.0,
            "adx": 32.0,
            "atr": 14.0,
            "volume_spike_ratio": 1.1,
            "iv_rank": 10.0,
            "price": 100.0,
            "close": 100.0,
            "price_momentum": 0.02,
        }


@pytest.mark.asyncio
async def test_refresh_from_indicators_updates_snapshot() -> None:
    """Ensure indicator refresh updates regime state and history."""

    detector = MarketRegimeDetector()
    indicators = DummyIndicators()
    manager = MarketRegimeManager(
        detector,
        indicators=indicators,
        regime_settings={
            "symbol": "NIFTY",
            "atr_trend_threshold": 1.5,
            "vol_threshold": 25.0,
        },
    )

    await manager.refresh_from_indicators()

    assert manager.get_current_regime() == "trend"
    history = manager.get_history(limit=5)
    assert any(snapshot.regime == "trend" for snapshot in history)


@pytest.mark.asyncio
async def test_refresh_from_indicators_does_not_fabricate_features() -> None:
    """A mapping-style .get() provider must not invent ema/price evidence."""

    class MappingOnlyIndicators:
        def get(self, key: str) -> float:
            return {"atr_trend": 2.0, "volatility_index": 10.0}[key]

    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(
        detector,
        indicators=MappingOnlyIndicators(),
        regime_settings={"symbol": "NIFTY"},
    )

    await manager.refresh_from_indicators()

    assert manager.get_latest_snapshot() is None


def test_can_trade_respects_bypass_toggle() -> None:
    """Verify bypass toggle allows trading even when regime blocks it."""

    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)
    manager.block_thresholds["VOLATILE"] = 0.5
    snapshot = RegimeSnapshot(
        symbol="NIFTY",
        regime="VOLATILE",
        confidence=0.9,
        reason="vol spike",
        updated_at=time.time(),
        adjustments={},
    )
    manager.ingest_snapshot(snapshot)

    assert manager.can_trade(record_decision=False) is False
    assert manager.toggle_bypass() is True
    assert manager.can_trade(record_decision=False) is True
