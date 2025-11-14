"""Unit tests for market regime manager indicator integration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from nifty_scalper_bot.core.market_regime import MarketRegimeDetector, RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager


@dataclass(slots=True)
class DummyIndicators:
    """Simple indicator provider used for testing."""

    atr_trend: float
    vol_index: float

    def get(self, key: str) -> float:
        """Return stored indicator value for *key*."""

        if key == "atr_trend":
            return self.atr_trend
        if key == "volatility_index":
            return self.vol_index
        raise KeyError(key)


@pytest.mark.asyncio
async def test_refresh_from_indicators_updates_snapshot() -> None:
    """Ensure indicator refresh updates regime state and history."""

    detector = MarketRegimeDetector()
    indicators = DummyIndicators(atr_trend=2.0, vol_index=10.0)
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

    assert manager.get_current_regime() == "TREND"
    history = manager.get_history(limit=5)
    assert any(snapshot.regime == "TREND" for snapshot in history)


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
