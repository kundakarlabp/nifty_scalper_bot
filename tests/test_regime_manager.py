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


@pytest.mark.asyncio
async def test_indicator_refresh_does_not_double_ingest_transition(monkeypatch) -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(
        detector,
        indicators=DummyIndicators(),
        transition_confirmations=2,
    )
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=0.80,
            reason="baseline",
            updated_at=time.time(),
            adjustments={},
        )
    )
    candidate = RegimeSnapshot(
        symbol="NIFTY",
        regime="range",
        confidence=0.70,
        reason="single-observation",
        updated_at=time.time(),
        adjustments={},
    )

    def _evaluate(*args, **kwargs):
        del args, kwargs
        detector._record_snapshot(candidate)
        return candidate

    monkeypatch.setattr(detector, "evaluate", _evaluate)
    await manager.refresh_from_indicators()

    assert manager.get_current_regime() == "trend"
    diagnostics = manager.build_diagnostics()
    assert diagnostics["transition"]["pending_regime"] == "RANGE"
    assert diagnostics["transition"]["pending_count"] == 1


@pytest.mark.asyncio
async def test_refresh_ignores_price_only_inputs_instead_of_fabricating_range() -> None:
    class PriceOnlyIndicators:
        def get_indicators(self, symbol: str, names=None) -> dict[str, float]:
            del symbol, names
            return {"price": 22750.0, "close": 22750.0}

    manager = MarketRegimeManager(
        MarketRegimeDetector(),
        indicators=PriceOnlyIndicators(),
    )

    await manager.refresh_from_indicators()

    assert manager.get_latest_snapshot() is None
    assert manager.get_raw_snapshot() is None


@pytest.mark.asyncio
async def test_verified_completed_session_history_feeds_range_momentum() -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    class HistoryIndicators:
        def get_indicators(self, symbol, names=None):
            return {"adx": 10.0, "atr": 4.0, "volume_spike_ratio": 1.0}

        def get_history(self, symbol, count=None, *, field="close"):
            assert field == "bars" and count == 16
            last = datetime.now(ZoneInfo("Asia/Kolkata")).replace(
                second=0, microsecond=0
            ) - timedelta(minutes=1)
            return [
                {
                    "timestamp": last - timedelta(minutes=idx),
                    "close": 25000.0 + (idx % 2) * 0.1,
                    "is_complete": True,
                    "is_provisional": False,
                }
                for idx in reversed(range(8))
            ]

    manager = MarketRegimeManager(
        MarketRegimeDetector(), indicators=HistoryIndicators()
    )
    await manager.refresh_from_indicators()
    assert manager.get_raw_snapshot().regime == "range"


@pytest.mark.asyncio
async def test_stale_or_unfinished_history_never_proves_range() -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    class StaleHistoryIndicators:
        def get_indicators(self, symbol, names=None):
            return {"adx": 10.0, "atr": 4.0}

        def get_history(self, symbol, count=None, *, field="close"):
            stamp = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(minutes=15)
            return [
                {
                    "timestamp": stamp - timedelta(minutes=i),
                    "close": 25000.0,
                    "is_complete": i > 0,
                    "is_provisional": i == 0,
                }
                for i in range(8)
            ]

    manager = MarketRegimeManager(
        MarketRegimeDetector(), indicators=StaleHistoryIndicators()
    )
    await manager.refresh_from_indicators()
    assert manager.get_raw_snapshot().regime == "unknown"
    assert manager.get_raw_snapshot().reason == "range_evidence_unverified"


def test_unknown_raw_observation_retains_stable_trend_but_filter_denies() -> None:
    """Retained strategy regime and the latest contradictory evidence are distinct.

    A new UNKNOWN reading cannot manufacture RANGE or erase hysteresis,
    but the manager must still expose the raw fail-closed gate result.
    """
    manager = MarketRegimeManager(MarketRegimeDetector())
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=0.82,
            reason="adx_trending",
            updated_at=time.time(),
            adjustments={},
        )
    )
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="unknown",
            confidence=0.0,
            reason="range_evidence_unverified",
            updated_at=time.time(),
            adjustments={},
        )
    )

    assert manager.get_latest_snapshot().regime == "trend"
    assert manager.get_raw_snapshot().regime == "unknown"
    assert manager.can_trade(record_decision=False) is False
    diagnostics = manager.build_diagnostics()
    assert diagnostics["current"]["regime"] == "trend"
    assert diagnostics["raw"]["regime"] == "unknown"
    assert diagnostics["raw"]["reason"] == "range_evidence_unverified"


def test_raw_low_activity_does_not_silently_replace_stable_trend() -> None:
    """The accepted trend remains visible until transition confirmation."""
    manager = MarketRegimeManager(MarketRegimeDetector(), transition_confirmations=3)
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=0.8,
            reason="initial",
            updated_at=time.time(),
            adjustments={},
        )
    )
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="low_activity",
            confidence=0.65,
            reason="low_volume_flat_trend_confirmed",
            updated_at=time.time(),
            adjustments={},
        )
    )
    diagnostics = manager.build_diagnostics()
    assert diagnostics["current"]["regime"] == "trend"
    assert diagnostics["raw"]["regime"] == "low_activity"
    assert diagnostics["transition"]["pending_count"] == 1
