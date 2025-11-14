import time

from nifty_scalper_bot.core.market_regime import RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.data.market_regime import MarketRegimeDetector


def _make_snapshot(regime: str, confidence: float) -> RegimeSnapshot:
    return RegimeSnapshot(
        symbol="NIFTY",
        regime=regime,
        confidence=confidence,
        reason="unit-test",
        updated_at=time.time(),
        adjustments={},
    )


def test_market_regime_detector_ingest_tick_produces_snapshot() -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)

    tick = {
        "symbol": "NIFTY",
        "regime": "trend",
        "confidence": 0.71,
        "reason": "unit-test",
    }

    snapshot = detector.ingest_tick(tick)
    assert snapshot is not None

    latest = manager.get_latest_snapshot()
    assert latest is not None
    assert latest.regime == "trend"
    assert latest.confidence == snapshot.confidence


def test_market_regime_manager_tracks_snapshot() -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)
    snapshot = _make_snapshot("trend", 0.75)
    manager.ingest_snapshot(snapshot)

    assert manager.get_current_regime() == "trend"
    assert manager.get_regime_confidence() == snapshot.confidence
    history = manager.get_history(limit=1)
    assert history and history[-1].regime == "trend"


def test_market_regime_manager_blocks_high_risk_regime() -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)
    manager.ingest_snapshot(_make_snapshot("event", 0.82))

    allowed = manager.can_trade()
    assert not allowed
    reasons = manager.get_filter_reasons()
    assert any("regime_block_event" in reason for reason in reasons)
    stats = manager.get_regime_filter_stats()
    assert stats.get("block") == 1


def test_market_regime_manager_bypass_overrides_block() -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)
    manager.ingest_snapshot(_make_snapshot("event", 0.9))
    manager.set_regime_filter_bypass(True)

    allowed = manager.can_trade()
    assert allowed
    assert manager.get_regime_filter_bypass() is True
    stats = manager.get_regime_filter_stats()
    assert stats.get("bypass") == 1


def test_market_regime_manager_builds_diagnostics() -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)
    snapshot = _make_snapshot("trend", 0.65)
    manager.ingest_snapshot(snapshot)
    manager.can_trade()

    diagnostics = manager.build_diagnostics()
    assert diagnostics["current"]["regime"] == "trend"
    assert isinstance(diagnostics.get("stats"), dict)
    assert diagnostics.get("history")


def test_market_regime_manager_fail_closed_blocks_on_error() -> None:
    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector, fail_closed=True)

    class ExplodingLock:
        def __init__(self) -> None:
            self.calls = 0

        def __enter__(self) -> None:  # noqa: D401 - test stub
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")

        def __exit__(self, *args: object) -> bool:  # noqa: D401 - test stub
            return False

    manager._lock = ExplodingLock()  # type: ignore[assignment]

    allowed = manager.can_trade()
    assert allowed is False
    reasons = manager.get_filter_reasons()
    assert "regime_fail_closed_error" in reasons
