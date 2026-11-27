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
async def test_verified_completed_session_history_feeds_range_momentum(
    monkeypatch,
) -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from nifty_scalper_bot.core import market_regime_manager as regime_module

    session_now = datetime(2026, 11, 20, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata"))

    class FixedSessionDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return session_now.astimezone(tz) if tz is not None else session_now

    monkeypatch.setattr(regime_module, "datetime", FixedSessionDateTime)

    class HistoryIndicators:
        def get_indicators(self, symbol, names=None):
            return {"adx": 10.0, "atr": 4.0, "volume_spike_ratio": 1.0}

        def get_history(self, symbol, count=None, *, field="close"):
            assert field == "bars" and count == 16
            last = session_now.replace(second=0, microsecond=0) - timedelta(minutes=1)
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


def test_raw_low_activity_blocks_while_stable_trend_is_held() -> None:
    manager = MarketRegimeManager(MarketRegimeDetector(), transition_confirmations=3)
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=0.8,
            reason="confirmed_trend",
            updated_at=time.time(),
            adjustments={},
        )
    )
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="low_activity",
            confidence=0.75,
            reason="current_low_activity",
            updated_at=time.time(),
            adjustments={},
        )
    )

    assert manager.get_latest_snapshot().regime == "trend"
    assert manager.get_raw_snapshot().regime == "low_activity"
    assert manager.get_execution_snapshot().regime == "low_activity"
    assert manager.can_trade(record_decision=False) is False
    assert "regime_low_activity" in manager.get_filter_reasons()


def test_raw_event_threshold_causes_immediate_safety_transition() -> None:
    manager = MarketRegimeManager(MarketRegimeDetector(), transition_confirmations=3)
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=0.8,
            reason="confirmed_trend",
            updated_at=time.time(),
            adjustments={},
        )
    )
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="event",
            confidence=0.9,
            reason="fresh_event",
            updated_at=time.time(),
            adjustments={},
        )
    )

    assert manager.get_latest_snapshot().regime == "event"
    assert manager.get_raw_snapshot().regime == "event"
    assert manager.can_trade(record_decision=False) is False
    assert any(
        reason.startswith("regime_block_") for reason in manager.get_filter_reasons()
    )


@pytest.mark.parametrize("invalid_time", ("future", "nan"))
def test_invalid_raw_regime_timestamp_fails_closed(invalid_time: str) -> None:
    manager = MarketRegimeManager(MarketRegimeDetector())
    timestamp = time.time() + 3600.0 if invalid_time == "future" else float("nan")
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=0.8,
            reason="unverified_clock",
            updated_at=timestamp,
            adjustments={},
        )
    )

    assert manager.can_trade(record_decision=False) is False
    assert "regime_timestamp_invalid" in manager.get_filter_reasons()


def test_nonfinite_raw_regime_confidence_fails_closed() -> None:
    manager = MarketRegimeManager(MarketRegimeDetector())
    manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="NIFTY",
            regime="trend",
            confidence=float("nan"),
            reason="unverified_confidence",
            updated_at=time.time(),
            adjustments={},
        )
    )

    assert manager.can_trade(record_decision=False) is False
    assert "confidence_invalid" in manager.get_filter_reasons()


def test_directional_transition_counts_only_adjacent_completed_bars() -> None:
    manager = MarketRegimeManager(MarketRegimeDetector(), transition_confirmations=2)
    now = time.time()
    first = float(int(now // 60) * 60 - 180)
    manager.ingest_snapshot(RegimeSnapshot("NIFTY", "trend", 0.8, "baseline", now))
    for i, bar_at in enumerate((first, first, first - 60, first + 120)):
        manager.ingest_snapshot(
            RegimeSnapshot(
                "NIFTY",
                "range",
                0.8,
                "candidate",
                now + i * 0.01,
                {"completed_bar_timestamp": bar_at},
            )
        )
        assert manager.get_current_regime() == "trend"
    manager.ingest_snapshot(
        RegimeSnapshot(
            "NIFTY",
            "range",
            0.8,
            "confirmed",
            now + 0.1,
            {"completed_bar_timestamp": first + 180},
        )
    )
    assert manager.get_current_regime() == "range"


def test_missing_candle_identity_cannot_confirm_directional_transition() -> None:
    manager = MarketRegimeManager(MarketRegimeDetector(), transition_confirmations=2)
    now = time.time()
    manager.ingest_snapshot(RegimeSnapshot("NIFTY", "trend", 0.8, "baseline", now))
    for offset in (0.01, 0.02, 0.03):
        manager.ingest_snapshot(
            RegimeSnapshot(
                "NIFTY",
                "range",
                0.8,
                "no completed evidence",
                now + offset,
                {"completed_bar_timestamp": 0.0},
            )
        )
    assert manager.get_current_regime() == "trend"
    manager.ingest_snapshot(RegimeSnapshot("NIFTY", "event", 0.99, "shock", now + 0.04))
    assert manager.get_current_regime() == "event"


def test_live_regime_requires_completed_evidence_and_reconfirms_after_gap(
    monkeypatch,
) -> None:
    from nifty_scalper_bot.core import market_regime_manager as regime_module

    clock = [float(int(time.time() // 60) * 60 + 15)]
    monkeypatch.setattr(regime_module.time, "time", lambda: clock[0])
    manager = MarketRegimeManager(
        MarketRegimeDetector(),
        regime_settings={"symbol": "NSE:NIFTY", "require_completed_evidence": True},
    )

    def ingest(*, stamp=None, symbol="NSE:NIFTY", regime="trend"):
        adjustments = {} if stamp is None else {"completed_bar_timestamp": stamp}
        manager.ingest_snapshot(
            RegimeSnapshot(symbol, regime, 0.8, "test", clock[0], adjustments)
        )

    ingest()
    assert manager.get_current_regime() is None
    assert not manager.can_trade()
    manager.set_regime_filter_bypass(True)
    assert not manager.can_trade()
    manager.set_regime_filter_bypass(False)
    first = clock[0] - 75
    ingest(stamp=first)
    clock[0] += 1
    ingest(stamp=first)
    assert not manager.can_trade()
    clock[0] += 59
    ingest(stamp=first + 60)
    assert manager.can_trade(
        expected_regime="TREND", expected_completed_bar_timestamp=first + 60
    )
    assert not manager.can_trade(
        expected_regime="RANGE", expected_completed_bar_timestamp=first + 60
    )
    assert not manager.can_trade(
        expected_regime="TREND", expected_completed_bar_timestamp=first
    )
    ingest(stamp=first + 60, symbol="NFO:OTHER", regime="event")
    assert manager.can_trade()
    clock[0] += 120
    ingest(stamp=first + 180)
    assert not manager.can_trade()
    clock[0] += 60
    ingest(stamp=first + 240)
    assert manager.can_trade()
    ingest(stamp=first + 180, regime="range")
    assert manager.can_trade(
        expected_regime="TREND", expected_completed_bar_timestamp=first + 240
    )
    clock[0] += 181
    assert not manager.can_trade()


def test_completed_regime_inputs_exclude_provisional_and_unverified_volume(
    monkeypatch,
) -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from nifty_scalper_bot.strategies.indicators import IndicatorEngine

    engine = IndicatorEngine()
    start = datetime(2026, 11, 27, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
    rows = []
    for i in range(65):
        close = 25000.0 + 1.25 * i
        rows.append(
            {
                "timestamp": start + timedelta(minutes=i),
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 1000,
                "is_complete": True,
                "is_provisional": False,
            }
        )
    now = start + timedelta(minutes=65, seconds=20)
    monkeypatch.setattr(engine, "get_history", lambda *args, **kwargs: list(rows))
    baseline = engine.get_completed_regime_inputs("NSE:NIFTY", now=now)
    assert baseline["verified_completed_bars"] == 65
    assert baseline["broader_60bar_directional_efficiency"] == 1
    assert baseline["adx"] == 100.0
    assert MarketRegimeDetector().evaluate("NSE:NIFTY", baseline).regime == "trend"
    assert "volume_spike_ratio" not in baseline
    assert (
        engine.get_completed_regime_inputs("NFO:NIFTY26NOVFUT", now=now)[
            "volume_spike_ratio"
        ]
        == 1
    )
    rows[-1]["volume"] = 0
    assert "volume_spike_ratio" not in engine.get_completed_regime_inputs(
        "NFO:NIFTY26NOVFUT", now=now
    )
    rows.append(dict(rows[-1], timestamp=now, close=1e9, is_provisional=True))
    assert engine.get_completed_regime_inputs("NSE:NIFTY", now=now) == baseline
    rows.pop()
    missing = rows.pop(-3)
    assert engine.get_completed_regime_inputs("NSE:NIFTY", now=now) == {}
    rows.append(missing)
    rows.append(dict(rows[-1], close=rows[-1]["close"] + 0.5))
    assert engine.get_completed_regime_inputs("NSE:NIFTY", now=now) == {}


def test_fresh_intrabar_shock_latches_until_completed_observation(monkeypatch) -> None:
    from types import SimpleNamespace

    from nifty_scalper_bot.core import market_regime_manager as regime_module

    clock = [float(int(time.time() // 60) * 60 + 15)]
    monkeypatch.setattr(regime_module.time, "time", lambda: clock[0])
    quote = {"symbol": "NSE:NIFTY", "ltp": 25000.0}

    def get_quote(_symbol, *, allow_pull):
        assert allow_pull is False
        return quote

    manager = MarketRegimeManager(
        MarketRegimeDetector(),
        datahub=SimpleNamespace(get_quote=get_quote),
        regime_settings={"symbol": "NSE:NIFTY", "require_completed_evidence": True},
    )

    def ingest(stamp):
        manager.ingest_snapshot(
            RegimeSnapshot(
                "NSE:NIFTY", "trend", 0.8, "completed", clock[0],
                {
                    "completed_bar_timestamp": stamp,
                    "completed_reference_price": 25000.0,
                    "intrabar_event_move_points": 35.0,
                },
            )
        )

    first = clock[0] - 75
    ingest(first)
    clock[0] += 60
    ingest(first + 60)
    quote["timestamp_ms"] = clock[0] * 1000
    assert manager.can_trade()
    quote.update(ltp=25040.0, timestamp_ms=(clock[0] - 5) * 1000)
    assert manager.can_trade()
    quote.update(symbol="NFO:OTHER", timestamp_ms=clock[0] * 1000)
    assert manager.can_trade()
    quote.update(symbol="NSE:NIFTY")
    assert not manager.can_trade()
    assert "regime_block_intrabar_spot_shock" in manager.get_filter_reasons()
    assert manager.get_current_regime() == "trend"
    quote.update(ltp=25000.0)
    assert not manager.can_trade()
    clock[0] += 60
    ingest(first + 120)
    quote["timestamp_ms"] = clock[0] * 1000
    assert manager.can_trade()
