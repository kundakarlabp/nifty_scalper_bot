from nifty_scalper_bot.data.market_regime import MarketRegimeDetector, RegimeParameters


def test_market_regime_detector_identifies_trend() -> None:
    detector = MarketRegimeDetector(RegimeParameters())
    snapshot = detector.evaluate(
        "NIFTY",
        {"trend_score": 0.75, "ema_fast": 102.0, "ema_slow": 100.0},
        {"atr": 12.0},
    )
    assert snapshot.regime == "trend"
    assert snapshot.adjustments["bias"] == "trend"
    assert snapshot.adjustments["display_regime"] == "trending"


def test_market_regime_detector_range_on_low_atr() -> None:
    detector = MarketRegimeDetector(RegimeParameters(atr_range_threshold=8.0))
    snapshot = detector.evaluate("NIFTY", {"atr": 4.0}, {"atr": 4.0})
    assert snapshot.regime == "range"
    assert snapshot.adjustments["bias"] == "neutral"
    assert snapshot.adjustments["display_regime"] == "ranging/choppy"


def test_market_regime_detector_broadcasts_events() -> None:
    detector = MarketRegimeDetector(RegimeParameters())
    captured: list[str] = []
    queue_ref = detector.register_queue()
    detector.subscribe(lambda snap: captured.append(snap.regime))
    detector.evaluate("NIFTY", {"atr": 45.0}, {"atr": 45.0})
    assert captured[-1] == "event"
    queue_snapshot = queue_ref.get()
    assert queue_snapshot.regime == "event"
    assert queue_snapshot.adjustments["display_regime"] == "event-driven"


def test_market_regime_detector_snapshot_fallback() -> None:
    detector = MarketRegimeDetector(RegimeParameters())
    snapshot = detector.evaluate(
        "NIFTY",
        {"volume_ratio": 1.3},
        {
            "ema_fast": 105.0,
            "ema_slow": 100.0,
            "trend_score": 0.72,
            "volume_ratio": 1.3,
        },
    )
    assert snapshot.regime == "trend"
    assert snapshot.confidence >= 0.6
    assert snapshot.adjustments["display_regime"] == "trending"


def test_market_regime_detector_uses_price_momentum() -> None:
    detector = MarketRegimeDetector(RegimeParameters(price_momentum_trend=0.002))
    snapshot = detector.evaluate(
        "NIFTY",
        {
            "price_series": [100.0, 100.5, 101.4],
            "volume_ratio": 1.1,
        },
        None,
    )
    assert snapshot.regime == "trend"
    assert snapshot.adjustments["display_regime"] == "trending"
    assert snapshot.reason in {
        "momentum_trending",
        "adx_momentum_alignment",
        "ema_divergence",
    }


def test_market_regime_detector_realized_volatility_flags_event() -> None:
    detector = MarketRegimeDetector(RegimeParameters(iv_realized_gap_event=10.0))
    snapshot = detector.evaluate(
        "NIFTY",
        {
            "iv_rank": 85.0,
            "realized_volatility": 0.3,
        },
        None,
    )
    assert snapshot.regime == "event"
    assert snapshot.reason == "iv_realized_gap"
    assert snapshot.adjustments["display_regime"] == "event-driven"


def test_market_regime_detector_realized_volatility_flags_volatile() -> None:
    detector = MarketRegimeDetector(
        RegimeParameters(realized_vol_volatile_threshold=28.0)
    )
    snapshot = detector.evaluate(
        "NIFTY",
        {
            "realized_volatility": 0.35,
            "volume_ratio": 1.0,
        },
        None,
    )
    assert snapshot.regime == "volatile"
    assert snapshot.adjustments["display_regime"] == "volatile"
    assert snapshot.reason in {
        "realized_vol_high",
        "atr_above_volatile",
        "momentum_spike",
    }
