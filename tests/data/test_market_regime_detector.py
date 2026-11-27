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


def test_low_atr_alone_cannot_authorize_a_tradeable_range() -> None:
    detector = MarketRegimeDetector(RegimeParameters(atr_range_threshold=8.0))
    snapshot = detector.evaluate("NIFTY", {"atr": 4.0})
    assert snapshot.regime == "unknown"
    assert snapshot.reason == "range_evidence_unverified"
    assert snapshot.confidence == 0.0

    adx_only = detector.evaluate("NIFTY", {"atr": 4.0, "adx": 10.0})
    assert adx_only.regime == "unknown"


def test_low_adx_with_independent_flat_price_history_proves_range() -> None:
    detector = MarketRegimeDetector(RegimeParameters(atr_range_threshold=8.0))
    snapshot = detector.evaluate(
        "NIFTY",
        {"atr": 4.0, "adx": 10.0, "price_series": [100.0, 100.02, 100.01, 100.0]},
    )
    assert snapshot.regime == "range"
    assert snapshot.adjustments["bias"] == "neutral"
    assert snapshot.adjustments["display_regime"] == "ranging"


def test_low_adx_with_converged_emas_proves_range_without_history() -> None:
    detector = MarketRegimeDetector(RegimeParameters(atr_range_threshold=8.0))
    snapshot = detector.evaluate(
        "NIFTY", {"atr": 4.0, "adx": 10.0, "ema_fast": 100.01, "ema_slow": 100.0}
    )
    assert snapshot.regime == "range"


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


def test_missing_regime_features_are_unknown_not_range() -> None:
    detector = MarketRegimeDetector(RegimeParameters())

    snapshot = detector.evaluate("NIFTY", {"price": 22750.0})

    assert snapshot.regime == "unknown"
    assert snapshot.confidence == 0.0
    assert snapshot.reason == "insufficient_regime_features"


def test_quiet_tape_requires_low_volume_and_flat_trend_proof() -> None:
    detector = MarketRegimeDetector(RegimeParameters())
    quiet = detector.evaluate(
        "NIFTY",
        {"adx": 10.0, "volume_ratio": 0.4, "price_series": [100.0, 100.01]},
    )
    assert quiet.regime == "low_activity"
    assert quiet.reason == "low_volume_flat_trend_confirmed"
    assert quiet.adjustments["display_regime"] == "low activity"
    assert quiet.adjustments["sizing_multiplier"] <= 1.0

    # Low volume without independent trend/momentum evidence is not enough.
    insufficient = detector.evaluate("NIFTY", {"volume_ratio": 0.4})
    assert insufficient.regime != "low_activity"


def test_choppy_and_normal_labels_do_not_authorize_mean_reversion() -> None:
    from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime

    assert normalize_regime("CHOPPY") is MarketRegime.UNKNOWN
    assert normalize_regime("NORMAL") is MarketRegime.UNKNOWN


def test_canonical_adx_units_and_volume_are_not_directional_proof() -> None:
    detector = MarketRegimeDetector()
    quiet = detector.evaluate(
        "NIFTY",
        {"adx": 0.75, "atr": 4.0, "price_series": [25000.0, 25000.5, 25000.0]},
    )
    assert quiet.regime == "range"
    assert detector.evaluate("NIFTY", {"volume_ratio": 1.2}).regime == "unknown"


def test_nonfinite_and_boolean_features_cannot_manufacture_regime() -> None:
    detector = MarketRegimeDetector()
    for features in (
        {"adx": float("inf"), "atr": float("nan")},
        {"adx": True, "atr": False},
        {"price_series": [25000.0, float("nan"), 25500.0]},
    ):
        assert detector.evaluate("NIFTY", features).regime == "unknown"


def test_independent_volatility_hazard_cannot_be_hidden_by_trend_score() -> None:
    from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager

    detector = MarketRegimeDetector()
    manager = MarketRegimeManager(detector)
    snapshot = detector.evaluate(
        "NIFTY",
        {
            "adx": 60.0,
            "price_series": [25000.0, 25300.0],
            "realized_volatility": 100.0,
        },
    )
    assert snapshot.regime == "trend"
    assert snapshot.adjustments["volatility_risk_confidence"] >= 0.95
    assert not manager.can_trade()
    assert "regime_block_independent_volatile" in manager.get_filter_reasons()


def test_normalised_atr_cannot_weaken_original_volatility_protection() -> None:
    detector = MarketRegimeDetector()
    for price, expected in ((20000.0, 25.0), (30000.0, 20.0)):
        snapshot = detector.evaluate(
            "NIFTY", {"price": price, "atr": 20.0, "completed_bar_timestamp": 60.0}
        )
        assert snapshot.adjustments["effective_atr_reference_points"] == expected
