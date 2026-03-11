from nifty_scalper_bot.strategies.market_regime_engine import MarketRegime, MarketRegimeEngine


def test_market_regime_trend_classification() -> None:
    engine = MarketRegimeEngine()
    snapshot = engine.classify(
        {
            'adx': 28.0,
            'atr': 10.0,
            'atr_average': 8.0,
            'vwap_slope': 0.2,
            'volume_expansion': 1.2,
        }
    )
    assert snapshot.regime == MarketRegime.TREND


def test_market_regime_volatile_classification() -> None:
    engine = MarketRegimeEngine()
    snapshot = engine.classify(
        {
            'adx': 18.0,
            'atr': 20.0,
            'atr_average': 10.0,
            'vwap_slope': 0.0,
            'volume_expansion': 1.0,
        }
    )
    assert snapshot.regime == MarketRegime.VOLATILE
