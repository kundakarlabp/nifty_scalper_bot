from nifty_scalper_bot.strategies.market_regime_engine import MarketRegime
from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_vwap_high_vol_soft_allow_uses_candidate_metadata() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._strategy_allowed_for_regime = lambda *_args, **_kwargs: False
    allowed, reason = runner._strategy_regime_decision(
        strategy='VWAPPro',
        regime=MarketRegime.VOLATILE,
        symbol='NIFTY',
        metadata={'candidate_selected': True, 'candidate_spread_pct': 0.4, 'candidate_rr': 1.8},
    )
    assert allowed is True
    assert reason == 'vwap_high_vol_execution_quality_soft_allow'
