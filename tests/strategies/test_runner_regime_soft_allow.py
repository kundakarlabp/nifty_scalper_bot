from nifty_scalper_bot.strategies.market_regime_engine import MarketRegime
from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_vwap_high_vol_soft_allow(monkeypatch):
    monkeypatch.setenv('RUNNER_VWAP_ALLOWED_REGIMES', 'TREND,NORMAL')
    runner = StrategyRunner.__new__(StrategyRunner)
    allowed, reason = runner._strategy_regime_decision(strategy='VWAPPro', regime=MarketRegime.VOLATILE, symbol='NIFTY', metadata={'candidate_selected': True, 'candidate_spread_pct': 0.5, 'candidate_rr': 1.8})
    assert allowed is True
    assert reason == 'vwap_high_vol_execution_quality_soft_allow'


def test_vwap_high_vol_soft_allow_fails_with_high_spread(monkeypatch):
    monkeypatch.setenv('RUNNER_VWAP_ALLOWED_REGIMES', 'TREND,NORMAL')
    runner = StrategyRunner.__new__(StrategyRunner)
    allowed, reason = runner._strategy_regime_decision(strategy='VWAPPro', regime=MarketRegime.VOLATILE, symbol='NIFTY', metadata={'candidate_selected': True, 'candidate_spread_pct': 1.1, 'candidate_rr': 1.8})
    assert allowed is False
    assert reason == 'vwap_high_vol_execution_quality_failed'
