from types import SimpleNamespace

from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.strategies.strategy_config import StrategyConfig


def test_iv_extreme_blocks_when_score_low(monkeypatch):
    s = EnhancedScalpingStrategy()
    cfg = StrategyConfig.load("config/strategy.yaml")
    s.runner = SimpleNamespace(strategy_cfg=cfg)
    plan = {"regime": "TREND", "score": 8, "adx": 25}
    monkeypatch.setattr(EnhancedScalpingStrategy, "_est_iv_pct", lambda self, S, K, T: 90)
    reason = s._iv_adx_reject_reason(plan, close=100.0)
    assert reason and reason[0] == "iv_extreme"
