from types import SimpleNamespace

from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.strategies.strategy_config import StrategyConfig


def test_weak_trend_blocks_plan():
    s = EnhancedScalpingStrategy()
    cfg = StrategyConfig.load("config/strategy.yaml")
    s.runner = SimpleNamespace(strategy_cfg=cfg)
    plan = {"regime": "TREND", "score": 9, "adx": 12}
    reason = s._iv_adx_reject_reason(plan, close=100.0)
    assert reason and reason[0] == "weak_trend"
