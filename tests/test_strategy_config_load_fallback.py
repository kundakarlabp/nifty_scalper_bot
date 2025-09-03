from src.strategies.strategy_config import StrategyConfig

def test_strategy_config_load_fallback(monkeypatch):
    monkeypatch.setenv("STRATEGY_CONFIG_FILE", "config/astrategy.yaml")
    cfg = StrategyConfig.load("config/astrategy.yaml")
    assert cfg.source_path == "config/strategy.yaml"

